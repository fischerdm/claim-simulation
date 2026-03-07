# Benchmark Study

This document describes the benchmark study design, scaling grids, and expected
runtimes for the two studies run by `python/benchmark.py`.

## What we are measuring

Non-life actuarial pricing requires two operations:

1. **Inference (single-year):** given a portfolio of N policies, predict the expected
   claim frequency λ for each policy. In practice this is a point estimate; simulation
   is only needed once severity is layered on top (to compute technical price distributions).
   *Question: should you use LightGBM `predict()` or export to ONNX for batch inference?*

2. **Recursive simulation (multi-year):** project the portfolio forward over T years.
   Each year, λ is recomputed because `PriorClaims3Y` — a rolling 3-year claim count —
   changes as simulated claims accumulate. There is no closed-form alternative; simulation
   is the only option.
   *Question: how does the Rust + ONNX engine scale with portfolio size and sim count?*

Both studies use the **v2 model** (`frequency_model_v2.onnx / .lgb`):

| Feature | Description |
|---|---|
| VehPower | Engine power (clipped at 15) |
| VehAge | Vehicle age in years |
| DrivAge | Driver age in years |
| Density | Population density of driver's municipality |
| **PriorClaims3Y** | Rolling 3-year claim count — updated each simulation year |
| Area | Area category A–F (label-encoded) |
| VehBrand | Vehicle brand (label-encoded) |
| VehGas | Fuel type Diesel/Regular (label-encoded) |
| Region | French administrative region (label-encoded) |

BonusMalus is excluded: it cannot be projected forward without a full BM transition model.

---

## Study 1 — Inference: LightGBM vs ONNX Runtime

### What is measured

Both engines receive the same `[N, 9]` float32 feature matrix and return λ per policy.
No simulation is performed. Each measurement is the minimum of 3 repetitions to suppress
OS scheduling jitter; a warmup call is made first to exclude library-init overhead.

### Scaling grid

| Dimension | Values |
|---|---|
| Portfolio fraction | 25% / 50% / 100% of `portfolio_v2.csv` |
| Engine | `lgb.Booster.predict()` vs `onnxruntime.InferenceSession.run()` |

### Expected result

ONNX Runtime is typically 1.5–3× faster than LightGBM `predict()` for batch inference.
LightGBM's Python binding adds interpreter overhead per call; ONNX Runtime compiles the
tree ensemble to an optimised execution plan that avoids it. The gap grows with batch size.

---

## Study 2 — Simulation: Rust + ONNX scaling

### Architecture

Each Rayon worker thread owns one ONNX session (loaded at startup).
Simulations are distributed across threads — no lock contention, no serialisation.

```
n_threads worker threads (one per core)
│
├── Thread 0: sims 0, 1, ..., n_sims/n_threads - 1
│   └── for each sim:
│       └── for year in 0..n_years:
│           ├── build [N, 9] feature matrix
│           ├── run_inference(matrix) → λ per policy   ← ONNX (single-threaded)
│           ├── draw Poisson(λ × exposure) per policy  ← Rayon parallel
│           └── shift rolling 3-year claim window
├── Thread 1: sims n_sims/n_threads, ...
└── ...
```

**Single-year (n_years=1):** ONNX is called once per simulation. With a fixed initial
`PriorClaims3Y` from the portfolio, year-0 lambdas are the same across simulations, so
only the Poisson draws differ. Runtime is dominated by the sampling loop.

**Multi-year (n_years=5):** ONNX is called once per year per simulation (5× heavier).
From year 1 onward each simulation has a distinct `PriorClaims3Y` (because the Poisson
draws in year 0 differ), so inference cannot be hoisted out of the sim loop.

### Scaling grid

| Dimension | Values |
|---|---|
| Portfolio fraction | 25% (~170K) / 50% (~340K) / 100% (~678K) of `portfolio_v2.csv` |
| Simulations | 2,000 / 5,000 / 10,000 |
| Projection years | 1 (single-year) / 5 (multi-year) |

3 × 3 × 2 = **18 cells**.

---

## Runtime estimates

Rough estimates based on ~300 ns per policy per ONNX call on a single thread
(LightGBM ensemble, ~400 trees, 63 leaves).

### Per-simulation cost

| Fraction | Policies | n_years=1 | n_years=5 |
|---|---|---|---|
| 25% | ~170K | ~50 ms | ~250 ms |
| 50% | ~340K | ~100 ms | ~500 ms |
| 100% | ~678K | ~200 ms | ~1,000 ms |

### Wall-clock estimate (Study 2 only, all 18 cells)

| Machine | Cores | n_years=1 total | n_years=5 total | **Full study** |
|---|---|---|---|---|
| Intel MacBook (4 cores) | 4 | ~15 min | ~60 min | **~75 min** |
| AWS c5.4xlarge | 16 | ~4 min | ~15 min | **~20 min** |
| AWS c5.18xlarge | 72 | ~1 min | ~4 min | **~5 min** |
| AWS c5.metal | 96 phys. | < 1 min | ~3 min | **~4 min** |

Speedup from cores is **nearly linear** — the Rayon parallelism is embarrassingly
parallel over simulations, with zero lock contention between threads.

Study 1 (inference) adds < 2 minutes regardless of machine.

> **Note:** these are order-of-magnitude estimates. Actual ONNX throughput depends on
> tree depth, memory bandwidth, and ONNX Runtime version. Run the quick test first
> (see below) to calibrate.

---

## Running locally

### Quick test (recommended before a full run)

```bash
QUICK_TEST=1 python python/benchmark.py
```

Uses a tiny grid (0.5% / 1% of portfolio, 200 / 500 sims) — completes in **< 1 minute**.
Validates the full pipeline end-to-end without waiting for a long run.

### Full benchmark

```bash
# Build the Rust engine first (one-time; ~30–60 s)
cd rust && cargo build --release && cd ..

# Run the benchmark
python python/benchmark.py
```

Results are saved to `results/benchmark_results.csv` with a `n_cores` column, so
results from different machines can be compared directly.

### Full benchmark on AWS

```bash
# Clone repo, set up environment (see README), then:
cd rust && cargo build --release && cd ..
python python/benchmark.py
```

For large instances (c5.18xlarge+) the full study completes in ~5 minutes.

---

## Output

### Console

```
Study 1 — Inference: LightGBM vs ONNX Runtime
  Policies  Fraction    LightGBM     ONNX RT   Speedup
  --------  --------  ----------  ----------  --------
   169,503      25%      45.2ms      18.3ms    2.47×
   339,007      50%      89.8ms      36.1ms    2.49×
   678,013     100%     179.1ms      72.4ms    2.47×

Study 2 — Simulation: Rust + ONNX  (ms / simulation)

  Single-year (n_years=1)
  Policies  Frac    2,000 sims    5,000 sims   10,000 sims
  --------  ----  ------------  ------------  ------------
   169,503   25%          52.1          51.8          51.9
   ...
```
*(Numbers above are illustrative; actual values depend on hardware.)*

### CSV (`results/benchmark_results.csv`)

One row per (study, engine, fraction, n_sims, n_years) combination, plus a `n_cores`
column. Designed to stack results from multiple machines for cross-instance comparison
in the AWS session.
