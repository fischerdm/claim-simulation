# Benchmark Study

This document describes the benchmark study design, the session-loading constraint,
how to calibrate expected runtimes, and how many simulations are sufficient for
different actuarial use cases.

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

Each Rayon worker thread owns one ONNX session, **loaded lazily on first use** (not at
startup). Simulations are distributed across threads via work-stealing with no lock
contention between threads once sessions are warm.

```
n_threads worker threads (one per core)
│
├── Thread 0: sims 0, 1, ..., n_sims/n_threads - 1
│   ├── [first sim only] load ONNX session  ← lazy, ~25–30 s one-time cost
│   └── for each sim:
│       └── for year in 0..n_years:
│           ├── build [N, 9] feature matrix
│           ├── run_inference(matrix) → λ per policy   ← ONNX call
│           ├── draw Poisson(λ × exposure) per policy
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

## The session-loading constraint

ONNX Runtime holds a global lock during session initialisation. Even though Rayon
dispatches all threads simultaneously, sessions load **sequentially**, costing roughly:

```
T_startup ≈ n_threads × 25 s   (observed ~25 s/session on macOS Intel)
```

This matters in two ways:

1. **Small portfolios are startup-dominated.** For 50–100 policies the actual simulation
   compute takes milliseconds — far less than the session loading overhead. The quick-test
   results (`QUICK_TEST=1`) essentially measure startup time only and cannot be used to
   extrapolate compute throughput.

2. **Linear core scaling breaks down for high core counts.** More cores reduce per-thread
   sim count (good), but also increase total session loading time (bad). The optimal
   thread count balances these two effects:

   ```
   T_total(k) = k × T_session + (n_sims / k) × n_years × T_inference(N)
   ```

   Optimal k ≈ √(n_sims × n_years × T_inference(N) / T_session)

   For production workloads (678K policies, 2500 sims, 5 years) — using AWS numbers
   once the calibration run is complete (plug in your observed T_inference and T_session):
   ```
   k* ≈ √(2500 × 5 × T_inference(678K) / T_session)
   ```
   On macOS Intel (T_inference(10K)≈1 s → T_inference(678K)≈67 s, T_session≈24 s),
   k* ≈ √(2500 × 5 × 67 / 24) ≈ 187 — but that machine is not a useful reference.
   On a fast Linux box (T_inference likely 5–20× lower), k* will be 10–40 threads.

---

## Calibration run (recommended before capacity planning)

The quick test is too small to measure compute throughput. Before estimating AWS costs,
run one compute-dominated measurement on your local machine:

```bash
ORT_DYLIB_PATH=.venv/lib/python3.12/site-packages/onnxruntime/capi/libonnxruntime.1.23.2.dylib \
  rust/target/release/claim-simulation --fraction 1.0 --n-sims 500 --years 5
```

This runs the full 10K-policy portfolio for 500 sims × 5 years. From the output:

```
T_total  = reported wall time
T_startup = n_threads × 25 s          (or measure separately with --n-sims 1)
T_compute = T_total - T_startup

throughput = (10_000 × 500 × 5) / T_compute   # policy-sim-years per second
```

With `throughput` you can estimate any (N, n_sims, n_years, n_threads) combination:

```
T_wall ≈ n_threads × T_session + (N × n_sims × n_years) / (throughput × n_threads)
```

> **Note:** session loading time on Linux (AWS) will differ from macOS. Once you have
> one real AWS data point, replace `T_session` with the observed value.

---

## How many simulations are enough?

The required simulation count depends on which statistic you care about. The standard
error on a quantile estimate at level p with n simulations is approximately
`SE ≈ √(p(1−p)) / (n × f(Qₚ))` where f(Qₚ) is the density at the quantile.

| Use case | Target statistic | Recommended n_sims |
|---|---|---|
| Pricing / expected loss | Mean frequency | 500–1,000 |
| Reserving, confidence intervals | P95 | 1,000–2,000 |
| Capital / risk margin | P99 | 2,000–5,000 |
| Solvency II / regulatory capital | P99.5 (SCR) | 5,000–10,000 |

**Default recommendation: 2,500 sims.** This gives reliable P99 estimates
(≈ 0.2% SE in probability space) at a reasonable compute cost for most pricing and
reserving tasks. If you specifically need Solvency II–grade tail estimates, use 5,000+.

Rule of thumb: run a pilot with 500 sims, then double until your P99 estimate stabilises
to within 1% relative change. For this dataset (mean frequency ≈ 10%), 2,500 sims is
typically sufficient.

---

## Runtime estimates

> **Status: AWS numbers TBD.** The table below will be filled in after the calibration
> run on AWS (see *Calibration run* above). The macOS Intel numbers are observed; the
> AWS columns are placeholders.

### Observed: macOS Intel (calibration run, 2026-03-07)

| Run | Policies | n_sims | n_years | Cores | T_total | T_startup | T_compute | T_inference/call |
|---|---|---|---|---|---|---|---|---|
| Calibration | 10,000 | 500 | 5 | 8 | 502 s | ~195 s | ~307 s | ~1 s |

- **T_inference(10K policies) ≈ 1 s/call** on macOS Intel (100 µs/policy).
  This is ~300× slower than the theoretical 300 ns/policy — consistent with an aging
  Intel Mac without AVX-512 and with macOS memory bandwidth limitations. Modern Linux
  instances (AVX-512, larger caches) should be substantially faster.
- T_startup ≈ 24 s/session, loading sequentially due to ONNX Runtime global lock.

> **Note on portfolio size:** `data/portfolio_v2.csv` contains **10,000 policies**
> (a stratified sample). The full 678K-policy benchmark requires re-running
> `python/export_portfolio.py` without a size cap to generate the full portfolio file.

### Wall-clock estimates — 2,500 sims × 5 years × 678K policies

Numbers below use the formula `T_wall ≈ k × T_session + (N × n_sims × n_years) / (throughput × k)`.
The throughput column is the key unknown; **fill in after the AWS calibration run**.

| Machine | Cores (k) | T_session | Throughput (policies/s/thread) | T_total (est.) | On-demand $/hr |
|---|---|---|---|---|---|
| Intel MacBook (dev) | 8 | ~24 s | ~10,000 | ~6 h | — |
| AWS c6i.2xlarge | 8 | TBD | TBD | TBD | $0.34 |
| AWS c6i.4xlarge | 16 | TBD | TBD | TBD | $0.68 |
| AWS c6i.8xlarge | 32 | TBD | TBD | TBD | $1.36 |
| AWS c6i.16xlarge | 64 | TBD | TBD | TBD | $2.72 |

> Spot instances reduce on-demand price by ~70% for interruptible workloads.
> The structural insight still holds: beyond the optimal thread count
> (`k* ≈ √(n_sims × n_years × T_inference(N) / T_session)`), session loading grows
> faster than compute shrinks — cost efficiency degrades sharply at high core counts.

---

## Running locally

### Quick test (pipeline validation only)

```bash
QUICK_TEST=1 python python/benchmark.py
```

Uses a tiny grid (0.5% / 1% of portfolio, 200 / 500 sims). Validates the full
pipeline end-to-end. **Not suitable for extrapolating compute throughput** — these
tiny portfolios are dominated by ONNX session loading time (~25 s/thread), not actual
simulation compute.

### Calibration run (throughput measurement)

```bash
# Run the Rust binary directly — no Python overhead, full 10K portfolio
ORT_DYLIB_PATH=... rust/target/release/claim-simulation \
  --fraction 1.0 --n-sims 500 --years 5
```

Use the output to estimate compute throughput (see *Calibration run* section above).

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
column. Designed to stack results from multiple machines for cross-instance comparison.
