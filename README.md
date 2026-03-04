# Claim Simulation

Non-life actuarial frequency model trained in Python (LightGBM + Poisson) and exported to ONNX
for high-speed claim simulation in Rust. The project demonstrates how to speed up a realistic
actuarial Monte Carlo simulation using Rust and Rayon, while keeping the modelling in Python.

Two simulation modes are implemented:

- **v1 — single year:** λ is computed once for all policies; 10,000 simulations draw Poisson(λ)
  in parallel. Classic setup, maximum throughput.
- **v2 — multi-year (5 years):** claim history feeds back into the model each year. After each
  draw, the rolling 3-year claim window updates PriorClaims3Y, and λ is recomputed. Each Rayon
  worker thread owns its own ONNX session to avoid contention.

## Project structure

```
claim-simulation/
├── data/
│   ├── freMTPL2freq.csv                # raw dataset (generated, not in git)
│   ├── freMTPL2freq_with_history.csv   # augmented with synthetic claim history (generated)
│   ├── portfolio.csv                   # v1 portfolio for Rust (generated)
│   ├── portfolio_v2.csv                # v2 portfolio — 10K sampled policies (generated)
│   └── eda/                            # EDA and validation plots (generated)
├── models/
│   ├── frequency_model.lgb             # v1 LightGBM model (generated)
│   ├── frequency_model.onnx            # v1 ONNX export (generated)
│   ├── feature_metadata.json           # v1 feature names and category encodings
│   ├── frequency_model_v2.lgb          # v2 LightGBM model (generated)
│   ├── frequency_model_v2.onnx         # v2 ONNX export (generated)
│   └── feature_metadata_v2.json        # v2 feature names and category encodings
├── python/
│   ├── data/
│   │   └── download.py                 # downloads freMTPL2freq from OpenML
│   ├── generate_history.py             # creates synthetic 3-year claim history
│   ├── train.py                        # trains both v1 and v2 LightGBM models
│   ├── export_onnx.py                  # exports both models to ONNX
│   ├── export_portfolio.py             # exports v1 and v2 portfolio CSVs for Rust
│   ├── validate.py                     # validates LightGBM vs ONNX agreement (v1)
│   ├── eda.py                          # exploratory data analysis, saves plots
│   └── benchmark.py                    # Python simulation baseline for benchmarking
└── rust/
    ├── .cargo/
    │   └── config.toml                 # sets ORT_DYLIB_PATH so cargo run works without extra setup
    ├── src/
    │   ├── main.rs                     # entry point — runs v1 then v2 with timings
    │   ├── model.rs                    # ONNX inference (shared by both simulations)
    │   ├── portfolio.rs                # Policy (single-year) and PolicyMultiYear structs + CSV loaders
    │   ├── simulator.rs                # single-year parallel simulation
    │   └── simulator_multiyear.rs      # multi-year parallel simulation
    └── Cargo.toml
```

## Dataset

[freMTPL2freq](https://www.openml.org/d/41214) — French Motor Third Party Liability frequency data.
678,013 policies with features: vehicle power, age, driver age, bonus-malus, region, etc.
Target: `ClaimNb` (number of claims per policy). Downloaded automatically from OpenML.

## Setup

### macOS prerequisite — OpenMP

LightGBM on macOS requires the OpenMP runtime. Install it with Homebrew:

```bash
brew install libomp
```

Without this, importing `lightgbm` fails with:
```
OSError: dlopen(...lib_lightgbm.dylib): Library not loaded: @rpath/libomp.dylib
```

### Python environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Rust

You need Rust installed (`rustup`). The engine links against the ONNX Runtime library that
ships with the Python `onnxruntime` package — so the Python venv must be set up first.

`rust/.cargo/config.toml` sets `ORT_DYLIB_PATH` automatically, so `cargo run` works without
any additional environment variables. **If you clone this repo on a different machine**, check
two version numbers in that file and adjust them:
- `python3.12` → your Python minor version (`python3 --version`)
- `1.23.2` → your onnxruntime version (`pip show onnxruntime`)

---

## Python pipeline

Run these scripts in order from the repo root.

### 1. Download data

```bash
python python/data/download.py
```

Downloads freMTPL2freq from OpenML and saves it to `data/freMTPL2freq.csv`.

> **Note:** `dataset.get_data()` must specify `target="ClaimNb"` explicitly.
> Without it, OpenML returns `y=None` and writes a blank `ClaimNb` column to the CSV,
> causing LightGBM to fail with `[poisson]: sum of labels is zero`.

### 2. Generate synthetic claim history

```bash
python python/generate_history.py
```

Uses the v1 ONNX model to generate a synthetic 3-year claim history for each policy by
drawing `Poisson(λ)` three times per policy (one per historical year). This bootstraps the
`PriorClaims3Y` feature needed to train the v2 model.

Saves `data/freMTPL2freq_with_history.csv` with three individual history columns
(`claims_hist_1`, `claims_hist_2`, `claims_hist_3`) so the Rust simulation can maintain
the rolling window per policy.

### 3. Train the frequency models

```bash
python python/train.py
```

Trains two LightGBM Poisson models back-to-back using a shared `ModelSpec` dataclass:

| | v1 | v2 |
|---|---|---|
| Features | VehPower, VehAge, DrivAge, **BonusMalus**, Density, Area, VehBrand, VehGas, Region | VehPower, VehAge, DrivAge, Density, Area, VehBrand, VehGas, Region, **PriorClaims3Y** |
| BonusMalus | included | dropped — hard to project forward without a BM transition model |
| PriorClaims3Y | — | sum of claims in the last 3 years (updated each simulation year) |

Both use the Poisson objective with `log(Exposure)` as offset (annual frequency model)
and early stopping on Poisson deviance.

Saves `frequency_model.lgb` + `feature_metadata.json` (v1) and
`frequency_model_v2.lgb` + `feature_metadata_v2.json` (v2).

### 4. Export to ONNX

```bash
python python/export_onnx.py
```

Converts both LightGBM models to ONNX using `onnxmltools`. For each model:
- **Input:** float32 tensor `[N, 9]` — all features in order (categoricals as label-encoded integers cast to float32)
- **Output:** float32 tensor `[N, 1]` — annual claim frequency λ per policy (already in original scale, not log scale)

Expected claims: `μ = λ × exposure`. Runs a validation check comparing LightGBM vs ONNX predictions.

### 5. Export portfolios for Rust

```bash
python python/export_portfolio.py
```

Exports two CSVs used by the Rust engine:

- **v1** `data/portfolio.csv` — full 678 K-policy portfolio, flat numeric CSV with `bonus_malus`.
- **v2** `data/portfolio_v2.csv` — 10 K sampled policies (tractable for the multi-year loop),
  with `claims_hist_1/2/3` as the rolling-window seed instead of `bonus_malus`.

### 6. Validate *(optional)*

```bash
python python/validate.py
```

End-to-end sanity check (v1 only):
- Compares LightGBM vs ONNX Runtime predictions (max diff, correlation)
- Reports predicted vs actual portfolio frequency
- Saves `data/eda/lgb_vs_onnx.png` scatter plot

### 7. EDA *(optional)*

```bash
python python/eda.py
```

Produces plots in `data/eda/`: claim distribution, empirical frequency per feature, and
correlation matrix of numeric features.

---

## Rust simulation engine

### How it works

#### v1 — Single-year simulation

1. Load `data/portfolio.csv` (678 K policies).
2. Run ONNX inference **once** — all 10,000 simulations share the same λ values.
3. Simulate in parallel (Rayon): each simulation draws `Poisson(λ × exposure)` per policy.
4. Report mean, std, and percentiles (P50–P99.5) of the claim frequency distribution.

#### v2 — Multi-year simulation (5 years)

1. Load `data/portfolio_v2.csv` (10 K policies with claim history seed).
2. For each of 10,000 simulations (in parallel, one ONNX session per Rayon worker thread):
   - **Year t=0..4:** build feature matrix with current `VehAge`, `DrivAge`, `PriorClaims3Y`
     → ONNX inference → draw `Poisson(λ)` per policy → shift rolling 3-year claim window
     → increment `VehAge` and `DrivAge` by 1.
3. Report per-year mean claims, mean frequency, P50, P95, P99.

The key architectural point: from year 1 onward each simulation has a distinct
`PriorClaims3Y` (because the Poisson draws differ), so ONNX must be called
once per year per simulation. Each Rayon worker thread owns one ONNX session to avoid
lock contention — zero serialisation overhead across cores.

### Run

```bash
cd rust
cargo run --release
```

`--release` enables compiler optimisations — always use it for benchmarking.
Without it the binary is 10–30× slower.

### Unit tests

```bash
cd rust
cargo test
```

---

## Benchmark: Python vs Rust

```bash
# Python baseline
python python/benchmark.py

# Rust (all cores)
cd rust && cargo run --release
```

The Python baseline uses `multiprocessing` to be a fair comparison. The Rust engine
parallelises with Rayon. Both run the same workload.

| Simulation | Engine | Parallelism | Workload |
|---|---|---|---|
| v1 single-year | Python | multiprocessing | ONNX once + 10K × 678K Poisson draws |
| v1 single-year | Rust | Rayon | ONNX once + 10K × 678K Poisson draws |
| v2 multi-year | Python | multiprocessing | 10K × 5 ONNX calls + 10K × 5 × 10K Poisson draws |
| v2 multi-year | Rust | Rayon (per-thread sessions) | 10K × 5 ONNX calls + 10K × 5 × 10K Poisson draws |

The v1 speedup comes from two sources:
1. **Parallelism** — Rayon distributes simulations across all cores with a one-line change.
2. **Compiled code** — the Poisson sampling loop compiles to native machine code with no
   interpreter overhead.

The v2 speedup additionally demonstrates that even when ONNX must be called repeatedly
(once per year per simulation), Rust's per-thread session design eliminates contention
and keeps all cores busy.
