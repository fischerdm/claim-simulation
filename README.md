# Claim Simulation

Non-life actuarial frequency model trained in Python (LightGBM + Poisson) and exported
to ONNX for high-speed recursive claim simulation in Rust.

The project has two goals:

1. **Is ONNX faster than native LightGBM for batch inference?**
   Relevant for the common single-year use case where you want a point estimate of λ per
   policy, not a full simulation.

2. **How fast can a multi-year recursive simulation be?**
   Each year, `PriorClaims3Y` (a rolling 3-year claim count) is updated from the draws of
   the previous year, so λ must be recomputed annually. There is no closed-form alternative.
   The Rust + ONNX engine parallelises across simulations using Rayon, with each worker
   thread owning its own ONNX session.

See [BENCHMARK.md](BENCHMARK.md) for the full study design, scaling grids, and runtime
estimates on different hardware.

## Model

The **v2 frequency model** is used for all studies:

| Feature | Notes |
|---|---|
| VehPower, VehAge, DrivAge | Vehicle and driver characteristics |
| Density | Population density of driver's municipality |
| **PriorClaims3Y** | Rolling 3-year claim count — updated each simulation year |
| Area, VehBrand, VehGas, Region | Categorical features (label-encoded) |

BonusMalus is excluded — it cannot be projected forward without a separate BM transition
model. PriorClaims3Y acts as a lightweight, simulatable experience feature.

## Project structure

```
claim-simulation/
├── data/
│   ├── freMTPL2freq.csv                # raw dataset (generated, not in git)
│   ├── freMTPL2freq_with_history.csv   # augmented with synthetic claim history (generated)
│   ├── portfolio.csv                   # v1 portfolio — kept for reference (generated)
│   └── portfolio_v2.csv                # v2 portfolio — full 678K policies (generated)
├── models/
│   ├── frequency_model_v2.lgb          # v2 LightGBM model (generated)
│   ├── frequency_model_v2.onnx         # v2 ONNX export (generated)
│   └── feature_metadata_v2.json        # feature names and category encodings
├── python/
│   ├── data/
│   │   └── download.py                 # downloads freMTPL2freq from OpenML
│   ├── generate_history.py             # creates synthetic 3-year claim history
│   ├── train.py                        # trains LightGBM models (v1 + v2)
│   ├── export_onnx.py                  # exports models to ONNX
│   ├── export_portfolio.py             # exports portfolio CSVs
│   ├── validate.py                     # validates LightGBM vs ONNX agreement
│   ├── eda.py                          # exploratory data analysis plots
│   └── benchmark.py                    # runs both benchmark studies
├── results/
│   └── benchmark_results.csv           # timing results (generated, not in git)
├── rust/
│   ├── .cargo/
│   │   └── config.toml                 # sets ORT_DYLIB_PATH for cargo run
│   ├── src/
│   │   ├── main.rs                     # CLI entry point (--n-sims, --years, --fraction)
│   │   ├── model.rs                    # ONNX inference wrapper
│   │   ├── portfolio.rs                # Policy struct and CSV loader
│   │   └── simulator_multiyear.rs      # parallel multi-year simulation (Rayon)
│   └── Cargo.toml
└── BENCHMARK.md                        # study design and runtime estimates
```

## Dataset

[freMTPL2freq](https://www.openml.org/d/41214) — French Motor Third Party Liability
frequency data. 678,013 policies. Target: `ClaimNb`. Downloaded automatically from OpenML.

---

## Setup

### macOS — OpenMP (required for LightGBM)

```bash
brew install libomp
```

Without this, `import lightgbm` fails with a missing `libomp.dylib` error.

### Python environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Rust

You need Rust installed (`rustup`). The Rust engine links against the ONNX Runtime
library bundled with the Python `onnxruntime` package — set up the venv first.

`rust/.cargo/config.toml` sets `ORT_DYLIB_PATH` automatically. **If you clone on a
different machine**, update two version strings in that file:
- `python3.12` → your Python minor version (`python3 --version`)
- `1.23.2` → your onnxruntime version (`pip show onnxruntime`)

---

## Python pipeline

Run these scripts in order from the repo root.

### 1. Download data

```bash
python python/data/download.py
```

Downloads freMTPL2freq from OpenML → `data/freMTPL2freq.csv`.

### 2. Generate synthetic claim history

```bash
python python/generate_history.py
```

Uses the v1 ONNX model to simulate a 3-year claim history for each policy by drawing
`Poisson(λ)` three times. This bootstraps `PriorClaims3Y` needed to train the v2 model.

Saves `data/freMTPL2freq_with_history.csv` with columns `claims_hist_1/2/3`.

### 3. Train the models

```bash
python python/train.py
```

Trains two LightGBM Poisson models. The **v2 model** is used for all benchmarks:
it replaces `BonusMalus` with `PriorClaims3Y` so claim history can be updated
each simulation year. Both models use `log(Exposure)` as offset.

### 4. Export to ONNX

```bash
python python/export_onnx.py
```

Converts both models to ONNX using `onnxmltools` (opset 15). Output is λ (annual
frequency) in original scale — `onnxmltools` preserves LightGBM's internal `exp()`.

Expected claims: `μ = λ × exposure` (not `exp(log_λ + log_exposure)`).

### 5. Export portfolio

```bash
python python/export_portfolio.py
```

Exports `data/portfolio_v2.csv` — the full 678K-policy portfolio with `claims_hist_1/2/3`
as the rolling-window seed. The Rust engine's `--fraction` arg then subsets this at
runtime (e.g. `--fraction 0.25` → ~170K policies).

### 6. Validate *(optional)*

```bash
python python/validate.py
```

Compares LightGBM vs ONNX predictions for the v1 model; reports max diff and
portfolio frequency. Saves a scatter plot to `data/eda/`.

---

## Running the Rust engine directly

```bash
cd rust
cargo run --release -- --n-sims 10000 --years 5 --fraction 1.0
```

**Always use `--release`** — the debug build is 10–30× slower.

CLI options:

| Flag | Default | Description |
|---|---|---|
| `--n-sims N` | 10000 | Number of Monte Carlo simulations |
| `--years Y` | 5 | Projection horizon (1 = single-year, 5 = multi-year) |
| `--fraction F` | 1.0 | Share of portfolio to use (0.0–1.0) |

---

## Running the benchmark

### Quick test (local, < 1 minute)

```bash
QUICK_TEST=1 python python/benchmark.py
```

Uses a tiny grid to validate the full pipeline before committing to a long run.

### Full benchmark

```bash
# Build Rust engine first (one-time, ~30–60 s after source changes)
cd rust && cargo build --release && cd ..

python python/benchmark.py
```

Results are saved to `results/benchmark_results.csv`. The `n_cores` column lets you
stack results from multiple machines for cross-instance comparison.

See [BENCHMARK.md](BENCHMARK.md) for the full study design, observed runtimes, and capacity planning guidance.
