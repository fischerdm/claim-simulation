# Claim Simulation

Non-life actuarial frequency model trained in Python (LightGBM + Poisson) and exported to ONNX
for high-speed claim simulation in Rust. The goal is to run millions of Monte Carlo simulations
over a policy portfolio efficiently, with parallelisation via Rayon.

## Project structure

```
claim-simulation/
├── data/
│   ├── freMTPL2freq.csv        # downloaded dataset (generated, not in git)
│   └── eda/                    # EDA and validation plots (generated)
├── models/
│   ├── frequency_model.lgb     # trained LightGBM model (generated)
│   ├── frequency_model.onnx    # ONNX export for Rust inference (generated)
│   └── feature_metadata.json  # feature names and category encodings
├── python/
│   ├── data/
│   │   └── download.py         # downloads freMTPL2freq from OpenML
│   ├── eda.py                  # exploratory data analysis, saves plots
│   ├── train.py                # trains the LightGBM frequency model
│   ├── export_onnx.py          # converts the model to ONNX format
│   └── validate.py             # validates LightGBM vs ONNX agreement
└── rust/
    ├── .cargo/
    │   └── config.toml         # sets ORT_DYLIB_PATH so cargo run works without extra setup
    ├── src/
    │   ├── main.rs             # entry point
    │   ├── model.rs            # loads and runs the ONNX model
    │   ├── portfolio.rs        # Policy struct and test portfolio
    │   └── simulator.rs        # Poisson draws and statistics
    └── Cargo.toml
```

## Dataset

[freMTPL2freq](https://www.openml.org/d/41214) — French Motor Third Party Liability frequency data.
678,013 policies with features: vehicle power, age, driver age, bonus-malus, region, etc.
Target: `ClaimNb` (number of claims per policy). Downloaded automatically from OpenML.

## Setup

### macOS prerequisite — OpenMP

LightGBM on macOS requires the OpenMP runtime, which is not installed by default. Install it with Homebrew:

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

## Python pipeline

Run the scripts in order:

### 1. Download data

```bash
python python/data/download.py
```

Downloads freMTPL2freq from OpenML and saves it to `data/freMTPL2freq.csv`.

> **Note:** The call to `dataset.get_data()` must specify `target="ClaimNb"` explicitly.
> Without it, OpenML returns `y=None` (no default target is set for this dataset),
> which writes a blank `ClaimNb` column to the CSV. LightGBM then fails during training
> with `[poisson]: sum of labels is zero`.

### 2. Exploratory data analysis

```bash
python python/eda.py
```

Produces three plots in `data/eda/`:
- `claim_distribution.png` — claim count distribution and frequency by exposure bucket
- `feature_frequency.png` — empirical claim frequency for each feature
- `correlation.png` — correlation matrix of numeric features

### 3. Train the frequency model

```bash
python python/train.py
```

Trains a LightGBM Poisson regression model. Key design decisions:
- **Poisson objective** — the natural choice for claim counts
- **log(Exposure) as offset** — the model learns annual claim frequency, not raw counts
- **Label-encoded categoricals** — required for LightGBM's native categorical support
- **Early stopping** on Poisson deviance (validation set, 20% holdout)

Saves `models/frequency_model.lgb` and `models/feature_metadata.json`.

### 4. Export to ONNX (and export portfolio)

```bash
python python/export_onnx.py
```

Converts the LightGBM model to ONNX format using `onnxmltools`. The ONNX model:
- **Input:** float32 tensor `[N, 9]` — all features in order (categoricals as label-encoded integers cast to float32)
- **Output:** float32 tensor `[N, 1]` — annual claim frequency λ per policy (already in original scale, not log scale)

To get the expected number of claims for a policy in the simulation period:
`μ = λ × exposure` (e.g., a policy with λ = 0.10 and exposure = 0.75 years expects 0.075 claims).
The Rust code then draws from Poisson(μ) for each policy.

Saves `models/frequency_model.onnx` and runs a quick sanity check against the Python predictions.

> **Note on opset version:** `target_opset=15` is used. The `onnxmltools` converter supports up
> to opset 15; requesting a higher version raises a `RuntimeError`. Opset 15 is sufficient for
> gradient boosted tree models — no relevant operators were added in later versions.

### 5. Export portfolio for Rust

```bash
python python/export_portfolio.py
```

Preprocesses `freMTPL2freq.csv` (same clipping and label-encoding as `train.py`) and saves
`data/portfolio.csv` — a flat numeric CSV with the 9 model features plus exposure.
This is the file the Rust engine reads at runtime.

Column order: `veh_power, veh_age, driv_age, bonus_malus, density, area, veh_brand, veh_gas, region, exposure`

### 6. Validate

```bash
python python/validate.py
```

End-to-end validation on real data:
- Compares LightGBM Python predictions vs ONNX Runtime predictions (max diff, correlation)
- Reports predicted vs actual portfolio frequency
- Benchmarks inference speed of LightGBM vs ONNX Runtime on the full 678K-row dataset
- Saves `data/eda/lgb_vs_onnx.png` scatter plot

## Rust simulation engine

The Rust engine loads `frequency_model.onnx` and runs 10,000 independent Monte Carlo
simulations over the full 678,013-policy freMTPL2freq portfolio.

### How it works

1. **Load model** — the ONNX model is loaded once from disk.
2. **Compute λ per policy** — the model is run once (it is deterministic: same policy always
   gives the same λ). Each λ is multiplied by the policy's exposure to get μ (expected claims).
3. **Simulate** — 10,000 simulations run in parallel across all CPU cores. Each simulation
   draws a random claim count from Poisson(μ) for every policy and sums them.
4. **Report** — prints mean, standard deviation, and percentiles (P50, P75, P95, P99, P99.5)
   of the simulated claim frequency distribution.

### Setup

You need Rust installed (`rustup`). The engine links against the ONNX Runtime library that
ships with the Python `onnxruntime` package — so you must have the Python venv set up first
(see [Python environment](#python-environment) above).

The file `rust/.cargo/config.toml` tells Cargo where to find the ONNX Runtime library.
It contains a path like:

```
../.venv/lib/python3.12/site-packages/onnxruntime/capi/libonnxruntime.1.23.2.dylib
```

**If you clone this repo on a different machine**, check two version numbers in that file
and adjust them to match your setup:
- `python3.12` → your Python minor version (`python3 --version`)
- `1.23.2` → your onnxruntime version (`pip show onnxruntime`)

### Run

```bash
cd rust
cargo run --release
```

Use `--release` to enable compiler optimisations. Without it the binary is 10–30× slower.
No extra environment variables needed — the config file handles it.

### Unit tests

```bash
cd rust
cargo test
```

The test portfolio (8 handcrafted policies) lives in `#[cfg(test)]` in `portfolio.rs` and
is only compiled when running tests, not in the production binary.

## Benchmark: Python vs Rust

Run the Python simulation first to establish a baseline:

```bash
python python/benchmark.py
```

Then run the Rust engine:

```bash
cd rust && cargo run --release
```

### What the benchmark measures

Both engines run the same workload: ONNX inference on 678 K policies followed by 10,000
independent Monte Carlo simulations, each drawing a Poisson count for every policy.

| Engine | Parallelism | Expected time (10 K sims, 678 K policies) |
|--------|-------------|-------------------------------------------|
| Python | Single-threaded (GIL prevents thread parallelism) | ~100 s |
| Rust   | All CPU cores via Rayon | ~2–10 s |

The speedup comes from two sources:
1. **Parallelism** — Rayon distributes simulations across cores with zero boilerplate
   (`into_par_iter()` is the only change needed). Python would require `multiprocessing`
   with subprocess spawning and pickle overhead.
2. **Compiled code** — Rust's Poisson sampling loop compiles to native SIMD instructions;
   NumPy's inner loop has Python interpreter overhead per simulation.
