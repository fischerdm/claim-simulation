# Claim Simulation

Non-life actuarial frequency model trained in Python (LightGBM + Poisson) and exported to ONNX
for high-speed claim simulation in Rust. The goal is to run thousands of Monte Carlo simulations
over a full policy portfolio efficiently, with parallelisation via Rayon.

## Project structure

```
claim-simulation/
├── data/
│   ├── freMTPL2freq.csv        # raw dataset (generated, not in git)
│   ├── portfolio.csv           # preprocessed portfolio for Rust (generated, not in git)
│   └── eda/                    # EDA and validation plots (generated)
├── models/
│   ├── frequency_model.lgb     # trained LightGBM model (generated)
│   ├── frequency_model.onnx    # ONNX export for Rust inference (generated)
│   └── feature_metadata.json   # feature names and category encodings
├── python/
│   ├── data/
│   │   └── download.py         # downloads freMTPL2freq from OpenML
│   ├── eda.py                  # exploratory data analysis, saves plots
│   ├── train.py                # trains the LightGBM frequency model
│   ├── export_onnx.py          # converts the model to ONNX format
│   ├── export_portfolio.py     # preprocesses the dataset → data/portfolio.csv for Rust
│   ├── validate.py             # validates LightGBM vs ONNX agreement
│   └── benchmark.py            # Python simulation baseline for benchmarking vs Rust
└── rust/
    ├── .cargo/
    │   └── config.toml         # sets ORT_DYLIB_PATH so cargo run works without extra setup
    ├── src/
    │   ├── main.rs             # entry point
    │   ├── model.rs            # ONNX inference
    │   ├── portfolio.rs        # Policy struct and CSV loader
    │   └── simulator.rs        # Poisson draws and statistics
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

The file `rust/.cargo/config.toml` tells Cargo where to find the ONNX Runtime library
(a path inside `.venv/`). **If you clone this repo on a different machine**, check two
version numbers in that file and adjust them:
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

### 2. Train the frequency model

```bash
python python/train.py
```

Trains a LightGBM Poisson regression model. Key design decisions:
- **Poisson objective** — the natural choice for claim counts
- **log(Exposure) as offset** — the model learns annual claim frequency, not raw counts
- **Label-encoded categoricals** — required for LightGBM's native categorical support
- **Early stopping** on Poisson deviance (validation set, 20% holdout)

Saves `models/frequency_model.lgb` and `models/feature_metadata.json`.

### 3. Export to ONNX

```bash
python python/export_onnx.py
```

Converts the LightGBM model to ONNX format using `onnxmltools`. The ONNX model:
- **Input:** float32 tensor `[N, 9]` — all features in order (categoricals as label-encoded integers cast to float32)
- **Output:** float32 tensor `[N, 1]` — annual claim frequency λ per policy (already in original scale, not log scale)

Expected claims for a policy: `μ = λ × exposure` (e.g., λ = 0.10, exposure = 0.75 yr → μ = 0.075).

Saves `models/frequency_model.onnx` and runs a quick sanity check against Python predictions.

> **Note on opset version:** `target_opset=15` is used. The `onnxmltools` converter supports up
> to opset 15; requesting a higher version raises a `RuntimeError`. Opset 15 is sufficient for
> gradient boosted tree models.

### 4. Export portfolio for Rust

```bash
python python/export_portfolio.py
```

Applies the same preprocessing as `train.py` (clipping, label-encoding using the saved
category orderings from `feature_metadata.json`) and saves `data/portfolio.csv` — a flat
numeric CSV that the Rust engine reads directly. Categoricals are already encoded as integers
so Rust only needs to parse floats.

Column order matches the ONNX model input and the `Policy` struct:
`veh_power, veh_age, driv_age, bonus_malus, density, area, veh_brand, veh_gas, region, exposure`

### 5. Validate *(optional)*

```bash
python python/validate.py
```

End-to-end sanity check:
- Compares LightGBM vs ONNX Runtime predictions (max diff, correlation)
- Reports predicted vs actual portfolio frequency
- Saves `data/eda/lgb_vs_onnx.png` scatter plot

### 6. EDA *(optional)*

```bash
python python/eda.py
```

Produces three plots in `data/eda/`:
- `claim_distribution.png` — claim count distribution and frequency by exposure bucket
- `feature_frequency.png` — empirical claim frequency per feature
- `correlation.png` — correlation matrix of numeric features

---

## Rust simulation engine

The Rust engine loads `frequency_model.onnx` and runs 10,000 independent Monte Carlo
simulations over the full 678,013-policy portfolio.

### How it works

1. **Load portfolio** — reads `data/portfolio.csv` (678 K rows, preprocessed by Python).
2. **Load model** — the ONNX model is loaded once from disk.
3. **Compute λ per policy** — a single deterministic ONNX inference pass; each λ is
   multiplied by the policy's exposure to get μ (expected claims in the period).
4. **Simulate** — 10,000 simulations run in parallel across all CPU cores via Rayon.
   Each simulation draws Poisson(μ) for every policy and sums the counts.
5. **Report** — prints mean, std, and percentiles (P50, P75, P95, P99, P99.5) of the
   simulated claim frequency distribution.

### Run

```bash
cd rust
cargo run --release
```

`--release` enables compiler optimisations — always use it for production runs and
benchmarking. Without it the binary is 10–30× slower.

### Unit tests

```bash
cd rust
cargo test
```

A small handcrafted test portfolio (8 policies covering a range of risk profiles) is defined
in `#[cfg(test)]` in `portfolio.rs`. It is only compiled when running tests, not included
in the production binary.

---

## Benchmark: Python vs Rust

Run the Python baseline first, then the Rust engine, and compare the timings.

```bash
# Python (single-threaded)
python python/benchmark.py

# Rust (all cores)
cd rust && cargo run --release
```

Both engines run the same workload: ONNX inference on 678 K policies followed by 10,000
independent Monte Carlo simulations.

| Engine | Parallelism | Expected time (10 K sims, 678 K policies) |
|--------|-------------|-------------------------------------------|
| Python | Single-threaded (GIL prevents thread-level parallelism) | ~100 s |
| Rust   | All CPU cores via Rayon | ~2–10 s |

The speedup comes from two sources:

1. **Parallelism** — Rayon distributes simulations across cores with a one-line change
   (`.into_par_iter()`). Python would require `multiprocessing` with subprocess spawning
   and pickle serialisation overhead.
2. **Compiled code** — the Rust Poisson sampling loop compiles to native machine code;
   NumPy's per-simulation overhead includes Python interpreter calls.
