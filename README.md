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
    └── src/                    # simulation engine (in development)
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

### 4. Export to ONNX

```bash
python python/export_onnx.py
```

Converts the LightGBM model to ONNX format using `onnxmltools`. The ONNX model:
- **Input:** float32 tensor `[N, 9]` — all features in order (categoricals as label-encoded integers cast to float32)
- **Output:** float32 tensor `[N]` — raw `log(λ)` **without** the exposure offset

The Rust simulation engine must therefore compute `λ = exp(log_lambda + log(exposure))` before
drawing from the Poisson distribution.

Saves `models/frequency_model.onnx` and runs a quick sanity check against the Python predictions.

> **Note on opset version:** `target_opset=15` is used. The `onnxmltools` converter supports up
> to opset 15; requesting a higher version raises a `RuntimeError`. Opset 15 is sufficient for
> gradient boosted tree models — no relevant operators were added in later versions.

### 5. Validate

```bash
python python/validate.py
```

End-to-end validation on real data:
- Compares LightGBM Python predictions vs ONNX Runtime predictions (max diff, correlation)
- Reports predicted vs actual portfolio frequency
- Benchmarks inference speed of LightGBM vs ONNX Runtime on the full 678K-row dataset
- Saves `data/eda/lgb_vs_onnx.png` scatter plot

## Rust simulation engine

*(in development)* — loads `frequency_model.onnx` via the `ort` crate, runs Monte Carlo
claim simulations over a policy portfolio in parallel using Rayon.
