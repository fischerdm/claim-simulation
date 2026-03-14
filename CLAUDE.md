# Claim Simulation — Project Guide for Claude

## Project Overview
Non-life insurance pricing side project: LightGBM Poisson frequency model (Python) →
exported to ONNX → Monte Carlo claim simulation (Rust, using the `ort` crate).

Dataset: **freMTPL2freq**, OpenML ID 41214, 678,013 rows, mean claim frequency ≈ 0.10.
Features: VehPower, VehAge, DrivAge, BonusMalus, Density, Area, VehBrand, VehGas, Region.
Categorical encodings stored in `models/feature_metadata.json`.

---

## Python Pipeline (run in order)

1. `python/generate_history.py`  → `data/freMTPL2freq_with_history.csv`
2. `python/train.py`             → v1 + v2 models + metadata (both versions)
3. `python/export_onnx.py`       → v1 + v2 ONNX models
4. `python/export_portfolio.py`  → `data/portfolio.csv` (v1) + `data/portfolio_v2.csv` (v2, 10K sampled rows)

> Note: `train_v2.py`, `export_onnx_v2.py`, `export_portfolio_v2.py` were deleted — their
> logic was merged into the shared scripts above using `ModelSpec` / `export_model()` patterns.

---

## Critical: LightGBM Poisson / ONNX Output

**`booster.predict()` returns λ (annual frequency) directly, NOT log(λ).**
`onnxmltools` preserves the `exp()` transform, so the ONNX model also outputs λ directly.

Correct formula for expected claims per policy:
```
μ = onnx_output × exposure    # λ × fraction-of-year
```
NOT:
```
μ = exp(onnx_output + log(exposure))   # WRONG — double-exponentiation
```
Verified: `mu_correct.sum() / exposure.sum()` = 0.1005 vs actual frequency 0.1007. ✓

---

## v2 Feature Order (9 features, same count as v1)

Matches `train.py` `ALL_FEATURES` = numeric_features + CATEGORICAL_FEATURES:

```
[VehPower, VehAge, DrivAge, Density, PriorClaims3Y, Area, VehBrand, VehGas, Region]
```
`PriorClaims3Y` is at **position 4** (before all categoricals).
Both Rust `to_feature_row` and Python `_simulate_v2_chunk` must use this order.
(Bug previously had `PriorClaims3Y` at position 8 — fixed.)

---

## Rust Build Setup

- Crate: `ort = "=2.0.0-rc.11"` (exact pin — only rc versions on crates.io)
- Feature flags: `load-dynamic` (not `download-binaries`; Intel Mac has no prebuilt binary)
- `rand = { version = "0.8", features = ["small_rng"] }` (`SmallRng` needs the feature)
- Build/run requires env var:
  ```sh
  ORT_DYLIB_PATH=.venv/lib/python3.12/site-packages/onnxruntime/capi/libonnxruntime.1.23.2.dylib cargo run
  ```

### ort 2.0.0-rc.11 API notes
- `ort::session::Session` (NOT `ort::Session`)
- `session.outputs()` is a method, NOT a field
- `outlet.name()` is a method, NOT a field
- `session.run(&mut self)` takes `&mut self` → cannot run from parallel Rayon threads
- Tensor creation: `Tensor::<f32>::from_array(([n, 9_usize], flat_vec))`
- Output extraction: `output.try_extract_tensor::<f32>()` → `(&Shape, &[f32])`
- Get owned output: `outputs.remove("name")` → `Option<DynValue>`
- ONNX input name: `"float_input"`, output name: `"variable"`, output shape: `[N, 1]`

---

## Architecture

### v1 — Single Year
- `model.rs`: `compute_lambdas` → 1 ONNX call; `run_inference(flat, n, n_feat)` → generic batch ONNX
- `simulator.rs`: `run_parallel(lambdas, exposure, n_sims)` → Rayon parallel Poisson draws
- Each simulation gets its own `SmallRng::seed_from_u64(i as u64)` → reproducible + independent

### v2 — Multi-Year (5 years, N_SIMS_V2 = 10,000)
- `portfolio.rs`: `PolicyMultiYear` — no BonusMalus; `claims_hist: [u32; 3]` rolling window
- `simulator_multiyear.rs`: ONNX sessions in `thread_local! { THREAD_MODEL }`, loaded lazily
  - Each simulation runs its own 5-year loop; ONNX called once per year per simulation
  - Rolling window shift: `[w[1], w[2], new_claims]` after each year
  - VehAge/DrivAge += 1.0 each year; exposure uses portfolio value at t=0, 1.0 for t=1..4
- Python v2 benchmark: `multiprocessing.Pool`, one ONNX session per worker (mirrors Rust design)
- Python benchmark auto-passes `ORT_DYLIB_PATH` to Rust subprocess via `_ort_dylib_path()`

#### ONNX Session Init — Why Lazy
Pre-allocating sessions upfront blocks the main thread for N_threads × T_session before any
sim starts. Lazy init spreads cost across the first Rayon task wave. However, ONNX Runtime
holds a **global lock during init**, so sessions still load sequentially:
> T_startup ≈ N_threads × T_session (not just T_session)

#### Calibration (macOS Intel, 2026-03-07)
10K policies, 500 sims, 5 years, 8 cores:
- T_total ≈ 502 s
- T_startup ≈ 195 s (T_session ≈ 24 s × 8 threads)
- T_compute ≈ 307 s
- T_inference(10K policies) ≈ 1 s/call (100 µs/policy — ~300× slower than Linux w/ AVX-512)

---

## macOS Intel Setup Notes
- `brew install libomp` required for LightGBM
- Rust ≥ 1.93.1 required (1.66.0 is too old for `ort` dependencies)

---

## Known Bugs Fixed
- `download.py`: must pass `get_data(target="ClaimNb")` explicitly
- `export_onnx.py`: use `target_opset=15` (onnxmltools max; 17 fails)
- `train.py` / `validate.py`: frequency formula was `exp(pred + log(exp))` → fixed to `pred * exp`
- v2 feature order: `PriorClaims3Y` was at position 8 (after categoricals) → moved to position 4
