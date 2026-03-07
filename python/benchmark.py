"""
benchmark.py
------------
Two benchmark studies using the v2 frequency model (PriorClaims3Y feature).

Study 1 — Inference (no simulation):
  Compare LightGBM native predict() vs ONNX Runtime for batch inference.
  Scaling: portfolio fraction ∈ {25%, 50%, 100%} of portfolio_v2.csv.
  Question: is ONNX Runtime faster than LightGBM for batch inference?

Study 2 — Simulation (Rust + ONNX engine):
  Single-year (n_years=1) and multi-year (n_years=5).
  Scaling grid: portfolio fraction × n_sims (3 × 3 = 9 cells per horizon).
  Question: how does throughput scale with portfolio size and simulation count?

Results are saved to results/benchmark_results.csv.

Usage:
    python python/benchmark.py
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import onnxruntime as rt
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

BASE_DIR       = Path(__file__).parent.parent
PORTFOLIO_PATH = BASE_DIR / "data"   / "portfolio_v2.csv"
ONNX_PATH      = BASE_DIR / "models" / "frequency_model_v2.onnx"
LGB_PATH       = BASE_DIR / "models" / "frequency_model_v2.lgb"
RESULTS_DIR    = BASE_DIR / "results"
RUST_DIR       = BASE_DIR / "rust"
RUST_BINARY    = RUST_DIR / "target" / "release" / "claim-simulation"

# Scaling grids
FRACTIONS   = [0.25, 0.50, 1.0]
N_SIMS_GRID = [2_000, 5_000, 10_000]
N_YEARS_GRID = [1, 5]

# Repetitions for inference timing — take the minimum to suppress OS jitter.
N_REPS = 3


# ---------------------------------------------------------------------------
# Portfolio loading
# ---------------------------------------------------------------------------

def load_portfolio() -> tuple[np.ndarray, np.ndarray]:
    """
    Load portfolio_v2.csv and build the [N, 9] feature matrix in model
    training order (matches train.py V2 ALL_FEATURES):

        [VehPower, VehAge, DrivAge, Density, PriorClaims3Y,
         Area, VehBrand, VehGas, Region]

    PriorClaims3Y is the sum of the three claims-history columns.

    Returns (features float32 [N, 9], exposure float64 [N]).
    """
    df       = pd.read_csv(PORTFOLIO_PATH)
    prior_3y = (
        df["claims_hist_1"] + df["claims_hist_2"] + df["claims_hist_3"]
    ).values.astype(np.float32)

    features = np.column_stack([
        df["veh_power"].values,   # 0: VehPower
        df["veh_age"].values,     # 1: VehAge
        df["driv_age"].values,    # 2: DrivAge
        df["density"].values,     # 3: Density
        prior_3y,                 # 4: PriorClaims3Y
        df["area"].values,        # 5: Area
        df["veh_brand"].values,   # 6: VehBrand
        df["veh_gas"].values,     # 7: VehGas
        df["region"].values,      # 8: Region
    ]).astype(np.float32)

    exposure = df["exposure"].values.astype(np.float64)
    return features, exposure


# ---------------------------------------------------------------------------
# Study 1: Inference benchmark
# ---------------------------------------------------------------------------

def run_inference_benchmark(features: np.ndarray) -> pd.DataFrame:
    """
    Time LightGBM predict() vs ONNX Runtime on portfolio subsets.

    Each measurement is the minimum of N_REPS runs to suppress OS scheduling
    jitter. A warmup call is made before timing to ensure lazy initialisation
    costs (JIT, library loading) are excluded from the measurement.

    Returns a DataFrame with one row per (engine, fraction).
    """
    logger.info("Loading LightGBM model (%s) ...", LGB_PATH)
    booster = lgb.Booster(model_file=str(LGB_PATH))

    logger.info("Loading ONNX session (%s) ...", ONNX_PATH)
    sess        = rt.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    input_name  = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    # Warmup — exclude lazy-init costs from measurements.
    _ = booster.predict(features[:10])
    _ = sess.run([output_name], {input_name: features[:10]})

    n_total = len(features)
    rows: list[dict] = []

    for fraction in FRACTIONS:
        n        = max(1, round(fraction * n_total))
        feat_sub = features[:n]

        lgb_elapsed  = min(
            _time_call(lambda: booster.predict(feat_sub))
            for _ in range(N_REPS)
        )
        onnx_elapsed = min(
            _time_call(lambda: sess.run([output_name], {input_name: feat_sub}))
            for _ in range(N_REPS)
        )
        speedup = lgb_elapsed / onnx_elapsed if onnx_elapsed > 0 else float("nan")

        logger.info(
            "Inference  n=%6d  lgb=%.1f ms  onnx=%.1f ms  speedup=%.2f×",
            n, lgb_elapsed * 1e3, onnx_elapsed * 1e3, speedup,
        )

        for engine, elapsed in [("lgb", lgb_elapsed), ("onnx", onnx_elapsed)]:
            rows.append({
                "engine":       engine,
                "fraction":     fraction,
                "n_policies":   n,
                "elapsed_s":    elapsed,
                "ms_per_call":  elapsed * 1e3,
                "onnx_speedup": speedup if engine == "onnx" else 1.0,
            })

    return pd.DataFrame(rows)


def _time_call(fn) -> float:
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Study 2: Simulation benchmark (Rust + ONNX)
# ---------------------------------------------------------------------------

def build_rust() -> bool:
    logger.info("Building Rust engine (cargo build --release) ...")
    result = subprocess.run(["cargo", "build", "--release"], cwd=RUST_DIR)
    if result.returncode != 0:
        logger.error("Rust build failed.")
        return False
    logger.info("Rust build complete.")
    return True


def run_rust_cell(fraction: float, n_sims: int, n_years: int) -> dict | None:
    """Run the Rust binary for one (fraction, n_sims, n_years) cell."""
    result = subprocess.run(
        [
            str(RUST_BINARY),
            "--fraction", str(fraction),
            "--n-sims",   str(n_sims),
            "--years",    str(n_years),
        ],
        cwd=RUST_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        logger.warning("Rust binary failed:\n%s", result.stderr[:300])
        return None

    # "Portfolio: N/M policies ..."
    m_pol  = re.search(r"Portfolio:\s+(\d+)/\d+", result.stdout)
    # "Done in X.XXXs  (Y.Y ms/simulation)"
    m_time = re.search(r"\(([\d.]+) ms/simulation\)", result.stdout)

    if not m_pol or not m_time:
        logger.warning("Could not parse Rust output:\n%s", result.stdout[:400])
        return None

    ms_per_sim = float(m_time.group(1))
    return {
        "fraction":   fraction,
        "n_policies": int(m_pol.group(1)),
        "n_sims":     n_sims,
        "n_years":    n_years,
        "elapsed_s":  ms_per_sim * n_sims / 1e3,
        "ms_per_sim": ms_per_sim,
        "engine":     "rust_onnx",
    }


def run_simulation_benchmark() -> pd.DataFrame:
    """Drive the Rust engine across all (n_years, n_sims, fraction) combinations."""
    if not RUST_BINARY.exists():
        logger.info("Binary not found — building first.")
        if not build_rust():
            return pd.DataFrame()

    total = len(N_YEARS_GRID) * len(N_SIMS_GRID) * len(FRACTIONS)
    done  = 0
    rows: list[dict] = []

    for n_years in N_YEARS_GRID:
        for n_sims in N_SIMS_GRID:
            for fraction in FRACTIONS:
                done += 1
                logger.info(
                    "[%d/%d] Rust  years=%d  n_sims=%5d  fraction=%.0f%%",
                    done, total, n_years, n_sims, fraction * 100,
                )
                row = run_rust_cell(fraction, n_sims, n_years)
                if row is not None:
                    rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Output tables
# ---------------------------------------------------------------------------

def print_inference_table(df: pd.DataFrame) -> None:
    W = 62
    print()
    print("=" * W)
    print("  Study 1 — Inference: LightGBM vs ONNX Runtime")
    print("=" * W)
    print(f"  {'Policies':>8}  {'Fraction':>8}  {'LightGBM':>10}  {'ONNX RT':>10}  {'Speedup':>8}")
    print(f"  {'-'*8}  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*8}")

    for n_policies in sorted(df["n_policies"].unique()):
        sub      = df[df["n_policies"] == n_policies]
        lgb_row  = sub[sub["engine"] == "lgb"].iloc[0]
        onnx_row = sub[sub["engine"] == "onnx"].iloc[0]
        print(
            f"  {n_policies:>8,}  {lgb_row['fraction']:>7.0%}  "
            f"{lgb_row['ms_per_call']:>9.1f}ms  "
            f"{onnx_row['ms_per_call']:>9.1f}ms  "
            f"{onnx_row['onnx_speedup']:>7.2f}×"
        )

    print("=" * W)
    print()


def print_simulation_table(df: pd.DataFrame) -> None:
    n_sims_vals = sorted(df["n_sims"].unique())
    col_w = 12
    W     = 22 + col_w * len(n_sims_vals)

    print()
    print("=" * W)
    print("  Study 2 — Simulation: Rust + ONNX  (ms / simulation)")
    print("=" * W)

    for n_years in sorted(df["n_years"].unique()):
        label = "Single-year (n_years=1)" if n_years == 1 else f"Multi-year  (n_years={n_years})"
        sub   = df[df["n_years"] == n_years]

        print(f"\n  {label}")
        header = f"  {'Policies':>8}  {'Frac':>4}"
        for ns in n_sims_vals:
            header += f"  {f'{ns:,} sims':>{col_w}}"
        print(header)
        print(f"  {'-'*8}  {'-'*4}" + f"  {'-'*col_w}" * len(n_sims_vals))

        for fraction in sorted(sub["fraction"].unique()):
            frac_rows = sub[sub["fraction"] == fraction]
            n_pol     = int(frac_rows["n_policies"].iloc[0])
            line      = f"  {n_pol:>8,}  {fraction:>3.0%} "
            for ns in n_sims_vals:
                cell = frac_rows[frac_rows["n_sims"] == ns]
                line += f"  {cell['ms_per_sim'].iloc[0]:>{col_w}.1f}" if len(cell) else f"  {'—':>{col_w}}"
            print(line)

    print()
    print("=" * W)
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    if not PORTFOLIO_PATH.exists():
        raise FileNotFoundError(
            f"Portfolio not found at {PORTFOLIO_PATH}. "
            "Run python/export_portfolio.py first."
        )

    n_cores = os.cpu_count() or "?"
    logger.info("Running on %s CPU cores", n_cores)

    logger.info("Loading portfolio from %s ...", PORTFOLIO_PATH)
    features, _exposure = load_portfolio()
    logger.info("Portfolio: %d policies", len(features))

    # Study 1 — Inference benchmark
    logger.info("=== Study 1: Inference benchmark ===")
    inference_df = run_inference_benchmark(features)
    inference_df.insert(0, "study", "inference")

    # Study 2 — Simulation benchmark
    logger.info("=== Study 2: Simulation benchmark ===")
    sim_df = run_simulation_benchmark()
    if not sim_df.empty:
        sim_df.insert(0, "study", "simulation")

    # Save combined results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = RESULTS_DIR / "benchmark_results.csv"
    frames = [inference_df] + ([sim_df] if not sim_df.empty else [])
    all_df = pd.concat(frames, ignore_index=True)
    all_df["n_cores"] = n_cores
    all_df.to_csv(results_path, index=False)
    logger.info("Results saved to %s", results_path)

    # Print summary tables
    print_inference_table(inference_df)
    if not sim_df.empty:
        print_simulation_table(sim_df)


if __name__ == "__main__":
    main()
