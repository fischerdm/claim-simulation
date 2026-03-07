"""
benchmark.py
------------
Runs the single-year (v1) and multi-year (v2) Monte Carlo claim simulations in
Python and Rust and prints a side-by-side timing comparison.

Python simulation: ONNX inference + multiprocessing Poisson loop.
Rust simulation:   same ONNX inference + parallel Rayon Poisson loop.

Using multiprocessing on the Python side is the fair comparison: both engines
use all available CPU cores. The remaining difference is purely the cost of
the Poisson sampling loop itself (Python/NumPy vs compiled Rust).

v1 — Single-year:
  Architecture: `mus` (678 K floats) is sent once to each worker process via
  the Pool initializer — not once per simulation — so pickling overhead is
  negligible. Each worker then runs its assigned chunk of simulations
  sequentially, drawing Poisson(mu) for every policy and summing.

v2 — Multi-year (5 years):
  Each worker process owns one ONNX session (created in the initializer).
  For each simulation the worker runs a 5-year loop: rebuild the feature
  matrix with updated VehAge, DrivAge, PriorClaims3Y → ONNX inference →
  Poisson draw → shift the rolling 3-year claim window → age policies.
  This mirrors the Rust per-thread ONNX session design.

Usage:
    python python/benchmark.py
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import onnxruntime as rt
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

PORTFOLIO_PATH    = Path(__file__).parent.parent / "data" / "portfolio.csv"
PORTFOLIO_PATH_V2 = Path(__file__).parent.parent / "data" / "portfolio_v2.csv"
ONNX_PATH         = Path(__file__).parent.parent / "models" / "frequency_model.onnx"
ONNX_PATH_V2      = Path(__file__).parent.parent / "models" / "frequency_model_v2.onnx"
RUST_DIR          = Path(__file__).parent.parent / "rust"

FEATURE_COLS = [
    "veh_power", "veh_age", "driv_age", "bonus_malus", "density",
    "area", "veh_brand", "veh_gas", "region",
]

# v2 static features (veh_age / driv_age change each year and are tracked separately)
FEATURE_COLS_V2_STATIC = ["veh_power", "density", "area", "veh_brand", "veh_gas", "region"]

N_SIMS    = 10_000
N_SIMS_V2 = 10_000
N_YEARS   = 5
N_WORKERS = os.cpu_count() or 4


# ---------------------------------------------------------------------------
# Portfolio + ONNX loading
# ---------------------------------------------------------------------------

def load_portfolio() -> tuple[np.ndarray, np.ndarray]:
    """Returns (features float32 [N, 9], exposure float64 [N])."""
    df       = pd.read_csv(PORTFOLIO_PATH)
    features = df[FEATURE_COLS].values.astype(np.float32)
    exposure = df["exposure"].values.astype(np.float64)
    return features, exposure


def compute_mus(features: np.ndarray, exposure: np.ndarray) -> np.ndarray:
    """Run ONNX inference; return μ = λ × exposure per policy."""
    sess        = rt.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    input_name  = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    lambdas     = sess.run([output_name], {input_name: features})[0].flatten().astype(np.float64)
    return lambdas * exposure


def load_portfolio_v2() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
        static_features: float32 [N, 6] — veh_power, density, area, veh_brand, veh_gas, region
        veh_age_init:    float32 [N]
        driv_age_init:   float32 [N]
        exposure:        float64 [N]
        claims_hist:     int32   [N, 3]
    """
    df              = pd.read_csv(PORTFOLIO_PATH_V2)
    static_features = df[FEATURE_COLS_V2_STATIC].values.astype(np.float32)
    veh_age_init    = df["veh_age"].values.astype(np.float32)
    driv_age_init   = df["driv_age"].values.astype(np.float32)
    exposure        = df["exposure"].values.astype(np.float64)
    claims_hist     = df[["claims_hist_1", "claims_hist_2", "claims_hist_3"]].values.astype(np.int32)
    return static_features, veh_age_init, driv_age_init, exposure, claims_hist


# ---------------------------------------------------------------------------
# Python v1 simulation — multiprocessing
# ---------------------------------------------------------------------------

# Module-level state shared across worker processes (set via initializer).
_worker_mus:            np.ndarray | None = None
_worker_total_exposure: float | None      = None


def _worker_init(mus: np.ndarray, total_exposure: float) -> None:
    """Called once per worker process to install shared read-only state."""
    global _worker_mus, _worker_total_exposure
    _worker_mus            = mus
    _worker_total_exposure = total_exposure


def _simulate_chunk(seeds: list[int]) -> list[float]:
    """
    Run one simulation per seed in this chunk.
    Each simulation draws Poisson(mu) for every policy (vectorised) and sums.
    Returns claim frequencies (total_claims / total_exposure).
    """
    results = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        results.append(rng.poisson(_worker_mus).sum() / _worker_total_exposure)
    return results


def simulate_parallel(mus: np.ndarray, total_exposure: float, n_sims: int) -> np.ndarray:
    """Distribute n_sims simulations across N_WORKERS processes."""
    seeds      = list(range(n_sims))
    chunk_size = max(1, (n_sims + N_WORKERS - 1) // N_WORKERS)
    chunks     = [seeds[i : i + chunk_size] for i in range(0, n_sims, chunk_size)]

    with Pool(
        processes=N_WORKERS,
        initializer=_worker_init,
        initargs=(mus, total_exposure),
    ) as pool:
        nested = pool.map(_simulate_chunk, chunks)

    return np.array([freq for chunk in nested for freq in chunk])


# ---------------------------------------------------------------------------
# Python v2 simulation — multiprocessing, one ONNX session per worker
# ---------------------------------------------------------------------------

# Worker state for v2 (set via initializer).
_worker_v2_static:      np.ndarray | None          = None  # [N, 6]
_worker_v2_veh_age:     np.ndarray | None          = None  # [N]
_worker_v2_driv_age:    np.ndarray | None          = None  # [N]
_worker_v2_exposure:    np.ndarray | None          = None  # [N]
_worker_v2_claims_hist: np.ndarray | None          = None  # [N, 3]
_worker_v2_sess:        rt.InferenceSession | None = None
_worker_v2_input_name:  str | None                 = None
_worker_v2_output_name: str | None                 = None


def _worker_v2_init(
    static_features: np.ndarray,
    veh_age_init:    np.ndarray,
    driv_age_init:   np.ndarray,
    exposure:        np.ndarray,
    claims_hist:     np.ndarray,
    onnx_path:       Path,
) -> None:
    """Called once per worker: installs shared data and creates a local ONNX session."""
    global _worker_v2_static, _worker_v2_veh_age, _worker_v2_driv_age
    global _worker_v2_exposure, _worker_v2_claims_hist
    global _worker_v2_sess, _worker_v2_input_name, _worker_v2_output_name
    _worker_v2_static      = static_features
    _worker_v2_veh_age     = veh_age_init
    _worker_v2_driv_age    = driv_age_init
    _worker_v2_exposure    = exposure
    _worker_v2_claims_hist = claims_hist
    _worker_v2_sess        = rt.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    _worker_v2_input_name  = _worker_v2_sess.get_inputs()[0].name
    _worker_v2_output_name = _worker_v2_sess.get_outputs()[0].name


def _simulate_v2_chunk(seeds: list[int]) -> list[list[float]]:
    """
    Run one 5-year simulation per seed.

    Feature order sent to ONNX (matches train.py ALL_FEATURES for V2):
        [VehPower, VehAge, DrivAge, Density, PriorClaims3Y, Area, VehBrand, VehGas, Region]

    static_features columns:
        0: veh_power  1: density  2: area  3: veh_brand  4: veh_gas  5: region

    Returns a list of [year_0_freq, ..., year_4_freq] per simulation.
    """
    results = []
    n       = len(_worker_v2_static)

    for seed in seeds:
        rng = np.random.default_rng(seed)

        # Clone mutable simulation state.
        veh_age  = _worker_v2_veh_age.copy()
        driv_age = _worker_v2_driv_age.copy()
        hist     = _worker_v2_claims_hist.copy()  # [N, 3]: [oldest t-3, t-2, newest t-1]

        sim_freqs: list[float] = []
        for year in range(N_YEARS):
            prior_3y = hist.sum(axis=1).astype(np.float32)  # [N]

            # Build [N, 9] feature matrix.
            # Column order must match ALL_FEATURES in train.py (V2):
            #   [VehPower, VehAge, DrivAge, Density, PriorClaims3Y, Area, VehBrand, VehGas, Region]
            features = np.column_stack([
                _worker_v2_static[:, 0],  # veh_power
                veh_age,
                driv_age,
                _worker_v2_static[:, 1],  # density
                prior_3y,                 # prior_claims_3y
                _worker_v2_static[:, 2],  # area
                _worker_v2_static[:, 3],  # veh_brand
                _worker_v2_static[:, 4],  # veh_gas
                _worker_v2_static[:, 5],  # region
            ]).astype(np.float32)

            lambdas = _worker_v2_sess.run(
                [_worker_v2_output_name], {_worker_v2_input_name: features}
            )[0].flatten().astype(np.float64)

            # Exposure: portfolio value at t=0, full year (1.0) thereafter.
            exposure = _worker_v2_exposure if year == 0 else np.ones(n, dtype=np.float64)
            claims   = rng.poisson(lambdas * exposure).astype(np.int32)

            # Shift rolling window: drop oldest, append this year's draw.
            hist = np.column_stack([hist[:, 1], hist[:, 2], claims])

            sim_freqs.append(float(claims.sum()) / n)

            veh_age  += 1.0
            driv_age += 1.0

        results.append(sim_freqs)

    return results


def simulate_v2_parallel(
    static_features: np.ndarray,
    veh_age_init:    np.ndarray,
    driv_age_init:   np.ndarray,
    exposure:        np.ndarray,
    claims_hist:     np.ndarray,
    n_sims:          int,
) -> np.ndarray:
    """
    Distribute n_sims v2 simulations across N_WORKERS processes.
    Returns an [n_sims, N_YEARS] array of per-year claim frequencies.
    """
    seeds      = list(range(n_sims))
    chunk_size = max(1, (n_sims + N_WORKERS - 1) // N_WORKERS)
    chunks     = [seeds[i : i + chunk_size] for i in range(0, n_sims, chunk_size)]

    with Pool(
        processes=N_WORKERS,
        initializer=_worker_v2_init,
        initargs=(static_features, veh_age_init, driv_age_init, exposure, claims_hist, ONNX_PATH_V2),
    ) as pool:
        nested = pool.map(_simulate_v2_chunk, chunks)

    all_sims = [sim for chunk in nested for sim in chunk]
    return np.array(all_sims)  # [n_sims, N_YEARS]


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def print_stats(label: str, frequencies: np.ndarray, total_exposure: float, n_sims: int) -> None:
    mean_claims = frequencies.mean() * total_exposure
    print(f"=== {label} — Simulation Results ({n_sims:,} simulations) ===")
    print(f"Portfolio exposure: {total_exposure:.2f} policy-years")
    print(f"Expected claims per simulation:  {mean_claims:.2f}")
    print("Claim frequency (claims / policy-year):")
    print(f"  Mean:   {frequencies.mean():.5f}")
    print(f"  Std:    {frequencies.std():.5f}")
    print(f"  P50:    {np.percentile(frequencies, 50):.5f}")
    print(f"  P75:    {np.percentile(frequencies, 75):.5f}")
    print(f"  P95:    {np.percentile(frequencies, 95):.5f}")
    print(f"  P99:    {np.percentile(frequencies, 99):.5f}")
    print(f"  P99.5:  {np.percentile(frequencies, 99.5):.5f}")
    print()


def print_stats_v2(
    label:           str,
    year_frequencies: np.ndarray,
    n_policies:      int,
    n_sims:          int,
) -> None:
    """year_frequencies: [n_sims, N_YEARS] array of per-year claim frequencies."""
    print(
        f"=== {label} — Multi-Year Simulation Results "
        f"({n_sims:,} sims × {N_YEARS} years × {n_policies:,} policies) ==="
    )
    print(
        f"{'Year':<6}  {'Mean claims':>12}  {'Mean freq':>10}  "
        f"{'P50':>10}  {'P95':>10}  {'P99':>10}"
    )
    print("-" * 64)
    for year in range(N_YEARS):
        freqs       = year_frequencies[:, year]
        mean_claims = freqs.mean() * n_policies
        print(
            f"t={year:<4}  {mean_claims:>12.1f}  {freqs.mean():>10.5f}  "
            f"{np.percentile(freqs, 50):>10.5f}  {np.percentile(freqs, 95):>10.5f}  "
            f"{np.percentile(freqs, 99):>10.5f}"
        )
    print()


# ---------------------------------------------------------------------------
# Rust simulation (via pre-built binary)
# ---------------------------------------------------------------------------

RUST_BINARY = RUST_DIR / "target" / "release" / "claim-simulation"


def build_rust() -> bool:
    """
    Compile the Rust engine with `cargo build --release`.
    Build output streams to the terminal so the user can see progress.
    Returns True on success.
    """
    logger.info("Building Rust engine (cargo build --release) ...")
    result = subprocess.run(
        ["cargo", "build", "--release"],
        cwd=RUST_DIR,
    )
    if result.returncode != 0:
        logger.warning("Rust build failed (exit %d).", result.returncode)
        return False
    logger.info("Build complete.")
    return True


def run_rust() -> tuple[float | None, float | None]:
    """
    Run the pre-built Rust binary and return (v1_elapsed_s, v2_elapsed_s).

    Separating build from run is essential for a fair benchmark: compilation
    time is irrelevant to simulation throughput and can take 30-60 s after
    source changes, completely swamping the actual simulation time.
    """
    if not RUST_BINARY.exists():
        logger.info("Binary not found — building first.")
        if not build_rust():
            return None, None

    logger.info("Running Rust binary (%s) ...", RUST_BINARY)
    result = subprocess.run(
        [str(RUST_BINARY)],
        cwd=RUST_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        logger.warning("Rust binary failed (exit %d):\n%s", result.returncode, result.stderr)
        return None, None

    print(result.stdout)

    # Parse v1: "Done in X.XXs  (Y.Y µs/simulation)"
    m1 = re.search(r"\(([\d.]+) µs/simulation\)", result.stdout)
    # Parse v2: "Done in X.XXXs  (Y.Y ms/simulation)"
    m2 = re.search(r"\(([\d.]+) ms/simulation\)", result.stdout)

    rust_v1 = float(m1.group(1)) * N_SIMS    / 1e6 if m1 else None
    rust_v2 = float(m2.group(1)) * N_SIMS_V2 / 1e3 if m2 else None

    return rust_v1, rust_v2


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    if not PORTFOLIO_PATH.exists():
        raise FileNotFoundError(
            f"Portfolio CSV not found at {PORTFOLIO_PATH}. "
            "Run python/export_portfolio.py first."
        )

    # ── v1: single-year ──────────────────────────────────────────────────────
    logger.info("Loading v1 portfolio (%s) ...", PORTFOLIO_PATH)
    features, exposure = load_portfolio()
    total_exposure     = float(exposure.sum())
    logger.info(
        "Portfolio: %d policies, %.2f total policy-years",
        len(features), total_exposure,
    )

    logger.info("Running ONNX inference (v1) ...")
    mus = compute_mus(features, exposure)

    logger.info("Running %d simulations (Python v1, %d workers) ...", N_SIMS, N_WORKERS)
    t0           = time.perf_counter()
    frequencies  = simulate_parallel(mus, total_exposure, N_SIMS)
    py_v1_elapsed = time.perf_counter() - t0
    logger.info(
        "Python v1 done in %.2fs  (%.1f µs/simulation)",
        py_v1_elapsed, py_v1_elapsed / N_SIMS * 1e6,
    )

    print()
    print_stats("Python v1", frequencies, total_exposure, N_SIMS)

    # ── v2: multi-year ───────────────────────────────────────────────────────
    py_v2_elapsed: float | None = None
    n_policies_v2               = 0

    if PORTFOLIO_PATH_V2.exists() and ONNX_PATH_V2.exists():
        logger.info("Loading v2 portfolio (%s) ...", PORTFOLIO_PATH_V2)
        static_features, veh_age_init, driv_age_init, exposure_v2, claims_hist = load_portfolio_v2()
        n_policies_v2 = len(static_features)
        logger.info("v2 portfolio: %d policies", n_policies_v2)

        logger.info(
            "Running %d simulations (Python v2, %d workers, %d years) ...",
            N_SIMS_V2, N_WORKERS, N_YEARS,
        )
        t0            = time.perf_counter()
        year_freqs    = simulate_v2_parallel(
            static_features, veh_age_init, driv_age_init, exposure_v2, claims_hist, N_SIMS_V2,
        )
        py_v2_elapsed = time.perf_counter() - t0
        logger.info(
            "Python v2 done in %.2fs  (%.1f ms/simulation)",
            py_v2_elapsed, py_v2_elapsed / N_SIMS_V2 * 1e3,
        )

        print()
        print_stats_v2("Python v2", year_freqs, n_policies_v2, N_SIMS_V2)
    else:
        logger.warning(
            "v2 portfolio or ONNX model not found — skipping Python v2 benchmark. "
            "Run python/export_portfolio.py first."
        )

    rust_v1_elapsed, rust_v2_elapsed = run_rust()

    # ── Summary table ─────────────────────────────────────────────────────────
    n_cores = os.cpu_count() or "?"
    W = 58
    print("=" * W)
    print("  Benchmark summary")
    print("=" * W)

    # v1
    print(f"  v1 — Single-year  ({N_SIMS:,} sims, {len(features):,} policies)")
    print(f"  {'Engine':<16}  {'Workers':>7}  {'Time':>8}  {'µs/sim':>8}")
    print(f"  {'-'*16}  {'-'*7}  {'-'*8}  {'-'*8}")
    print(
        f"  {'Python':<16}  {N_WORKERS:>7}  "
        f"{py_v1_elapsed:>7.2f}s  {py_v1_elapsed / N_SIMS * 1e6:>7.1f}"
    )
    if rust_v1_elapsed is not None:
        print(
            f"  {'Rust (Rayon)':<16}  {n_cores:>7}  "
            f"{rust_v1_elapsed:>7.2f}s  {rust_v1_elapsed / N_SIMS * 1e6:>7.1f}"
        )
        print(f"  Speedup v1: {py_v1_elapsed / rust_v1_elapsed:.1f}×")

    # v2
    if py_v2_elapsed is not None or rust_v2_elapsed is not None:
        print()
        print(f"  v2 — Multi-year  ({N_SIMS_V2:,} sims × {N_YEARS} years, {n_policies_v2:,} policies)")
        print(f"  {'Engine':<16}  {'Workers':>7}  {'Time':>8}  {'ms/sim':>8}")
        print(f"  {'-'*16}  {'-'*7}  {'-'*8}  {'-'*8}")
        if py_v2_elapsed is not None:
            print(
                f"  {'Python':<16}  {N_WORKERS:>7}  "
                f"{py_v2_elapsed:>7.2f}s  {py_v2_elapsed / N_SIMS_V2 * 1e3:>7.1f}"
            )
        if rust_v2_elapsed is not None:
            print(
                f"  {'Rust (Rayon)':<16}  {n_cores:>7}  "
                f"{rust_v2_elapsed:>7.2f}s  {rust_v2_elapsed / N_SIMS_V2 * 1e3:>7.1f}"
            )
        if py_v2_elapsed is not None and rust_v2_elapsed is not None:
            print(f"  Speedup v2: {py_v2_elapsed / rust_v2_elapsed:.1f}×")

    print("=" * W)


if __name__ == "__main__":
    main()
