"""
benchmark.py
------------
Runs the single-year Monte Carlo claim simulation in Python and Rust and
prints a side-by-side timing comparison.

Python simulation: ONNX inference + multiprocessing Poisson loop.
Rust simulation:   same ONNX inference + parallel Rayon Poisson loop.

Using multiprocessing on the Python side is the fair comparison: both engines
use all available CPU cores. The remaining difference is purely the cost of
the Poisson sampling loop itself (Python/NumPy vs compiled Rust).

Architecture: `mus` (678 K floats) is sent once to each worker process via the
Pool initializer — not once per simulation — so pickling overhead is negligible.
Each worker then runs its assigned chunk of simulations sequentially, drawing
Poisson(mu) for every policy and summing.

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

PORTFOLIO_PATH = Path(__file__).parent.parent / "data" / "portfolio.csv"
ONNX_PATH      = Path(__file__).parent.parent / "models" / "frequency_model.onnx"
RUST_DIR       = Path(__file__).parent.parent / "rust"

FEATURE_COLS = [
    "veh_power", "veh_age", "driv_age", "bonus_malus", "density",
    "area", "veh_brand", "veh_gas", "region",
]

N_SIMS    = 10_000
N_WORKERS = os.cpu_count() or 4


# ---------------------------------------------------------------------------
# Portfolio + ONNX
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


# ---------------------------------------------------------------------------
# Python simulation — multiprocessing
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
# Statistics
# ---------------------------------------------------------------------------

def print_stats(label: str, frequencies: np.ndarray, total_exposure: float, n_sims: int) -> None:
    mean_claims = frequencies.mean() * total_exposure
    print(f"=== {label} — Simulation Results ({n_sims} simulations) ===")
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


# ---------------------------------------------------------------------------
# Rust simulation (via subprocess)
# ---------------------------------------------------------------------------

def run_rust() -> float | None:
    """
    Run the Rust engine via `cargo run --release`.
    Returns the v1 simulation time in seconds, or None if the run fails.
    """
    logger.info("Running Rust engine (cargo run --release) ...")
    result = subprocess.run(
        ["cargo", "run", "--release"],
        cwd=RUST_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        logger.warning("Rust engine failed (exit %d):\n%s", result.returncode, result.stderr)
        return None

    print(result.stdout)

    # Parse "Done in X.XXs  (Y.Y µs/simulation)" from the v1 section
    match = re.search(r"v1.*?Done in.*?\(([\d.]+) µs/simulation\)", result.stdout, re.DOTALL)
    if not match:
        # Fall back to first match
        match = re.search(r"\(([\d.]+) µs/simulation\)", result.stdout)
    if not match:
        logger.warning("Could not parse Rust v1 timing from output.")
        return None

    us_per_sim = float(match.group(1))
    return us_per_sim * N_SIMS / 1e6


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    if not PORTFOLIO_PATH.exists():
        raise FileNotFoundError(
            f"Portfolio CSV not found at {PORTFOLIO_PATH}. "
            "Run python/export_portfolio.py first."
        )

    logger.info("Loading portfolio (%s) ...", PORTFOLIO_PATH)
    features, exposure = load_portfolio()
    total_exposure     = float(exposure.sum())
    logger.info(
        "Portfolio: %d policies, %.2f total policy-years",
        len(features), total_exposure,
    )

    logger.info("Running ONNX inference ...")
    mus = compute_mus(features, exposure)

    logger.info(
        "Running %d simulations (Python, %d workers) ...", N_SIMS, N_WORKERS,
    )
    t0               = time.perf_counter()
    frequencies      = simulate_parallel(mus, total_exposure, N_SIMS)
    python_elapsed   = time.perf_counter() - t0
    logger.info(
        "Python done in %.2fs  (%.1f µs/simulation)",
        python_elapsed, python_elapsed / N_SIMS * 1e6,
    )

    print()
    print_stats("Python", frequencies, total_exposure, N_SIMS)

    rust_elapsed = run_rust()

    print("=" * 56)
    print("  Benchmark summary (v1 single-year, 10 K sims, 678 K policies)")
    print("=" * 56)
    print(f"  {'Engine':<16}  {'Workers':>7}  {'Time':>8}  {'µs/sim':>8}")
    print(f"  {'-'*16}  {'-'*7}  {'-'*8}  {'-'*8}")
    print(
        f"  {'Python':<16}  {N_WORKERS:>7}  "
        f"{python_elapsed:>7.2f}s  {python_elapsed / N_SIMS * 1e6:>7.1f}"
    )
    if rust_elapsed is not None:
        n_rust_threads = os.cpu_count() or "?"
        speedup = python_elapsed / rust_elapsed
        print(
            f"  {'Rust (Rayon)':<16}  {n_rust_threads:>7}  "
            f"{rust_elapsed:>7.2f}s  {rust_elapsed / N_SIMS * 1e6:>7.1f}"
        )
        print(f"  {'-'*16}  {'-'*7}  {'-'*8}  {'-'*8}")
        print(f"  Speedup: {speedup:.1f}×")
    print("=" * 56)


if __name__ == "__main__":
    main()
