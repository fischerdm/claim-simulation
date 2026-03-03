"""
benchmark.py
------------
Runs the Monte Carlo claim simulation in both Python and Rust, then prints a
side-by-side timing comparison.

Python simulation: ONNX inference + sequential NumPy Poisson loop (no parallelism).
Rust simulation:   same ONNX inference + parallel Rayon Poisson loop (all CPU cores).

Usage:
    python python/benchmark.py
"""

from __future__ import annotations

import logging
import re
import subprocess
import time
from pathlib import Path

import numpy as np
import onnxruntime as rt
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

PORTFOLIO_PATH = Path(__file__).parent.parent / "data" / "portfolio.csv"
ONNX_PATH = Path(__file__).parent.parent / "models" / "frequency_model.onnx"
RUST_DIR = Path(__file__).parent.parent / "rust"

FEATURE_COLS = [
    "veh_power", "veh_age", "driv_age", "bonus_malus", "density",
    "area", "veh_brand", "veh_gas", "region",
]

N_SIMS = 10_000


# ---------------------------------------------------------------------------
# Python simulation
# ---------------------------------------------------------------------------

def load_portfolio() -> tuple[np.ndarray, np.ndarray]:
    """Returns (features float32 [N, 9], exposure float64 [N])."""
    df = pd.read_csv(PORTFOLIO_PATH)
    features = df[FEATURE_COLS].values.astype(np.float32)
    exposure = df["exposure"].values.astype(np.float64)
    return features, exposure


def compute_lambdas(features: np.ndarray, exposure: np.ndarray) -> np.ndarray:
    """Run ONNX inference; return μ per policy (λ × exposure)."""
    sess = rt.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    lambdas = sess.run([output_name], {input_name: features})[0].flatten().astype(np.float64)
    return lambdas * exposure


def simulate(mus: np.ndarray, total_exposure: float, n_sims: int, seed: int = 42) -> np.ndarray:
    """
    Sequential Poisson simulation — one Python loop iteration per simulation.
    Each iteration is vectorised (NumPy draws all N policy counts at once),
    but the GIL prevents running simulations in parallel across threads.
    """
    rng = np.random.default_rng(seed)
    claims = np.empty(n_sims, dtype=np.float64)
    for i in range(n_sims):
        claims[i] = rng.poisson(mus).sum()
    return claims / total_exposure


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

    Cargo's build output goes to stderr (visible in the terminal so the user
    sees progress). The Rust engine's stdout is captured for timing extraction.

    Returns the simulation time in seconds, or None if the run fails.
    """
    logger.info("Running Rust engine (cargo run --release) ...")
    result = subprocess.run(
        ["cargo", "run", "--release"],
        cwd=RUST_DIR,
        stdout=subprocess.PIPE,  # capture for parsing
        text=True,               # stderr flows through → user sees cargo output
    )

    if result.returncode != 0:
        logger.warning("Rust engine exited with code %d", result.returncode)
        return None

    print(result.stdout)

    # Parse "Done in X.XXs  (Y.Y µs/simulation)"
    match = re.search(r"\(([\d.]+) µs/simulation\)", result.stdout)
    if not match:
        logger.warning("Could not parse Rust timing from output.")
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

    # --- Python ---
    logger.info("Loading portfolio (%s) ...", PORTFOLIO_PATH)
    features, exposure = load_portfolio()
    total_exposure = float(exposure.sum())
    logger.info("Portfolio: %d policies, %.2f total policy-years", len(features), total_exposure)

    logger.info("Running ONNX inference ...")
    mus = compute_lambdas(features, exposure)

    logger.info("Running %d simulations (Python, single-threaded) ...", N_SIMS)
    t0 = time.perf_counter()
    frequencies = simulate(mus, total_exposure, N_SIMS)
    python_elapsed = time.perf_counter() - t0
    logger.info("Python done in %.2fs  (%.1f µs/simulation)", python_elapsed, python_elapsed / N_SIMS * 1e6)

    print()
    print_stats("Python", frequencies, total_exposure, N_SIMS)

    # --- Rust ---
    rust_elapsed = run_rust()

    # --- Comparison ---
    print("=" * 52)
    print("  Benchmark summary")
    print("=" * 52)
    print(f"  {'Engine':<12}  {'Time':>10}  {'µs/sim':>10}")
    print(f"  {'-'*12}  {'-'*10}  {'-'*10}")
    print(f"  {'Python':<12}  {python_elapsed:>9.2f}s  {python_elapsed / N_SIMS * 1e6:>9.1f}")
    if rust_elapsed is not None:
        speedup = python_elapsed / rust_elapsed
        print(f"  {'Rust':<12}  {rust_elapsed:>9.2f}s  {rust_elapsed / N_SIMS * 1e6:>9.1f}")
        print(f"  {'-'*12}  {'-'*10}  {'-'*10}")
        print(f"  Speedup: {speedup:.1f}×")
    print("=" * 52)


if __name__ == "__main__":
    main()
