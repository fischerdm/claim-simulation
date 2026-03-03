"""
benchmark.py
------------
Python Monte Carlo claim simulation — used to benchmark against the Rust engine.

Runs N_SIMS simulations using ONNX inference (same model as Rust) and NumPy
vectorised Poisson draws. Each simulation is a sequential Python loop iteration;
there is no Rayon-style automatic parallelism (adding multiprocessing would require
significant boilerplate and pickle overhead, which itself is an argument for Rust).

Output format mirrors the Rust engine so you can compare results side-by-side.

Usage:
    python python/benchmark.py

Then compare with the Rust engine:
    cd rust && cargo run --release
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import onnxruntime as rt
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

PORTFOLIO_PATH = Path(__file__).parent.parent / "data" / "portfolio.csv"
ONNX_PATH = Path(__file__).parent.parent / "models" / "frequency_model.onnx"

FEATURE_COLS = [
    "veh_power", "veh_age", "driv_age", "bonus_malus", "density",
    "area", "veh_brand", "veh_gas", "region",
]

N_SIMS = 10_000


def load_portfolio() -> tuple[np.ndarray, np.ndarray]:
    """Returns (features float32 [N, 9], exposure float64 [N])."""
    df = pd.read_csv(PORTFOLIO_PATH)
    features = df[FEATURE_COLS].values.astype(np.float32)
    exposure = df["exposure"].values.astype(np.float64)
    return features, exposure


def compute_lambdas(features: np.ndarray, exposure: np.ndarray) -> np.ndarray:
    """Run ONNX inference, return μ per policy (λ × exposure)."""
    sess = rt.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    lambdas = sess.run([output_name], {input_name: features})[0].flatten().astype(np.float64)
    return lambdas * exposure  # μ = λ × exposure


def simulate(mus: np.ndarray, total_exposure: float, n_sims: int, seed: int = 42) -> np.ndarray:
    """
    Run n_sims independent Poisson simulations.

    Each iteration draws Poisson(μ_i) for all N policies at once (vectorised),
    then sums across policies for the portfolio-level claim count. The loop
    over simulations is sequential — the Python GIL prevents true thread-level
    parallelism (in contrast to Rust's Rayon which parallelises across all cores
    with zero overhead).

    With 678 K policies, each numpy Poisson draw takes ~10 ms, so
    10 000 simulations ≈ 100 s single-threaded Python.
    """
    rng = np.random.default_rng(seed)
    claims = np.empty(n_sims, dtype=np.float64)
    for i in range(n_sims):
        claims[i] = rng.poisson(mus).sum()
    return claims / total_exposure  # → frequencies


def print_stats(frequencies: np.ndarray, total_exposure: float, n_sims: int) -> None:
    mean_claims = frequencies.mean() * total_exposure
    print()
    print(f"=== Claim Simulation Results ({n_sims} simulations) ===")
    print(f"Portfolio exposure: {total_exposure:.2f} policy-years")
    print()
    print(f"Expected claims per simulation:  {mean_claims:.2f}")
    print()
    print("Claim frequency (claims / policy-year):")
    print(f"  Mean:   {frequencies.mean():.5f}")
    print(f"  Std:    {frequencies.std():.5f}")
    print(f"  P50:    {np.percentile(frequencies, 50):.5f}")
    print(f"  P75:    {np.percentile(frequencies, 75):.5f}")
    print(f"  P95:    {np.percentile(frequencies, 95):.5f}")
    print(f"  P99:    {np.percentile(frequencies, 99):.5f}")
    print(f"  P99.5:  {np.percentile(frequencies, 99.5):.5f}")
    print()


def main() -> None:
    if not PORTFOLIO_PATH.exists():
        raise FileNotFoundError(
            f"Portfolio CSV not found at {PORTFOLIO_PATH}. "
            "Run python/export_portfolio.py first."
        )

    logger.info("Loading portfolio from %s ...", PORTFOLIO_PATH)
    features, exposure = load_portfolio()
    n_policies = len(features)
    total_exposure = float(exposure.sum())
    logger.info("Portfolio: %d policies, %.2f total policy-years", n_policies, total_exposure)

    logger.info("Running ONNX inference ...")
    mus = compute_lambdas(features, exposure)
    logger.info("Mean μ per policy: %.5f", mus.mean())

    logger.info("Running %d simulations (single-threaded Python) ...", N_SIMS)
    t0 = time.perf_counter()
    frequencies = simulate(mus, total_exposure, N_SIMS)
    elapsed = time.perf_counter() - t0

    logger.info(
        "Done in %.2fs  (%.1f µs/simulation)",
        elapsed,
        elapsed / N_SIMS * 1e6,
    )

    print_stats(frequencies, total_exposure, N_SIMS)

    print("To compare: cd rust && cargo run --release")
    print("(--release enables compiler optimisations; always use it for benchmarking)")


if __name__ == "__main__":
    main()
