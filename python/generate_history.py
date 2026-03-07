"""
generate_history.py
-------------------
Uses the trained v1 ONNX model to generate a synthetic 3-year claim history
for each policy in freMTPL2freq, bootstrapping the PriorClaims3Y feature
needed to train model v2.

For each policy:
  - λ is predicted by the v1 ONNX model (annual claim frequency)
  - Three independent Poisson(λ) draws simulate claims in years t=-3, t=-2, t=-1
  - Exposure = 1.0 for each historical year (full year assumed)

Three individual columns are saved (not just the sum) so that the Rust
simulation can maintain a rolling window — at year t+1 the oldest year is
dropped and the newly simulated year is appended.

Outputs data/freMTPL2freq_with_history.csv, which train_v2.py reads.

Usage:
    python python/generate_history.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import onnxruntime as rt
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DATA_PATH  = Path(__file__).parent.parent / "data" / "freMTPL2freq.csv"
META_PATH  = Path(__file__).parent.parent / "models" / "feature_metadata.json"
ONNX_PATH  = Path(__file__).parent.parent / "models" / "frequency_model.onnx"
OUT_PATH   = Path(__file__).parent.parent / "data" / "freMTPL2freq_with_history.csv"

# Feature names expected by the v1 ONNX model (must match train.py order)
NUMERIC_FEATURES     = ["VehPower", "VehAge", "DrivAge", "BonusMalus", "Density"]
CATEGORICAL_FEATURES = ["Area", "VehBrand", "VehGas", "Region"]
ALL_FEATURES_V1      = NUMERIC_FEATURES + CATEGORICAL_FEATURES

RANDOM_STATE = 42


def preprocess(df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    """Apply the same clipping and label encoding as train.py."""
    df = df.copy()
    df["Exposure"]   = df["Exposure"].clip(lower=1e-6, upper=1.0)
    df["BonusMalus"] = df["BonusMalus"].clip(upper=200)
    df["VehPower"]   = df["VehPower"].clip(upper=15)
    df["ClaimNb"]    = df["ClaimNb"].clip(upper=10)

    for col in CATEGORICAL_FEATURES:
        classes  = metadata["categorical_encodings"][col]
        encoding = {label: i for i, label in enumerate(classes)}
        df[col]  = df[col].astype(str).map(encoding).fillna(0).astype(int)

    return df


def main() -> None:
    for p in (DATA_PATH, META_PATH, ONNX_PATH):
        if not p.exists():
            raise FileNotFoundError(f"Required file not found: {p}")

    with open(META_PATH) as f:
        metadata = json.load(f)

    df = pd.read_csv(DATA_PATH)
    logger.info("Loaded %d rows from %s", len(df), DATA_PATH)
    df = preprocess(df, metadata)

    # Run v1 ONNX model to get the annual claim rate λ per policy.
    # We use λ directly (not λ × exposure) because historical years are
    # assumed to be full years (exposure = 1.0).
    sess = rt.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    input_name  = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    X = df[ALL_FEATURES_V1].values.astype(np.float32)
    lambdas = sess.run([output_name], {input_name: X})[0].flatten()
    logger.info(
        "v1 λ: min=%.4f  max=%.4f  mean=%.4f",
        lambdas.min(), lambdas.max(), lambdas.mean(),
    )

    # Draw 3 independent Poisson(λ) samples per policy — one per historical year.
    # Simplification: we use the current policy features for all three years
    # (ignoring that VehAge/DrivAge were different then). Acceptable for a toy example.
    rng = np.random.default_rng(RANDOM_STATE)
    df["claims_hist_1"] = rng.poisson(lambdas).astype(np.int32)  # t = -3  (oldest)
    df["claims_hist_2"] = rng.poisson(lambdas).astype(np.int32)  # t = -2
    df["claims_hist_3"] = rng.poisson(lambdas).astype(np.int32)  # t = -1  (most recent)
    df["PriorClaims3Y"] = (
        df["claims_hist_1"] + df["claims_hist_2"] + df["claims_hist_3"]
    )

    logger.info(
        "PriorClaims3Y: mean=%.3f  max=%d  zeros=%.1f%%",
        df["PriorClaims3Y"].mean(),
        df["PriorClaims3Y"].max(),
        (df["PriorClaims3Y"] == 0).mean() * 100,
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    logger.info("Saved %d rows to %s", len(df), OUT_PATH)


if __name__ == "__main__":
    main()
