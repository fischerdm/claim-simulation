"""
validate.py
-----------
End-to-end validation of the full pipeline on real data:
1. Loads the raw dataset
2. Applies the same preprocessing as train.py
3. Runs predictions with both LightGBM Python and the ONNX model
4. Compares outputs and reports key metrics

Use this after running train.py and export_onnx.py.
Also produces a scatter plot of LightGBM vs ONNX predictions.

Usage:
    python python/validate.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as rt
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DATA_PATH = Path(__file__).parent.parent / "data" / "freMTPL2freq.csv"
MODELS_DIR = Path(__file__).parent.parent / "models"
ONNX_PATH = MODELS_DIR / "frequency_model.onnx"
LGB_PATH = MODELS_DIR / "frequency_model.lgb"
META_PATH = MODELS_DIR / "feature_metadata.json"
PLOT_PATH = Path(__file__).parent.parent / "data" / "eda" / "lgb_vs_onnx.png"


def load_artifacts() -> tuple[lgb.Booster, rt.InferenceSession, dict]:
    booster = lgb.Booster(model_file=str(LGB_PATH))
    sess = rt.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    with open(META_PATH) as f:
        metadata = json.load(f)
    return booster, sess, metadata


def preprocess(df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    """Re-apply the same preprocessing as train.py using saved category orderings."""
    df = df.copy()
    df["Exposure"] = df["Exposure"].clip(lower=1e-6, upper=1.0)
    df["BonusMalus"] = df["BonusMalus"].clip(upper=200)
    df["VehPower"] = df["VehPower"].clip(upper=15)

    for col in metadata["categorical_features"]:
        classes = metadata["categorical_encodings"][col]
        encoding = {label: i for i, label in enumerate(classes)}
        df[col] = df[col].astype(str).map(encoding).fillna(0).astype(int)

    return df


def run_validation(
    booster: lgb.Booster,
    sess: rt.InferenceSession,
    df: pd.DataFrame,
    metadata: dict,
    sample_n: int = 10_000,
) -> None:
    features = metadata["feature_names"]

    # Use a fixed sample for speed; set sample_n=None to use full dataset
    if sample_n and len(df) > sample_n:
        df_sample = df.sample(n=sample_n, random_state=42)
        logger.info("Validating on %d sampled rows (of %d total)", sample_n, len(df))
    else:
        df_sample = df

    X = df_sample[features].values.astype(np.float32)
    exposure = df_sample["Exposure"].values
    y = df_sample["ClaimNb"].values

    # LightGBM predictions.
    # booster.predict() for Poisson returns λ (annual frequency) directly.
    # Expected count in the observation period = λ × exposure.
    lgb_lambda = booster.predict(X)          # λ per policy per year
    lgb_mu = lgb_lambda * exposure           # expected claims in period

    # ONNX predictions (onnxmltools preserves the exp() — same scale as Python predict).
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    onnx_lambda = sess.run([output_name], {input_name: X})[0].flatten()  # λ per policy
    onnx_mu = onnx_lambda * exposure         # expected claims in period

    # Agreement between LightGBM and ONNX
    diff = np.abs(lgb_lambda - onnx_lambda)
    logger.info("LGB vs ONNX — max abs diff in λ:  %.6f", diff.max())
    logger.info("LGB vs ONNX — mean abs diff in λ: %.6f", diff.mean())
    logger.info("LGB vs ONNX — correlation:         %.8f", np.corrcoef(lgb_lambda, onnx_lambda)[0, 1])

    # Predictive quality: sum(μ_i) / sum(exposure_i) = exposure-weighted mean frequency
    actual_freq = y.sum() / exposure.sum()
    pred_freq_lgb = lgb_mu.sum() / exposure.sum()
    pred_freq_onnx = onnx_mu.sum() / exposure.sum()
    logger.info("Actual frequency:      %.4f", actual_freq)
    logger.info("LGB predicted freq:    %.4f", pred_freq_lgb)
    logger.info("ONNX predicted freq:   %.4f", pred_freq_onnx)

    # Scatter plot
    PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(lgb_lambda, onnx_lambda, alpha=0.15, s=4, color="steelblue")
    lims = [min(lgb_lambda.min(), onnx_lambda.min()), max(lgb_lambda.max(), onnx_lambda.max())]
    ax.plot(lims, lims, "r--", linewidth=1, label="y = x")
    ax.set_xlabel("LightGBM λ (Python)")
    ax.set_ylabel("ONNX λ (ort)")
    ax.set_title("LightGBM vs ONNX Predictions (λ per policy)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
    plt.close()
    logger.info("Saved scatter plot to %s", PLOT_PATH)

    # Inference speed benchmark
    import time

    X_bench = df[features].values.astype(np.float32)

    t0 = time.perf_counter()
    booster.predict(X_bench)
    t_lgb = time.perf_counter() - t0

    t0 = time.perf_counter()
    sess.run([output_name], {input_name: X_bench})
    t_onnx = time.perf_counter() - t0

    n = len(X_bench)
    logger.info("Inference speed on %d policies:", n)
    logger.info("  LightGBM Python: %.1f ms (%.0f µs/policy)", t_lgb * 1000, t_lgb / n * 1e6)
    logger.info("  ONNX Runtime:    %.1f ms (%.0f µs/policy)", t_onnx * 1000, t_onnx / n * 1e6)


def main() -> None:
    booster, sess, metadata = load_artifacts()
    df = pd.read_csv(DATA_PATH)
    df = preprocess(df, metadata)
    run_validation(booster, sess, df, metadata)


if __name__ == "__main__":
    main()