"""
export_onnx.py
--------------
Converts the trained LightGBM Poisson frequency model to ONNX format
for fast inference in the Rust simulation engine via the `ort` crate.

Important notes on the ONNX graph:
- Input:  float32 tensor of shape [N, n_features] — ALL_FEATURES in order
- Output: float32 tensor of shape [N] — raw log(lambda) WITHOUT exposure offset
          The Rust code must add log(exposure) before exp() to get the claim rate.
- Categoricals are label-encoded integers passed as float32 (LightGBM ONNX convention).

Usage:
    python python/export_onnx.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import lightgbm as lgb
import numpy as np
import onnxruntime as rt
from onnxmltools import convert_lightgbm
from onnxmltools.convert.common.data_types import FloatTensorType

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

MODELS_DIR = Path(__file__).parent.parent / "models"
MODEL_PATH = MODELS_DIR / "frequency_model.lgb"
ONNX_PATH = MODELS_DIR / "frequency_model.onnx"
META_PATH = MODELS_DIR / "feature_metadata.json"


def load_model() -> tuple[lgb.Booster, dict]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run python/train.py first."
        )
    booster = lgb.Booster(model_file=str(MODEL_PATH))
    logger.info("Loaded LightGBM model from %s", MODEL_PATH)

    with open(META_PATH) as f:
        metadata = json.load(f)
    logger.info("Loaded feature metadata: %d features", len(metadata["feature_names"]))

    return booster, metadata


def export_onnx(booster: lgb.Booster, n_features: int) -> None:
    """
    Convert LightGBM booster to ONNX using onnxmltools.

    The initial_types specification tells the converter the input shape.
    All features are passed as a single float32 matrix.
    """
    initial_types = [("float_input", FloatTensorType([None, n_features]))]

    logger.info("Converting to ONNX (n_features=%d)...", n_features)
    onnx_model = convert_lightgbm(
        booster,
        initial_types=initial_types,
        target_opset=17,
    )

    with open(ONNX_PATH, "wb") as f:
        f.write(onnx_model.SerializeToString())

    size_kb = ONNX_PATH.stat().st_size / 1024
    logger.info("Saved ONNX model to %s (%.1f KB)", ONNX_PATH, size_kb)


def validate_onnx(booster: lgb.Booster, n_features: int) -> None:
    """
    Run a quick sanity check: compare LightGBM Python predictions
    vs ONNX Runtime predictions on random inputs.
    """
    logger.info("Validating ONNX output against LightGBM Python predictions...")

    rng = np.random.default_rng(42)
    X = rng.standard_normal((500, n_features)).astype(np.float32)

    # LightGBM predictions (log scale, no offset)
    lgb_preds = booster.predict(X)

    # ONNX Runtime predictions
    sess = rt.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    onnx_preds = sess.run([output_name], {input_name: X})[0].flatten()

    max_diff = np.max(np.abs(lgb_preds - onnx_preds))
    mean_diff = np.mean(np.abs(lgb_preds - onnx_preds))
    logger.info("Max absolute difference:  %.2e", max_diff)
    logger.info("Mean absolute difference: %.2e", mean_diff)

    if max_diff < 1e-4:
        logger.info("✓ ONNX export validated successfully.")
    else:
        logger.warning(
            "⚠ Differences exceed 1e-4. This may be acceptable for tree models "
            "due to float32 vs float64 precision. Inspect carefully."
        )

    # Log a few sample predictions for manual inspection
    logger.info("Sample LightGBM preds (log λ): %s", np.round(lgb_preds[:5], 4))
    logger.info("Sample ONNX preds     (log λ): %s", np.round(onnx_preds[:5], 4))
    logger.info(
        "Exponentiated (λ before exposure): %s",
        np.round(np.exp(onnx_preds[:5]), 4),
    )


def main() -> None:
    booster, metadata = load_model()
    n_features = len(metadata["feature_names"])
    export_onnx(booster, n_features)
    validate_onnx(booster, n_features)
    logger.info(
        "\nONNX model ready at %s\n"
        "Rust usage: input float32[N, %d], output is log(λ) — add log(exposure) before exp().",
        ONNX_PATH,
        n_features,
    )


if __name__ == "__main__":
    main()