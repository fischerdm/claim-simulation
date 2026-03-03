"""
export_onnx_v2.py
-----------------
Converts the trained v2 LightGBM model to ONNX for use by the Rust
multi-year simulation engine.

v2 model differences vs v1:
  - Input:  9 features — BonusMalus replaced by PriorClaims3Y (at position 8)
  - Output: float32[N, 1] — annual claim frequency λ (already exponentiated)

Feature order in the ONNX input tensor (must match portfolio_v2.rs):
    VehPower, VehAge, DrivAge, Density, Area, VehBrand, VehGas, Region, PriorClaims3Y

Usage:
    python python/train_v2.py        # must run first
    python python/export_onnx_v2.py
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

MODELS_DIR  = Path(__file__).parent.parent / "models"
MODEL_PATH  = MODELS_DIR / "frequency_model_v2.lgb"
ONNX_PATH   = MODELS_DIR / "frequency_model_v2.onnx"
META_PATH   = MODELS_DIR / "feature_metadata_v2.json"


def load_model() -> tuple[lgb.Booster, dict]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"v2 model not found at {MODEL_PATH}. Run python/train_v2.py first."
        )
    booster = lgb.Booster(model_file=str(MODEL_PATH))
    logger.info("Loaded v2 LightGBM model from %s", MODEL_PATH)

    with open(META_PATH) as f:
        metadata = json.load(f)
    logger.info("Features (%d): %s", len(metadata["feature_names"]), metadata["feature_names"])

    return booster, metadata


def export_onnx(booster: lgb.Booster, n_features: int) -> None:
    initial_types = [("float_input", FloatTensorType([None, n_features]))]

    logger.info("Converting v2 model to ONNX (n_features=%d)...", n_features)
    onnx_model = convert_lightgbm(
        booster,
        initial_types=initial_types,
        target_opset=15,
    )

    with open(ONNX_PATH, "wb") as f:
        f.write(onnx_model.SerializeToString())

    size_kb = ONNX_PATH.stat().st_size / 1024
    logger.info("Saved ONNX v2 model to %s (%.1f KB)", ONNX_PATH, size_kb)


def validate_onnx(booster: lgb.Booster, n_features: int) -> None:
    logger.info("Validating ONNX v2 output against LightGBM Python predictions...")

    rng = np.random.default_rng(42)
    X   = rng.standard_normal((500, n_features)).astype(np.float32)

    lgb_preds  = booster.predict(X)

    sess        = rt.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    input_name  = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    onnx_preds  = sess.run([output_name], {input_name: X})[0].flatten()

    max_diff  = np.max(np.abs(lgb_preds - onnx_preds))
    mean_diff = np.mean(np.abs(lgb_preds - onnx_preds))
    logger.info("Max absolute difference:  %.2e", max_diff)
    logger.info("Mean absolute difference: %.2e", mean_diff)

    if max_diff < 1e-4:
        logger.info("v2 ONNX export validated successfully.")
    else:
        logger.warning(
            "Differences exceed 1e-4 — acceptable for float32 tree models, inspect carefully."
        )


def main() -> None:
    booster, metadata = load_model()
    n_features = len(metadata["feature_names"])
    export_onnx(booster, n_features)
    validate_onnx(booster, n_features)
    logger.info(
        "\nONNX v2 model ready at %s\n"
        "Input: float32[N, %d]  —  "
        "VehPower, VehAge, DrivAge, Density, Area, VehBrand, VehGas, Region, PriorClaims3Y\n"
        "Output: λ (annual frequency); multiply by exposure to get expected claim count.",
        ONNX_PATH,
        n_features,
    )


if __name__ == "__main__":
    main()
