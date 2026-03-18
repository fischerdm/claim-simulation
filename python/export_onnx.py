"""
export_onnx.py
--------------
Converts trained LightGBM frequency models to ONNX for inference in Rust.

Exports both models in one run:
  v1 — frequency_model.onnx    (9 features incl. BonusMalus)
  v2 — frequency_model_v2.onnx (9 features incl. PriorClaims3Y)

Important notes on the ONNX graph:
- Input:  float32 tensor of shape [N, n_features]
- Output: float32 tensor of shape [N, 1] — annual claim frequency λ (already
          exponentiated). onnxmltools preserves LightGBM's exp() transform.
          Rust multiplies by exposure to get expected claim count: μ = λ × exposure.
- Categoricals are label-encoded integers passed as float32.

Usage:
    python python/train.py       # must run first
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

# (lgb model path, onnx output path, metadata path)
EXPORT_TARGETS = [
    (
        MODELS_DIR / "frequency_model.lgb",
        MODELS_DIR / "frequency_model.onnx",
        MODELS_DIR / "feature_metadata.json",
    ),
    (
        MODELS_DIR / "frequency_model_v2.lgb",
        MODELS_DIR / "frequency_model_v2.onnx",
        MODELS_DIR / "feature_metadata_v2.json",
    ),
]


def export_model(lgb_path: Path, onnx_path: Path, meta_path: Path) -> None:
    """Load, convert to ONNX, validate, and save one model."""
    if not lgb_path.exists():
        raise FileNotFoundError(
            f"Model not found at {lgb_path}. Run python/train.py first."
        )

    booster = lgb.Booster(model_file=str(lgb_path))
    with open(meta_path) as f:
        metadata = json.load(f)
    n_features = len(metadata["feature_names"])

    logger.info("── Exporting %s (%d features) ──", lgb_path.name, n_features)
    logger.info("  Features: %s", metadata["feature_names"])

    # Convert
    initial_types = [("float_input", FloatTensorType([None, n_features]))]
    onnx_model = convert_lightgbm(booster, initial_types=initial_types, target_opset=15)
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    logger.info("  Saved to %s (%.1f KB)", onnx_path, onnx_path.stat().st_size / 1024)

    # Validate: compare LightGBM Python vs ONNX Runtime on random inputs
    rng       = np.random.default_rng(42)
    X         = rng.standard_normal((500, n_features)).astype(np.float32)
    lgb_preds = booster.predict(X)

    sess        = rt.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name  = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    onnx_preds  = sess.run([output_name], {input_name: X})[0].flatten()

    max_diff = np.max(np.abs(lgb_preds - onnx_preds))
    logger.info(
        "  Max abs diff LGB vs ONNX: %.2e — %s",
        max_diff,
        "OK" if max_diff < 1e-4 else "WARN: exceeds 1e-4, inspect carefully",
    )
    logger.info("  Sample LGB λ:  %s", np.round(lgb_preds[:5], 4))
    logger.info("  Sample ONNX λ: %s", np.round(onnx_preds[:5], 4))
    logger.info("  Done.\n")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Export LightGBM model to ONNX.")
    parser.add_argument("version", choices=["v1", "v2"], help="Model version to export.")
    args = parser.parse_args()

    idx = 0 if args.version == "v1" else 1
    export_model(*EXPORT_TARGETS[idx])


if __name__ == "__main__":
    main()
