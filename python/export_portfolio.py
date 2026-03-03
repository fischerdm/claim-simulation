"""
export_portfolio.py
-------------------
Preprocesses freMTPL2freq and exports a numeric CSV for the Rust simulation engine.

The Rust engine reads float32 values directly and does not do label encoding.
This script applies the same preprocessing as train.py (clipping, label encoding
using the saved category orderings from feature_metadata.json) and writes the
9 model features plus exposure to data/portfolio.csv.

Column order matches the ONNX model input and the Rust Policy struct:
    veh_power, veh_age, driv_age, bonus_malus, density,
    area, veh_brand, veh_gas, region, exposure

Usage:
    python python/export_portfolio.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DATA_PATH = Path(__file__).parent.parent / "data" / "freMTPL2freq.csv"
META_PATH = Path(__file__).parent.parent / "models" / "feature_metadata.json"
OUT_PATH = Path(__file__).parent.parent / "data" / "portfolio.csv"

NUMERIC_FEATURES = ["VehPower", "VehAge", "DrivAge", "BonusMalus", "Density"]
CATEGORICAL_FEATURES = ["Area", "VehBrand", "VehGas", "Region"]
EXPOSURE = "Exposure"


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Run python/data/download.py first."
        )
    if not META_PATH.exists():
        raise FileNotFoundError(
            f"Feature metadata not found at {META_PATH}. Run python/train.py first."
        )

    df = pd.read_csv(DATA_PATH)
    logger.info("Loaded %d rows from %s", len(df), DATA_PATH)

    with open(META_PATH) as f:
        metadata = json.load(f)

    # Apply the same preprocessing as train.py
    df[EXPOSURE] = df[EXPOSURE].clip(lower=1e-6, upper=1.0)
    df["BonusMalus"] = df["BonusMalus"].clip(upper=200)
    df["VehPower"] = df["VehPower"].clip(upper=15)

    for col in CATEGORICAL_FEATURES:
        classes = metadata["categorical_encodings"][col]
        encoding = {label: i for i, label in enumerate(classes)}
        # map(dict) is vectorised (C-level); fillna(0) handles any unseen labels
        df[col] = df[col].astype(str).map(encoding).fillna(0).astype(int)

    # Build output with lowercase column names matching the Rust Policy struct
    out = pd.DataFrame(
        {
            "veh_power":   df["VehPower"].astype(float),
            "veh_age":     df["VehAge"].astype(float),
            "driv_age":    df["DrivAge"].astype(float),
            "bonus_malus": df["BonusMalus"].astype(float),
            "density":     df["Density"].astype(float),
            "area":        df["Area"].astype(float),
            "veh_brand":   df["VehBrand"].astype(float),
            "veh_gas":     df["VehGas"].astype(float),
            "region":      df["Region"].astype(float),
            "exposure":    df[EXPOSURE].astype(float),
        }
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    logger.info("Saved %d policies to %s", len(out), OUT_PATH)
    logger.info("Total policy-years: %.1f", out["exposure"].sum())


if __name__ == "__main__":
    main()
