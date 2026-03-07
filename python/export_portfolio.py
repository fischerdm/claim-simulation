"""
export_portfolio.py
-------------------
Preprocesses freMTPL2freq and exports portfolio CSVs for the Rust simulation engine.

Exports two portfolios in one run:
  v1 — data/portfolio.csv      full dataset, includes bonus_malus
  v2 — data/portfolio_v2.csv   10K sampled policies, includes claims history seed

The Rust engine reads float32 values directly and does not do label encoding.
This script applies the same preprocessing as train.py (clipping, label encoding
using the saved category orderings from feature_metadata.json).

v1 column order matches portfolio.rs / Policy:
    veh_power, veh_age, driv_age, bonus_malus, density,
    area, veh_brand, veh_gas, region, exposure

v2 column order matches portfolio_v2.rs / PolicyV2:
    veh_power, veh_age, driv_age, density,
    area, veh_brand, veh_gas, region, exposure,
    claims_hist_1, claims_hist_2, claims_hist_3

Usage:
    python python/generate_history.py   # prerequisite for v2
    python python/export_portfolio.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

BASE_DIR = Path(__file__).parent.parent

CATEGORICAL_FEATURES = ["Area", "VehBrand", "VehGas", "Region"]
EXPOSURE             = "Exposure"

# Number of policies to sample for the v2 multi-year simulation.
# 10K policies × 10K sims × 5 years is comfortably tractable in Rust.
N_PORTFOLIO_V2 = 10_000
RANDOM_STATE   = 42


def apply_preprocessing(df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    """Apply clipping and label encoding shared by both portfolio exports."""
    df = df.copy()
    df[EXPOSURE]   = df[EXPOSURE].clip(lower=1e-6, upper=1.0)
    df["VehPower"] = df["VehPower"].clip(upper=15)

    for col in CATEGORICAL_FEATURES:
        classes  = metadata["categorical_encodings"][col]
        encoding = {label: i for i, label in enumerate(classes)}
        df[col]  = df[col].astype(str).map(encoding).fillna(0).astype(int)

    return df


def export_v1(df: pd.DataFrame) -> None:
    """Export the full portfolio for the v1 single-year simulation."""
    out_path = BASE_DIR / "data" / "portfolio.csv"
    out = pd.DataFrame({
        "veh_power":   df["VehPower"].astype(float),
        "veh_age":     df["VehAge"].astype(float),
        "driv_age":    df["DrivAge"].astype(float),
        "bonus_malus": df["BonusMalus"].clip(upper=200).astype(float),
        "density":     df["Density"].astype(float),
        "area":        df["Area"].astype(float),
        "veh_brand":   df["VehBrand"].astype(float),
        "veh_gas":     df["VehGas"].astype(float),
        "region":      df["Region"].astype(float),
        "exposure":    df[EXPOSURE].astype(float),
    })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    logger.info(
        "v1 portfolio: %d policies → %s  (%.1f total policy-years)",
        len(out), out_path, out["exposure"].sum(),
    )


def export_v2(df: pd.DataFrame) -> None:
    """Export a sampled portfolio with claim history for the v2 multi-year simulation."""
    out_path = BASE_DIR / "data" / "portfolio_v2.csv"

    if N_PORTFOLIO_V2 and len(df) > N_PORTFOLIO_V2:
        df = df.sample(n=N_PORTFOLIO_V2, random_state=RANDOM_STATE).reset_index(drop=True)
        logger.info("v2 portfolio: sampled %d policies from full dataset", len(df))

    out = pd.DataFrame({
        "veh_power":     df["VehPower"].astype(float),
        "veh_age":       df["VehAge"].astype(float),
        "driv_age":      df["DrivAge"].astype(float),
        "density":       df["Density"].astype(float),
        "area":          df["Area"].astype(float),
        "veh_brand":     df["VehBrand"].astype(float),
        "veh_gas":       df["VehGas"].astype(float),
        "region":        df["Region"].astype(float),
        "exposure":      df[EXPOSURE].astype(float),
        "claims_hist_1": df["claims_hist_1"].astype(int),  # t = -3 (oldest)
        "claims_hist_2": df["claims_hist_2"].astype(int),  # t = -2
        "claims_hist_3": df["claims_hist_3"].astype(int),  # t = -1 (most recent)
    })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    prior_3y = out["claims_hist_1"] + out["claims_hist_2"] + out["claims_hist_3"]
    logger.info(
        "v2 portfolio: %d policies → %s  (prior claims mean=%.3f  max=%d)",
        len(out), out_path, prior_3y.mean(), prior_3y.max(),
    )


def main() -> None:
    # v1 — uses original dataset and v1 metadata
    meta_v1_path = BASE_DIR / "models" / "feature_metadata.json"
    data_v1_path = BASE_DIR / "data" / "freMTPL2freq.csv"
    with open(meta_v1_path) as f:
        meta_v1 = json.load(f)
    df_v1 = pd.read_csv(data_v1_path)
    logger.info("Loaded %d rows for v1", len(df_v1))
    df_v1 = apply_preprocessing(df_v1, meta_v1)
    export_v1(df_v1)

    # v2 — uses augmented dataset (claims history columns added by generate_history.py)
    meta_v2_path = BASE_DIR / "models" / "feature_metadata_v2.json"
    data_v2_path = BASE_DIR / "data" / "freMTPL2freq_with_history.csv"
    if not data_v2_path.exists():
        logger.warning(
            "Augmented dataset not found at %s — skipping v2 export.\n"
            "Run python/generate_history.py first.",
            data_v2_path,
        )
        return
    with open(meta_v2_path) as f:
        meta_v2 = json.load(f)
    df_v2 = pd.read_csv(data_v2_path)
    logger.info("Loaded %d rows for v2", len(df_v2))
    df_v2 = apply_preprocessing(df_v2, meta_v2)
    export_v2(df_v2)


if __name__ == "__main__":
    main()
