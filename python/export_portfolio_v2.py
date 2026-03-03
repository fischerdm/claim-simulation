"""
export_portfolio_v2.py
----------------------
Exports the portfolio CSV for the v2 multi-year Rust simulation.

Differences vs export_portfolio.py (v1):
  - Drops bonus_malus column
  - Adds claims_hist_1, claims_hist_2, claims_hist_3 — the synthetic claim
    history seed for the rolling 3-year window in the Rust simulation loop
  - Samples N_PORTFOLIO policies to keep the multi-year simulation tractable

The three history columns represent claims in years t=-3 (oldest), t=-2, t=-1.
At simulation year t, the Rust engine computes:
    PriorClaims3Y = claims_hist_1 + claims_hist_2 + claims_hist_3
then shifts the window after each drawn year.

Column order matches portfolio_v2.rs / PortfolioRowV2:
    veh_power, veh_age, driv_age, density, area, veh_brand, veh_gas, region,
    exposure, claims_hist_1, claims_hist_2, claims_hist_3

Usage:
    python python/generate_history.py   # must run first
    python python/export_portfolio_v2.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DATA_PATH  = Path(__file__).parent.parent / "data" / "freMTPL2freq_with_history.csv"
META_PATH  = Path(__file__).parent.parent / "models" / "feature_metadata_v2.json"
OUT_PATH   = Path(__file__).parent.parent / "data" / "portfolio_v2.csv"

# Portfolio size for the multi-year simulation.
# 10 000 policies × 10 000 sims × 5 years is comfortably tractable in Rust.
# Use None to export the full dataset.
N_PORTFOLIO  = 10_000
RANDOM_STATE = 42

CATEGORICAL_FEATURES = ["Area", "VehBrand", "VehGas", "Region"]
EXPOSURE             = "Exposure"


def main() -> None:
    for p in (DATA_PATH, META_PATH):
        if not p.exists():
            raise FileNotFoundError(
                f"Required file not found: {p}\n"
                "Run python/generate_history.py and python/train_v2.py first."
            )

    with open(META_PATH) as f:
        metadata = json.load(f)

    df = pd.read_csv(DATA_PATH)
    logger.info("Loaded %d rows from %s", len(df), DATA_PATH)

    # Apply same preprocessing as train_v2.py
    df[EXPOSURE]   = df[EXPOSURE].clip(lower=1e-6, upper=1.0)
    df["VehPower"] = df["VehPower"].clip(upper=15)

    for col in CATEGORICAL_FEATURES:
        classes  = metadata["categorical_encodings"][col]
        encoding = {label: i for i, label in enumerate(classes)}
        df[col]  = df[col].astype(str).map(encoding).fillna(0).astype(int)

    # Sample a representative subset for the multi-year simulation
    if N_PORTFOLIO and len(df) > N_PORTFOLIO:
        df = df.sample(n=N_PORTFOLIO, random_state=RANDOM_STATE).reset_index(drop=True)
        logger.info("Sampled %d policies for v2 portfolio", len(df))

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
        "claims_hist_1": df["claims_hist_1"].astype(int),   # t = -3 (oldest)
        "claims_hist_2": df["claims_hist_2"].astype(int),   # t = -2
        "claims_hist_3": df["claims_hist_3"].astype(int),   # t = -1 (most recent)
    })

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)

    logger.info("Saved %d policies to %s", len(out), OUT_PATH)
    logger.info(
        "Prior claims per policy: mean=%.3f  max=%d",
        (out["claims_hist_1"] + out["claims_hist_2"] + out["claims_hist_3"]).mean(),
        (out["claims_hist_1"] + out["claims_hist_2"] + out["claims_hist_3"]).max(),
    )


if __name__ == "__main__":
    main()
