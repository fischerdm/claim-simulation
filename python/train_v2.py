"""
train_v2.py
-----------
Trains the v2 LightGBM Poisson frequency model on the augmented dataset
produced by generate_history.py.

Changes vs. train.py (v1):
  - Drops BonusMalus (hard to project forward without a BM transition model)
  - Adds PriorClaims3Y (sum of simulated claims over the last 3 years)

The v2 model is the one used in the multi-year Rust simulation: at each
projection year the Rust engine recomputes λ per policy using the updated
VehAge, DrivAge, and rolling PriorClaims3Y window.

Outputs:
  models/frequency_model_v2.lgb
  models/feature_metadata_v2.json

Usage:
    python python/generate_history.py   # must run first
    python python/train_v2.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DATA_PATH  = Path(__file__).parent.parent / "data" / "freMTPL2freq_with_history.csv"
MODELS_DIR = Path(__file__).parent.parent / "models"

# Feature order matters — must match portfolio_v2.rs to_feature_row() in Rust.
# BonusMalus is dropped; PriorClaims3Y is appended at the end.
NUMERIC_FEATURES     = ["VehPower", "VehAge", "DrivAge", "Density", "PriorClaims3Y"]
CATEGORICAL_FEATURES = ["Area", "VehBrand", "VehGas", "Region"]
ALL_FEATURES         = NUMERIC_FEATURES + CATEGORICAL_FEATURES

TARGET       = "ClaimNb"
EXPOSURE     = "Exposure"
RANDOM_STATE = 42


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Augmented dataset not found at {DATA_PATH}. "
            "Run python/generate_history.py first."
        )
    df = pd.read_csv(DATA_PATH)
    logger.info("Loaded %d rows from %s", len(df), DATA_PATH)
    return df


def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    """
    The augmented dataset already has clipping and label encoding from
    generate_history.py, so we only need to re-encode the categoricals
    to get fresh LabelEncoder objects for the metadata file.
    """
    df = df.copy()

    # Re-apply clipping to be safe (idempotent)
    df[EXPOSURE]         = df[EXPOSURE].clip(lower=1e-6, upper=1.0)
    df["VehPower"]       = df["VehPower"].clip(upper=15)
    df[TARGET]           = df[TARGET].clip(upper=10)
    df["PriorClaims3Y"]  = df["PriorClaims3Y"].clip(upper=10)  # rare but possible outlier

    encoders: dict[str, LabelEncoder] = {}
    for col in CATEGORICAL_FEATURES:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        logger.info("Encoded %s → %d categories", col, len(le.classes_))

    return df, encoders


def build_datasets(
    df: pd.DataFrame,
) -> tuple[lgb.Dataset, lgb.Dataset, pd.DataFrame, pd.DataFrame]:
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=RANDOM_STATE)
    logger.info("Train: %d rows | Val: %d rows", len(train_df), len(val_df))

    dtrain = lgb.Dataset(
        data=train_df[ALL_FEATURES],
        label=train_df[TARGET],
        init_score=np.log(train_df[EXPOSURE]),
        categorical_feature=CATEGORICAL_FEATURES,
        free_raw_data=False,
    )
    dval = lgb.Dataset(
        data=val_df[ALL_FEATURES],
        label=val_df[TARGET],
        init_score=np.log(val_df[EXPOSURE]),
        categorical_feature=CATEGORICAL_FEATURES,
        reference=dtrain,
        free_raw_data=False,
    )
    return dtrain, dval, train_df, val_df


def train(dtrain: lgb.Dataset, dval: lgb.Dataset) -> lgb.Booster:
    params = {
        "objective":              "poisson",
        "metric":                 "poisson",
        "poisson_max_delta_step": 0.7,
        "learning_rate":          0.05,
        "num_leaves":             63,
        "min_child_samples":      50,
        "feature_fraction":       0.8,
        "bagging_fraction":       0.8,
        "bagging_freq":           5,
        "reg_alpha":              0.1,
        "reg_lambda":             1.0,
        "verbose":                -1,
        "n_jobs":                 -1,
        "seed":                   RANDOM_STATE,
    }
    callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=True),
        lgb.log_evaluation(period=50),
    ]
    return lgb.train(
        params=params,
        train_set=dtrain,
        num_boost_round=500,
        valid_sets=[dtrain, dval],
        valid_names=["train", "val"],
        callbacks=callbacks,
    )


def evaluate(booster: lgb.Booster, val_df: pd.DataFrame) -> None:
    X_val    = val_df[ALL_FEATURES]
    mu       = booster.predict(X_val) * val_df[EXPOSURE].values
    y        = val_df[TARGET].values
    exposure = val_df[EXPOSURE].values

    eps      = 1e-10
    deviance = 2 * np.sum(y * np.log((y + eps) / (mu + eps)) - (y - mu))
    logger.info("Val Poisson deviance / n:  %.6f", deviance / len(y))
    logger.info("Actual frequency:          %.4f", y.sum() / exposure.sum())
    logger.info("Predicted frequency:       %.4f", mu.sum() / exposure.sum())

    importances = pd.Series(
        booster.feature_importance(importance_type="gain"),
        index=ALL_FEATURES,
    ).sort_values(ascending=False)
    logger.info("Feature importances (gain):\n%s", importances.to_string())


def save_artifacts(booster: lgb.Booster, encoders: dict[str, LabelEncoder]) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODELS_DIR / "frequency_model_v2.lgb"
    booster.save_model(str(model_path))
    logger.info("Saved v2 LightGBM model to %s", model_path)

    metadata = {
        "feature_names":       ALL_FEATURES,
        "numeric_features":    NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "categorical_encodings": {
            col: list(le.classes_) for col, le in encoders.items()
        },
    }
    meta_path = MODELS_DIR / "feature_metadata_v2.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Saved v2 feature metadata to %s", meta_path)


def main() -> None:
    df = load_data()
    df, encoders = preprocess(df)
    dtrain, dval, _, val_df = build_datasets(df)
    booster = train(dtrain, dval)
    evaluate(booster, val_df)
    save_artifacts(booster, encoders)
    logger.info("v2 training complete.")


if __name__ == "__main__":
    main()
