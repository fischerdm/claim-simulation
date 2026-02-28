"""
train.py
--------
Trains a LightGBM Poisson frequency model on the freMTPL2freq dataset.

Key design decisions:
- Poisson objective with log(Exposure) as offset → models claims/year directly
- Label encoding for categoricals (required for LightGBM cat feature support)
- BonusMalus is the most predictive feature and also used as the lagged
  "prior claims" proxy in the Rust simulation loop
- Model saved as LightGBM native format + feature metadata for ONNX export

Usage:
    python python/train.py
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

DATA_PATH = Path(__file__).parent.parent / "data" / "freMTPL2freq.csv"
MODELS_DIR = Path(__file__).parent.parent / "models"

# Features used for training — order matters for ONNX inference in Rust
NUMERIC_FEATURES = ["VehPower", "VehAge", "DrivAge", "BonusMalus", "Density"]
CATEGORICAL_FEATURES = ["Area", "VehBrand", "VehGas", "Region"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

TARGET = "ClaimNb"
EXPOSURE = "Exposure"
RANDOM_STATE = 42


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Run python/data/download.py first."
        )
    df = pd.read_csv(DATA_PATH)
    logger.info("Loaded %d rows from %s", len(df), DATA_PATH)
    return df


def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    """
    Encode categoricals with LabelEncoder and clip extreme values.
    Returns the processed DataFrame and a dict of fitted encoders (needed for ONNX).
    """
    df = df.copy()

    # Clip exposure to (0, 1] — policies are annual, exposure is fraction of year
    df[EXPOSURE] = df[EXPOSURE].clip(lower=1e-6, upper=1.0)

    # Clip ClaimNb outliers (rare but can destabilise Poisson deviance)
    df[TARGET] = df[TARGET].clip(upper=10)

    # Clip BonusMalus (scale 50–350, outliers above 200 are very rare)
    df["BonusMalus"] = df["BonusMalus"].clip(upper=200)

    # VehPower: clip at 15
    df["VehPower"] = df["VehPower"].clip(upper=15)

    # Label-encode categoricals
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
        # log(exposure) as offset so the model predicts annual frequency
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
        "objective": "poisson",
        "metric": "poisson",           # Poisson deviance
        "poisson_max_delta_step": 0.7, # regularisation for Poisson
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_child_samples": 50,       # avoid overfitting on sparse claim counts
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "verbose": -1,
        "n_jobs": -1,
        "seed": RANDOM_STATE,
    }

    callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=True),
        lgb.log_evaluation(period=50),
    ]

    booster = lgb.train(
        params=params,
        train_set=dtrain,
        num_boost_round=500,
        valid_sets=[dtrain, dval],
        valid_names=["train", "val"],
        callbacks=callbacks,
    )
    return booster


def evaluate(booster: lgb.Booster, val_df: pd.DataFrame) -> None:
    """Log key metrics on the validation set."""
    X_val = val_df[ALL_FEATURES]
    # Predict log(lambda), then add log(exposure) offset
    log_mu = booster.predict(X_val) + np.log(val_df[EXPOSURE])
    mu = np.exp(log_mu)

    y = val_df[TARGET].values
    exposure = val_df[EXPOSURE].values

    # Poisson deviance
    eps = 1e-10
    deviance = 2 * np.sum(y * np.log((y + eps) / (mu + eps)) - (y - mu))
    n = len(y)
    logger.info("Val Poisson deviance / n:  %.6f", deviance / n)

    # Predicted vs actual frequency
    pred_freq = mu.sum() / exposure.sum()
    actual_freq = y.sum() / exposure.sum()
    logger.info("Actual frequency:          %.4f", actual_freq)
    logger.info("Predicted frequency:       %.4f", pred_freq)

    # Feature importances
    importances = pd.Series(
        booster.feature_importance(importance_type="gain"),
        index=ALL_FEATURES,
    ).sort_values(ascending=False)
    logger.info("Feature importances (gain):\n%s", importances.to_string())


def save_artifacts(
    booster: lgb.Booster,
    encoders: dict[str, LabelEncoder],
) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Native LightGBM model (used by export_onnx.py)
    model_path = MODELS_DIR / "frequency_model.lgb"
    booster.save_model(str(model_path))
    logger.info("Saved LightGBM model to %s", model_path)

    # Feature metadata — used by Rust to build the feature vector in the right order
    metadata = {
        "feature_names": ALL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "categorical_encodings": {
            col: list(le.classes_) for col, le in encoders.items()
        },
    }
    meta_path = MODELS_DIR / "feature_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Saved feature metadata to %s", meta_path)


def main() -> None:
    df = load_data()
    df, encoders = preprocess(df)
    dtrain, dval, _, val_df = build_datasets(df)
    booster = train(dtrain, dval)
    evaluate(booster, val_df)
    save_artifacts(booster, encoders)
    logger.info("Training complete.")


if __name__ == "__main__":
    main()