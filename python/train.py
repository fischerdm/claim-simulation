"""
train.py
--------
Trains two LightGBM Poisson frequency models on the freMTPL2freq dataset:

  v1 — baseline: BonusMalus included, no claim history feature.
  v2 — simulation model: BonusMalus dropped, PriorClaims3Y added.
       Requires generate_history.py to have run first.

Key design decisions:
- Poisson objective with log(Exposure) as offset → models claims/year directly
- Label encoding for categoricals (required for LightGBM cat feature support)
- v2 adds PriorClaims3Y so the Rust simulation can recompute λ each year as
  the rolling claim window evolves

Outputs:
  models/frequency_model.lgb      + feature_metadata.json       (v1)
  models/frequency_model_v2.lgb   + feature_metadata_v2.json    (v2)

Usage:
    python python/generate_history.py   # prerequisite for v2
    python python/train.py
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

BASE_DIR   = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"

CATEGORICAL_FEATURES = ["Area", "VehBrand", "VehGas", "Region"]
TARGET               = "ClaimNb"
EXPOSURE             = "Exposure"
RANDOM_STATE         = 42


@dataclass
class ModelSpec:
    """All version-specific configuration for one training run."""
    label:            str               # display name used in log messages
    data_path:        Path
    numeric_features: list[str]         # order must match portfolio export and Rust structs
    extra_clip:       dict[str, float]  # version-specific clips on top of the shared ones
    model_suffix:     str               # "" → frequency_model.lgb, "_v2" → frequency_model_v2.lgb


V1 = ModelSpec(
    label="v1 (BonusMalus, no claim history)",
    data_path=BASE_DIR / "data" / "freMTPL2freq.csv",
    numeric_features=["VehPower", "VehAge", "DrivAge", "BonusMalus", "Density"],
    extra_clip={"BonusMalus": 200},
    model_suffix="",
)

V2 = ModelSpec(
    label="v2 (PriorClaims3Y, no BonusMalus)",
    data_path=BASE_DIR / "data" / "freMTPL2freq_with_history.csv",
    numeric_features=["VehPower", "VehAge", "DrivAge", "Density", "PriorClaims3Y"],
    extra_clip={"PriorClaims3Y": 10},
    model_suffix="_v2",
)


# ── Shared helpers ────────────────────────────────────────────────────────────

def preprocess(
    df: pd.DataFrame, spec: ModelSpec
) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    """Clip extreme values and label-encode categoricals."""
    df = df.copy()
    df[EXPOSURE]   = df[EXPOSURE].clip(lower=1e-6, upper=1.0)
    df["VehPower"] = df["VehPower"].clip(upper=15)
    df[TARGET]     = df[TARGET].clip(upper=10)
    for col, upper in spec.extra_clip.items():
        df[col] = df[col].clip(upper=upper)

    encoders: dict[str, LabelEncoder] = {}
    for col in CATEGORICAL_FEATURES:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        logger.info("  Encoded %s → %d categories", col, len(le.classes_))

    return df, encoders


def build_datasets(
    df: pd.DataFrame, spec: ModelSpec
) -> tuple[lgb.Dataset, lgb.Dataset, pd.DataFrame, pd.DataFrame]:
    all_features = spec.numeric_features + CATEGORICAL_FEATURES
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=RANDOM_STATE)
    logger.info("  Train: %d rows | Val: %d rows", len(train_df), len(val_df))

    dtrain = lgb.Dataset(
        data=train_df[all_features],
        label=train_df[TARGET],
        init_score=np.log(train_df[EXPOSURE]),
        categorical_feature=CATEGORICAL_FEATURES,
        free_raw_data=False,
    )
    dval = lgb.Dataset(
        data=val_df[all_features],
        label=val_df[TARGET],
        init_score=np.log(val_df[EXPOSURE]),
        categorical_feature=CATEGORICAL_FEATURES,
        reference=dtrain,
        free_raw_data=False,
    )
    return dtrain, dval, train_df, val_df


def fit(dtrain: lgb.Dataset, dval: lgb.Dataset) -> lgb.Booster:
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


def evaluate(booster: lgb.Booster, val_df: pd.DataFrame, spec: ModelSpec) -> None:
    all_features = spec.numeric_features + CATEGORICAL_FEATURES
    mu           = booster.predict(val_df[all_features]) * val_df[EXPOSURE].values
    y            = val_df[TARGET].values
    exposure     = val_df[EXPOSURE].values

    eps      = 1e-10
    deviance = 2 * np.sum(y * np.log((y + eps) / (mu + eps)) - (y - mu))
    logger.info("  Val Poisson deviance / n:  %.6f", deviance / len(y))
    logger.info("  Actual frequency:          %.4f", y.sum() / exposure.sum())
    logger.info("  Predicted frequency:       %.4f", mu.sum() / exposure.sum())

    importances = pd.Series(
        booster.feature_importance(importance_type="gain"),
        index=all_features,
    ).sort_values(ascending=False)
    logger.info("  Feature importances (gain):\n%s", importances.to_string())


def save_artifacts(
    booster: lgb.Booster,
    encoders: dict[str, LabelEncoder],
    spec: ModelSpec,
) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    all_features = spec.numeric_features + CATEGORICAL_FEATURES

    model_path = MODELS_DIR / f"frequency_model{spec.model_suffix}.lgb"
    booster.save_model(str(model_path))
    logger.info("  Saved model to %s", model_path)

    metadata = {
        "feature_names":         all_features,
        "numeric_features":      spec.numeric_features,
        "categorical_features":  CATEGORICAL_FEATURES,
        "categorical_encodings": {col: list(le.classes_) for col, le in encoders.items()},
    }
    meta_path = MODELS_DIR / f"feature_metadata{spec.model_suffix}.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("  Saved metadata to %s", meta_path)


# ── Per-model entry point ─────────────────────────────────────────────────────

def train_one(spec: ModelSpec) -> None:
    logger.info("── Training %s ──", spec.label)
    if not spec.data_path.exists():
        hint = (
            "Run python/generate_history.py first."
            if "history" in spec.data_path.name
            else "Run python/data/download.py first."
        )
        raise FileNotFoundError(f"Dataset not found at {spec.data_path}. {hint}")

    df = pd.read_csv(spec.data_path)
    logger.info("  Loaded %d rows from %s", len(df), spec.data_path)

    df, encoders             = preprocess(df, spec)
    dtrain, dval, _, val_df  = build_datasets(df, spec)
    booster                  = fit(dtrain, dval)
    evaluate(booster, val_df, spec)
    save_artifacts(booster, encoders, spec)
    logger.info("  Done.\n")


def main() -> None:
    train_one(V1)
    train_one(V2)


if __name__ == "__main__":
    main()
