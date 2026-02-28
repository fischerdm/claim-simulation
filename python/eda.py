"""
eda.py
------
Exploratory data analysis for the freMTPL2freq dataset.
Produces plots saved under data/eda/.

Usage:
    python python/eda.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DATA_PATH = Path(__file__).parent.parent / "data" / "freMTPL2freq.csv"
EDA_DIR = Path(__file__).parent.parent / "data" / "eda"


def load() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Run python/data/download.py first."
        )
    return pd.read_csv(DATA_PATH)


def basic_stats(df: pd.DataFrame) -> None:
    print("=" * 60)
    print(f"Shape: {df.shape}")
    print(f"\nMissing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    print(f"\nDescribe:\n{df.describe()}")

    total_claims = df["ClaimNb"].sum()
    total_exposure = df["Exposure"].sum()
    print(f"\nTotal claims:        {total_claims:,.0f}")
    print(f"Total exposure:      {total_exposure:,.1f} years")
    print(f"Overall frequency:   {total_claims / total_exposure:.4f} claims/year")
    print(f"% policies 0 claims: {(df['ClaimNb'] == 0).mean() * 100:.1f}%")
    print("=" * 60)


def plot_claim_distribution(df: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    counts = df["ClaimNb"].value_counts().sort_index()
    axes[0].bar(counts.index, counts.values, color="steelblue", edgecolor="white")
    axes[0].set_title("Claim Count Distribution")
    axes[0].set_xlabel("Number of Claims")
    axes[0].set_ylabel("Number of Policies")
    axes[0].set_yscale("log")

    # Empirical frequency by exposure bucket
    df_copy = df.copy()
    df_copy["exposure_bucket"] = pd.cut(df_copy["Exposure"], bins=10)
    freq_by_exp = df_copy.groupby("exposure_bucket", observed=True).apply(
        lambda g: g["ClaimNb"].sum() / g["Exposure"].sum()
    )
    freq_by_exp.plot(kind="bar", ax=axes[1], color="darkorange", edgecolor="white")
    axes[1].set_title("Claim Frequency by Exposure Bucket")
    axes[1].set_xlabel("Exposure (years)")
    axes[1].set_ylabel("Frequency (claims/year)")
    axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(out_dir / "claim_distribution.png", dpi=150)
    plt.close()
    logger.info("Saved claim_distribution.png")


def plot_feature_frequency(df: pd.DataFrame, out_dir: Path) -> None:
    """Plot empirical claim frequency for each categorical / binned feature."""
    cat_features = ["Area", "VehPower", "VehBrand", "VehGas", "Region"]
    num_features = {"VehAge": 10, "DrivAge": 10, "BonusMalus": 15, "Density": 10}

    df_copy = df.copy()
    df_copy["freq"] = df_copy["ClaimNb"] / df_copy["Exposure"].clip(lower=0.01)

    n_plots = len(cat_features) + len(num_features)
    fig, axes = plt.subplots(3, 4, figsize=(20, 14))
    axes = axes.flatten()

    for i, feat in enumerate(cat_features):
        if feat not in df_copy.columns:
            continue
        grp = df_copy.groupby(feat, observed=True).apply(
            lambda g: g["ClaimNb"].sum() / g["Exposure"].sum()
        ).sort_values(ascending=False)
        grp.plot(kind="bar", ax=axes[i], color="steelblue", edgecolor="white")
        axes[i].set_title(f"Frequency by {feat}")
        axes[i].set_ylabel("Claims / Exposure")
        axes[i].tick_params(axis="x", rotation=45)

    for j, (feat, bins) in enumerate(num_features.items()):
        if feat not in df_copy.columns:
            continue
        ax = axes[len(cat_features) + j]
        df_copy["_bucket"] = pd.cut(df_copy[feat], bins=bins)
        grp = df_copy.groupby("_bucket", observed=True).apply(
            lambda g: g["ClaimNb"].sum() / g["Exposure"].sum()
        )
        grp.plot(kind="bar", ax=ax, color="darkorange", edgecolor="white")
        ax.set_title(f"Frequency by {feat}")
        ax.set_ylabel("Claims / Exposure")
        ax.tick_params(axis="x", rotation=45)

    for ax in axes[n_plots:]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.savefig(out_dir / "feature_frequency.png", dpi=150)
    plt.close()
    logger.info("Saved feature_frequency.png")


def plot_correlation(df: pd.DataFrame, out_dir: Path) -> None:
    num_cols = ["ClaimNb", "Exposure", "VehPower", "VehAge", "DrivAge", "BonusMalus", "Density"]
    available = [c for c in num_cols if c in df.columns]
    corr = df[available].corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig(out_dir / "correlation.png", dpi=150)
    plt.close()
    logger.info("Saved correlation.png")


def main() -> None:
    EDA_DIR.mkdir(parents=True, exist_ok=True)
    df = load()
    basic_stats(df)
    plot_claim_distribution(df, EDA_DIR)
    plot_feature_frequency(df, EDA_DIR)
    plot_correlation(df, EDA_DIR)
    logger.info("EDA complete. Plots saved to %s", EDA_DIR)


if __name__ == "__main__":
    main()