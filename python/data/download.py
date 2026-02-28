"""
download.py
-----------
Downloads the French Motor Third Party Liability frequency dataset (freMTPL2freq)
from OpenML (dataset ID 41214) and caches it locally under data/.

Usage:
    python python/data/download.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import openml
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# freMTPL2freq on OpenML
DATASET_ID = 41214
DATA_DIR = Path(__file__).parent.parent.parent / "data"
RAW_PATH = DATA_DIR / "freMTPL2freq.csv"


def download() -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if RAW_PATH.exists():
        logger.info("Dataset already cached at %s — skipping download.", RAW_PATH)
        return pd.read_csv(RAW_PATH)

    logger.info("Fetching freMTPL2freq from OpenML (dataset ID %d)...", DATASET_ID)
    dataset = openml.datasets.get_dataset(
        DATASET_ID,
        download_data=True,
        download_qualities=False,
        download_features_meta_data=False,
    )
    X, y, _, attribute_names = dataset.get_data(dataset_format="dataframe")

    # OpenML returns X without the target; y is ClaimNb
    df = X.copy()
    df["ClaimNb"] = y

    df.to_csv(RAW_PATH, index=False)
    logger.info("Saved %d rows to %s", len(df), RAW_PATH)
    return df


if __name__ == "__main__":
    df = download()
    print(df.head())
    print(f"\nShape: {df.shape}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nDtypes:\n{df.dtypes}")