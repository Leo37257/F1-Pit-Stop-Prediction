"""Ingest task: read the raw Kaggle CSV, validate schema, write parquet.

Data source: Kaggle Playground Series S5E6 — F1 Pit-Stop Prediction.
To pull from scratch:
    kaggle competitions download -c playground-series-s5e6 -p data/ && unzip data/*.zip -d data/

This script assumes the CSV is already at `data/train.csv` (Member A's pull).
"""

from __future__ import annotations

import logging

import pandas as pd

from src import config

logger = logging.getLogger(__name__)


def run() -> str:
    config.ensure_dirs()
    if not config.RAW_TRAIN_CSV.exists():
        raise FileNotFoundError(
            f"Raw CSV not found at {config.RAW_TRAIN_CSV}. "
            "Pull from Kaggle: `kaggle competitions download -c playground-series-s5e6 -p data/` "
            "and unzip into data/, or restore via `dvc checkout`."
        )
    logger.info("Reading raw CSV from %s", config.RAW_TRAIN_CSV)
    df = pd.read_csv(config.RAW_TRAIN_CSV)
    _validate_schema(df)
    df.to_parquet(config.RAW_PARQUET, index=False)
    logger.info("Wrote %s rows to %s", len(df), config.RAW_PARQUET)
    return str(config.RAW_PARQUET)


def _validate_schema(df: pd.DataFrame) -> None:
    missing = [c for c in config.RAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Raw CSV missing expected columns: {missing}")
    extra = [c for c in df.columns if c not in config.RAW_COLUMNS]
    if extra:
        raise ValueError(f"Raw CSV has unexpected columns: {extra}")
    years = sorted(df["Year"].unique().tolist())
    expected_years = list(config.TRAIN_YEARS) + list(config.TEST_YEARS)
    if years != expected_years:
        raise ValueError(f"Unexpected Year values: {years}, expected {expected_years}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    path = run()
    df = pd.read_parquet(path)
    print(f"\nIngest complete: {len(df):,} rows, {len(df.columns)} cols -> {path}")
    print(df.head())
