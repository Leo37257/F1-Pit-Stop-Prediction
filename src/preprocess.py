"""Preprocess task: split raw by Year, apply feature engineering, build ColumnTransformer."""

from __future__ import annotations

import logging

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from src import config
from src.feature_engineering import apply_feature_engineering

logger = logging.getLogger(__name__)


NUMERIC_FEATURES = [
    "PitStop",
    "LapNumber",
    "Stint",
    "TyreLife",
    "Position",
    "LapTime (s)",
    "LapTime_Delta",
    "Cumulative_Degradation",
    "RaceProgress",
    "Position_Change",
    "LapTime_lag1", "LapTime_lag2", "LapTime_lag3",
    "LapTime_Delta_lag1", "LapTime_Delta_lag2", "LapTime_Delta_lag3",
    "Position_lag1", "Position_lag2", "Position_lag3",
    "LapTime_roll3_mean", "LapTime_roll3_std",
    "LapTime_roll5_mean", "LapTime_roll5_std",
    "LapTimeDelta_roll3_mean", "LapTimeDelta_roll5_mean",
    "StintMin_sofar",
    "LapTime_vs_StintMin",
    "IsEarlyRace",
    "IsLateRace",
]

CATEGORICAL_FEATURES = ["Compound", "TyreLife_bucket"]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def run() -> tuple[str, str]:
    config.ensure_dirs()
    logger.info("Reading %s", config.RAW_PARQUET)
    df = pd.read_parquet(config.RAW_PARQUET)

    train = df[df["Year"].isin(config.TRAIN_YEARS)].reset_index(drop=True)
    test = df[df["Year"].isin(config.TEST_YEARS)].reset_index(drop=True)
    logger.info("Year split -> train=%s, test=%s", len(train), len(test))

    train_fe = apply_feature_engineering(train)
    test_fe = apply_feature_engineering(test)

    train_fe.to_parquet(config.PROCESSED_TRAIN, index=False)
    test_fe.to_parquet(config.PROCESSED_TEST, index=False)
    logger.info("Wrote %s and %s", config.PROCESSED_TRAIN, config.PROCESSED_TEST)
    return str(config.PROCESSED_TRAIN), str(config.PROCESSED_TEST)


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("numeric", "passthrough", NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    train_path, test_path = run()
    train_fe = pd.read_parquet(train_path)
    test_fe = pd.read_parquet(test_path)
    print(f"\nPreprocess complete:")
    print(f"  Train: {train_fe.shape} -> {train_path}")
    print(f"  Test:  {test_fe.shape}  -> {test_path}")
    print(f"\nFeature columns to model ({len(FEATURE_COLUMNS)}):")
    print(f"  numeric ({len(NUMERIC_FEATURES)}): {NUMERIC_FEATURES}")
    print(f"  categorical ({len(CATEGORICAL_FEATURES)}): {CATEGORICAL_FEATURES}")
