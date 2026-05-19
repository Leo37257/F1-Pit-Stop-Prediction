"""Inference utilities for the exported MLflow champion model."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mlflow.pyfunc
import mlflow.sklearn
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

from src import config
from src.feature_engineering import apply_feature_engineering
from src.preprocess import FEATURE_COLUMNS

RAW_INPUT_COLUMNS = [c for c in config.RAW_COLUMNS if c != config.TARGET_COL]
ORDER_COL = "__input_order"


@dataclass(frozen=True)
class InferenceMetrics:
    rows: int
    f1_macro: float
    roc_auc: float


def load_pyfunc_model(model_uri: str | Path = config.CHAMPION_EXPORT_DIR) -> Any:
    return mlflow.pyfunc.load_model(str(model_uri))


def load_sklearn_model(model_uri: str | Path = config.CHAMPION_EXPORT_DIR) -> Any:
    return mlflow.sklearn.load_model(str(model_uri))


def prepare_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in RAW_INPUT_COLUMNS if c not in raw_df.columns]
    if missing:
        raise ValueError(f"Raw input missing expected columns: {missing}")

    work = raw_df.copy()
    work[ORDER_COL] = range(len(work))
    fe = apply_feature_engineering(work)
    fe = fe.sort_values(ORDER_COL).reset_index(drop=True)
    return fe[FEATURE_COLUMNS]


def predict_dataframe(raw_df: pd.DataFrame, model: Any | None = None) -> list[int]:
    fitted = model if model is not None else load_pyfunc_model()
    features = prepare_features(raw_df)
    return [int(x) for x in fitted.predict(features)]


def predict_records(records: list[dict[str, Any]], model: Any | None = None) -> list[int]:
    if not records:
        raise ValueError("records must contain at least one row")
    return predict_dataframe(pd.DataFrame.from_records(records), model=model)


def score_raw_dataframe(
    raw_df: pd.DataFrame,
    model: Any | None = None,
    year: int | None = None,
) -> InferenceMetrics:
    if config.TARGET_COL not in raw_df.columns:
        raise ValueError(f"Raw input must include {config.TARGET_COL} to calculate F1/AUC")

    scored = raw_df.copy()
    if year is not None:
        scored = scored[scored["Year"] == year].reset_index(drop=True)
    if scored.empty:
        raise ValueError("No rows available for scoring after filtering")

    fitted = model if model is not None else load_sklearn_model()
    features = prepare_features(scored)
    y_true = scored[config.TARGET_COL].astype(int)
    y_pred = fitted.predict(features)
    if not hasattr(fitted, "predict_proba"):
        raise ValueError("Model must expose predict_proba to calculate ROC-AUC")
    y_score = fitted.predict_proba(features)[:, 1]

    return InferenceMetrics(
        rows=int(len(scored)),
        f1_macro=float(f1_score(y_true, y_pred, average="macro")),
        roc_auc=float(roc_auc_score(y_true, y_score)),
    )


def _run_cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(config.DATA_DIR / "test.csv"))
    parser.add_argument("--output", default=str(config.DATA_DIR / "test_predictions.csv"))
    parser.add_argument("--model-uri", default=str(config.CHAMPION_EXPORT_DIR))
    parser.add_argument("--year", type=int, default=None)
    args = parser.parse_args()

    raw = pd.read_csv(args.input)
    if args.year is not None and "Year" in raw.columns:
        raw = raw[raw["Year"] == args.year].reset_index(drop=True)

    pyfunc_model = load_pyfunc_model(args.model_uri)
    predictions = predict_dataframe(raw, model=pyfunc_model)

    output = raw.copy()
    output["prediction"] = predictions
    output.to_csv(args.output, index=False)
    print(f"Wrote {len(output):,} predictions -> {args.output}")

    if config.TARGET_COL in raw.columns:
        sklearn_model = load_sklearn_model(args.model_uri)
        metrics = score_raw_dataframe(raw, model=sklearn_model)
        print(json.dumps(asdict(metrics), indent=2))
    else:
        print(f"Metrics skipped: {args.input} has no {config.TARGET_COL} column")


if __name__ == "__main__":
    _run_cli()
