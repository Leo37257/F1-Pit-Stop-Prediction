"""Do slice-targeted features help? Compare feature sets out-of-fold, then test the winner once on 2025.

Monitoring found the champion weakest on SOFT-tyre laps and on a handful of races. Candidate fixes:
  V1  + Race as a categorical feature (track-specific strategy)
  V2  + the driver's race history so far: pits taken, distinct compounds used, laps into the current stint
  V3  V1 + V2
Every variant uses the champion's XGBoost hyper-parameters and the same protocol as threshold_tuning.py:
5-fold GroupKFold by (Year, Race) on 2022-2024, threshold tuned on out-of-fold 2022 + 2024, and the selection
made on out-of-fold macro-F1 only. 2025 is scored once, for the baseline and the selected variant.

    python evaluation/feature_experiments.py
Output: evaluation/feature_experiments.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.threshold_tuning import TUNE_YEARS, best_threshold  # noqa: E402
from src import config  # noqa: E402
from src.preprocess import CATEGORICAL_FEATURES, NUMERIC_FEATURES  # noqa: E402

HISTORY = ["PitsSoFar", "CompoundsUsedSoFar", "LapsIntoStint"]
VARIANTS = {
    "V0_baseline": (NUMERIC_FEATURES, CATEGORICAL_FEATURES),
    "V1_race": (NUMERIC_FEATURES, CATEGORICAL_FEATURES + ["Race"]),
    "V2_history": (NUMERIC_FEATURES + HISTORY, CATEGORICAL_FEATURES),
    "V3_race_history": (NUMERIC_FEATURES + HISTORY, CATEGORICAL_FEATURES + ["Race"]),
}


def add_history(df: pd.DataFrame) -> pd.DataFrame:
    """Per driver and race, using only laps before the current one (no look-ahead)."""
    out = df.sort_values(["Year", "Race", "Driver", "LapNumber"]).copy()
    g = out.groupby(["Year", "Race", "Driver"], sort=False)
    out["PitsSoFar"] = g["PitStop"].transform(lambda s: s.shift(1, fill_value=0).cumsum())
    out["CompoundsUsedSoFar"] = g["Compound"].transform(lambda s: (~s.duplicated()).cumsum())
    out["LapsIntoStint"] = out["LapNumber"] - out.groupby(["Year", "Race", "Driver", "Stint"])["LapNumber"].transform("min")
    return out.sort_index()


def pipeline(estimator, numeric, categorical) -> Pipeline:
    pre = ColumnTransformer([("numeric", "passthrough", numeric),
                             ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical)],
                            remainder="drop")
    return Pipeline([("pre", pre), ("clf", clone(estimator))])


def slice_f1(frame: pd.DataFrame, y, pred, col, value) -> float:
    m = (frame[col] == value).to_numpy()
    return round(float(f1_score(y[m], pred[m], average="macro")), 4)


def main() -> None:
    import mlflow.sklearn

    estimator = mlflow.sklearn.load_model(str(config.CHAMPION_EXPORT_DIR)).named_steps["clf"].model.estimator
    train = add_history(pd.read_parquet(config.PROCESSED_TRAIN).reset_index(drop=True))
    test = add_history(pd.read_parquet(config.PROCESSED_TEST).reset_index(drop=True))
    y = train[config.TARGET_COL].to_numpy()
    groups = train["Year"].astype(str) + "|" + train["Race"].astype(str)
    folds = list(GroupKFold(n_splits=5).split(train, y, groups))
    tune = train["Year"].isin(TUNE_YEARS).to_numpy()

    results = {}
    for name, (num, cat) in VARIANTS.items():
        t0, oof = time.time(), np.zeros(len(train))
        for tr, va in folds:
            m = pipeline(estimator, num, cat).fit(train.iloc[tr], y[tr])
            oof[va] = m.predict_proba(train.iloc[va])[:, 1]
        thr, f1 = best_threshold(y[tune], oof[tune])
        pred = (oof >= thr).astype(int)
        clean = train[tune]
        results[name] = {"threshold": thr, "oof_macro_f1": round(f1, 4),
                         "oof_soft_f1": slice_f1(clean, y[tune], pred[tune], "Compound", "SOFT"),
                         "seconds": round(time.time() - t0)}
        print(name, results[name], flush=True)

    best = max(results, key=lambda k: results[k]["oof_macro_f1"])
    yt = test[config.TARGET_COL].to_numpy()
    for name in dict.fromkeys(["V0_baseline", best]):
        num, cat = VARIANTS[name]
        m = pipeline(estimator, num, cat).fit(train, y)            # refit on all 2022-2024, like the champion
        p = m.predict_proba(test)[:, 1]
        pred = (p >= results[name]["threshold"]).astype(int)
        races = {r: slice_f1(test, yt, pred, "Race", r) for r in ["Miami Grand Prix", "Australian Grand Prix",
                                                                 "Austrian Grand Prix", "Mexico City Grand Prix"]}
        results[name]["test_2025"] = {"macro_f1": round(float(f1_score(yt, pred, average="macro")), 4),
                                      "soft_f1": slice_f1(test, yt, pred, "Compound", "SOFT"),
                                      "medium_f1": slice_f1(test, yt, pred, "Compound", "MEDIUM"), "races": races}
        print(name, "2025:", results[name]["test_2025"], flush=True)

    summary = {"protocol": "5-fold GroupKFold by (Year, Race) on 2022-2024; threshold and selection on out-of-fold "
                           "2022 + 2024 only; 2025 scored once for the baseline and the selected variant",
               "selected": best, "variants": results}
    (ROOT / "evaluation" / "feature_experiments.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
