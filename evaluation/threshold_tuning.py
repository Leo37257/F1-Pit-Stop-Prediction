"""Pick the decision threshold for the champion without touching the test season, then check calibration.

The champion was refit on all of 2022-2024, so its own predictions on those seasons are in-sample. Instead the
same configuration (same preprocessor, same XGBoost hyper-parameters) is refit in 5 folds grouped by race, which
gives out-of-fold probabilities for every training lap. The threshold that maximises macro-F1 on those is chosen
using 2022 and 2024 only: 2023 is a synthetic season with a 1% pit rate and would drag the threshold toward a
regime the test season does not have. The chosen threshold is then applied once to the champion on 2025.

    python evaluation/threshold_tuning.py
Outputs (evaluation/): threshold_tuning.json, reliability_2025.csv
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import brier_score_loss, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import config  # noqa: E402

OUT = ROOT / "evaluation"
GRID = np.round(np.arange(0.10, 0.901, 0.01), 2)
TUNE_YEARS = (2022, 2024)


def best_threshold(y, p, grid=GRID) -> tuple[float, float]:
    scores = [f1_score(y, (p >= t).astype(int), average="macro") for t in grid]
    i = int(np.argmax(scores))
    return float(grid[i]), float(scores[i])


def reliability(y, p, bins=10) -> pd.DataFrame:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    df = pd.DataFrame({"bin": idx, "y": y, "p": p})
    g = df.groupby("bin").agg(rows=("y", "size"), mean_predicted=("p", "mean"), observed_rate=("y", "mean"))
    g["low"], g["high"] = edges[g.index], edges[g.index + 1]
    return g.reset_index(drop=True)[["low", "high", "rows", "mean_predicted", "observed_rate"]]


def main() -> None:
    import mlflow.sklearn

    champion = mlflow.sklearn.load_model(str(config.CHAMPION_EXPORT_DIR))
    pre, automl = champion.named_steps["pre"], champion.named_steps["clf"]
    estimator = automl.model.estimator          # the XGBClassifier FLAML selected, with its tuned parameters

    train = pd.read_parquet(config.PROCESSED_TRAIN).reset_index(drop=True)
    test = pd.read_parquet(config.PROCESSED_TEST).reset_index(drop=True)
    X, y = train.drop(columns=[config.TARGET_COL]), train[config.TARGET_COL].to_numpy()
    groups = train["Year"].astype(str) + "|" + train["Race"].astype(str)

    oof = np.zeros(len(train))
    t0 = time.time()
    for k, (tr, va) in enumerate(GroupKFold(n_splits=5).split(X, y, groups)):
        fold = Pipeline([("pre", clone(pre)), ("clf", clone(estimator))])
        fold.fit(X.iloc[tr], y[tr])
        oof[va] = fold.predict_proba(X.iloc[va])[:, 1]
        print(f"fold {k + 1}/5 done ({time.time() - t0:.0f}s)", flush=True)

    mask = train["Year"].isin(TUNE_YEARS).to_numpy()
    thr, oof_f1_at_thr = best_threshold(y[mask], oof[mask])
    oof_f1_at_05 = f1_score(y[mask], (oof[mask] >= 0.5).astype(int), average="macro")

    Xt, yt = test.drop(columns=[config.TARGET_COL]), test[config.TARGET_COL].to_numpy()
    pt = champion.predict_proba(Xt)[:, 1]
    res = {}
    for name, t in (("default_0.5", 0.5), ("tuned", thr)):
        pred = (pt >= t).astype(int)
        res[name] = {"threshold": t, "macro_f1": round(float(f1_score(yt, pred, average="macro")), 4),
                     "pit_rate_predicted": round(float(pred.mean()), 4)}
    summary = {
        "method": "5-fold GroupKFold by (Year, Race) on 2022-2024 with the champion's configuration; "
                  "threshold chosen on out-of-fold predictions for 2022 and 2024; evaluated once on 2025",
        "chosen_threshold": thr,
        "oof_macro_f1": {"at_0.5": round(float(oof_f1_at_05), 4), "at_chosen": round(oof_f1_at_thr, 4)},
        "test_2025": res,
        "test_2025_macro_f1_gain": round(res["tuned"]["macro_f1"] - res["default_0.5"]["macro_f1"], 4),
        "test_2025_pit_rate_actual": round(float(yt.mean()), 4),
        "calibration_2025": {"brier": round(float(brier_score_loss(yt, pt)), 4),
                             "roc_auc": round(float(roc_auc_score(yt, pt)), 4),
                             "mean_predicted": round(float(pt.mean()), 4), "observed_rate": round(float(yt.mean()), 4)},
        "calibration_oof_2022_2024_clean": {"brier": round(float(brier_score_loss(y[mask], oof[mask])), 4),
                                            "mean_predicted": round(float(oof[mask].mean()), 4),
                                            "observed_rate": round(float(y[mask].mean()), 4)},
    }
    OUT.mkdir(exist_ok=True)
    (OUT / "threshold_tuning.json").write_text(json.dumps(summary, indent=2))
    reliability(yt, pt).round(4).to_csv(OUT / "reliability_2025.csv", index=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
