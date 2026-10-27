"""Natural drift: how 2025 differs from the 2022-2024 training years, without any injected change.

The four scenarios in model_monitoring.py inject drift on purpose. This script asks the production question:
on the real 2025 season, which features moved, is that more than the normal year-to-year movement, and in which
slices (tyre compound, race) does the champion lose accuracy?

    python monitoring/natural_drift.py                  # report only, always exits 0 (baseline: 2024 season)
    python monitoring/natural_drift.py --reference-years 2022,2023,2024   # pooled baseline instead
    python monitoring/natural_drift.py --fail-on alert  # exit 1 if macro-F1 on current data drops past the alert line
Feature drift and weak slices are reported as warnings but never fail the run: features drift every season
(see year_over_year) and the model has held up; what should stop a pipeline is lost accuracy.
Outputs (monitoring_reports/): natural_drift.json, drift_by_feature.csv, slice_performance.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from monitoring import drift_checks as dc  # noqa: E402
from src.inference import apply_threshold  # noqa: E402

OUT_DIR = ROOT / "monitoring_reports"
NOT_FEATURES = {"id", "Year", config.TARGET_COL}
MIN_SLICE_ROWS = 500


def feature_lists(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    categorical = [c for c in config.CATEGORICAL_COLUMNS if c in df.columns]
    numeric = [c for c in df.columns if c not in NOT_FEATURES and c not in categorical
               and pd.api.types.is_numeric_dtype(df[c])]
    return numeric, categorical


def year_over_year(df: pd.DataFrame, numeric: list[str], categorical: list[str]) -> list[dict]:
    """PSI between consecutive seasons: the 'normal' amount of drift to compare 2025 against."""
    years = sorted(df["Year"].unique())
    out = []
    for a, b in zip(years, years[1:]):
        t = dc.feature_drift_table(df[df.Year == a], df[df.Year == b], numeric, categorical)
        out.append({"from": int(a), "to": int(b), "median_psi": round(float(t.psi.median()), 4),
                    "features_alert": int((t.status == "alert").sum()), "features_warn": int((t.status == "warn").sum())})
    return out


def scores(y_true, y_pred, y_score) -> dict:
    auc = roc_auc_score(y_true, y_score) if len(set(y_true)) > 1 else float("nan")
    return {"rows": int(len(y_true)), "f1_macro": round(float(f1_score(y_true, y_pred, average="macro")), 4),
            "roc_auc": round(float(auc), 4), "pit_rate_actual": round(float(np.mean(y_true)), 4),
            "pit_rate_predicted": round(float(np.mean(y_pred)), 4)}


def slice_performance(test: pd.DataFrame, y_pred, y_score, overall_f1: float) -> pd.DataFrame:
    rows = []
    for col in ("Compound", "Race"):
        for value, idx in test.groupby(col).groups.items():
            if len(idx) < MIN_SLICE_ROWS:
                continue
            pos = test.index.get_indexer(idx)
            s = scores(test[config.TARGET_COL].to_numpy()[pos], y_pred[pos], y_score[pos])
            s.update({"slice": col, "value": value, **dc.performance_status(overall_f1, s["f1_macro"])})
            rows.append(s)
    cols = ["slice", "value", "rows", "f1_macro", "roc_auc", "pit_rate_actual", "pit_rate_predicted", "f1_drop", "status"]
    return pd.DataFrame(rows)[cols].sort_values(["slice", "f1_macro"]).reset_index(drop=True)


DEFAULT_REFERENCE_YEARS = (2024,)   # latest clean season; the pooled 2022-2024 baseline includes the synthetic 2023


def main(reference_years: tuple[int, ...] = DEFAULT_REFERENCE_YEARS) -> str:
    import mlflow.sklearn

    train = pd.read_parquet(config.PROCESSED_TRAIN)
    test = pd.read_parquet(config.PROCESSED_TEST).reset_index(drop=True)
    numeric, categorical = feature_lists(train)

    reference = train[train.Year.isin(reference_years)]
    drift = dc.feature_drift_table(reference, test, numeric, categorical)
    # keep the pooled comparison for context: it shows how much the synthetic 2023 season inflates the alerts
    drift_pooled = dc.feature_drift_table(train, test, numeric, categorical)
    yoy = year_over_year(pd.concat([train, test], ignore_index=True), numeric, categorical)

    model = mlflow.sklearn.load_model(str(config.CHAMPION_EXPORT_DIR))
    X = test.drop(columns=[config.TARGET_COL])
    y_pred, y_score = apply_threshold(model, X).astype(int), model.predict_proba(X)[:, 1]   # served cut-off
    overall = scores(test[config.TARGET_COL].to_numpy(), y_pred, y_score)
    slices = slice_performance(test, y_pred, y_score, overall["f1_macro"])

    meta = json.loads((config.CHAMPION_EXPORT_DIR / "CHAMPION.json").read_text())["metrics"]
    champion_f1 = meta.get("test_macro_f1_at_decision_threshold", meta["test_macro_f1"])
    perf_vs_champion = dc.performance_status(round(champion_f1, 4), overall["f1_macro"])

    label_rate = {int(y): round(float(r), 4) for y, r in
                  pd.concat([train, test]).groupby("Year")[config.TARGET_COL].mean().items()}
    summary = {
        "reference": f"processed train, season(s) {', '.join(map(str, reference_years))}",
        "current": "processed test, season 2025",
        "thresholds": {"psi_warn": dc.PSI_WARN, "psi_alert": dc.PSI_ALERT,
                       "f1_drop_warn": dc.F1_DROP_WARN, "f1_drop_alert": dc.F1_DROP_ALERT},
        "features_checked": int(len(drift)),
        "features_by_status": drift.status.value_counts().to_dict(),
        "top_drifted": drift.head(5)[["feature", "psi", "ks", "status"]].round(4).to_dict("records"),
        "features_by_status_vs_pooled_2022_2024": drift_pooled.status.value_counts().to_dict(),
        "year_over_year": yoy,
        "pit_rate_by_year": label_rate,
        "overall_2025": overall,
        "worst_slices": slices.sort_values("f1_macro").head(5).round(4).to_dict("records"),
        "slices_by_status": slices.status.value_counts().to_dict(),
        "performance_vs_champion": {"champion_test_macro_f1": round(champion_f1, 4), **perf_vs_champion},
        "status": dc.worst_status(list(drift.status) + list(slices.status)),
        "gate_status": perf_vs_champion["status"],
    }
    OUT_DIR.mkdir(exist_ok=True)
    drift.round(4).to_csv(OUT_DIR / "drift_by_feature.csv", index=False)
    slices.round(4).to_csv(OUT_DIR / "slice_performance.csv", index=False)
    (OUT_DIR / "natural_drift.json").write_text(json.dumps(summary, indent=2, default=str))

    print(f"features vs pooled 2022-2024: {summary['features_by_status_vs_pooled_2022_2024']}")
    print(f"features: {summary['features_by_status']}  top: " +
          ", ".join(f"{r['feature']} {r['psi']:.3f}" for r in summary["top_drifted"]))
    print("year over year:", [(y['from'], y['to'], y['median_psi'], y['features_alert']) for y in yoy])
    print("pit rate by year:", label_rate)
    print("2025 overall:", overall)
    print("worst slices:", [(r['slice'], r['value'], r['rows'], r['f1_macro'], r['status']) for r in summary["worst_slices"]])
    print("slices by status:", summary["slices_by_status"], "| overall status:", summary["status"])
    print("gate (macro-F1 vs champion):", summary["performance_vs_champion"])
    return summary["gate_status"]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Natural drift and slice performance on the current season")
    parser.add_argument("--reference-years", default=",".join(map(str, DEFAULT_REFERENCE_YEARS)),
                        help="comma-separated training seasons to use as the drift baseline")
    parser.add_argument("--fail-on", choices=["never", "warn", "alert"], default="never",
                        help="exit 1 when macro-F1 versus the champion reaches this level")
    args = parser.parse_args()
    fail_on = args.fail_on
    gate = main(tuple(int(y) for y in args.reference_years.split(",")))
    rank, level = {"ok": 1, "warn": 2, "alert": 3}, {"never": 99, "warn": 2, "alert": 3}
    sys.exit(1 if rank.get(gate, 0) >= level[fail_on] else 0)
