"""
Model Monitoring for F1 Pit-Stop Prediction  —  Part D
=======================================================
Directly reuses the team's existing pipeline:
  ingest.run()     → data/interim/raw.parquet
  preprocess.run() → data/processed/train.parquet  (Year 2022-2024, feature-engineered)
                   → data/processed/test.parquet   (Year 2025,     feature-engineered)

The champion model is an sklearn Pipeline (preprocessor + XGBoost),
so model.predict(X) handles all preprocessing internally.

Four monitoring scenarios
─────────────────────────
  S1  Step 7  : test data, no changes              → baseline
  S2  Step 9a : TyreLife += 20                     → tyre-life drift
  S3  Step 9b : Compound → SOFT                    → compound drift
  S4  Step 9c : Cumulative_Degradation × 2         → degradation drift

S2 / S3 / S4 are all independent (each starts from the original test data).

Run
───
  pip install evidently==0.4.33
  python monitoring.py

Outputs → monitoring_reports/
  scenario_1_original.html
  scenario_2_tyrelife.html
  scenario_3_compound.html
  scenario_4_degradation.html
  metrics_summary.json
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import mlflow.sklearn
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

from evidently import ColumnMapping
from evidently.metric_preset import (
    ClassificationPreset,
    DataDriftPreset,
    DataQualityPreset,
)
from evidently.report import Report

# project root
ROOT = Path(__file__).resolve().parent
if not (ROOT / "src").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.WARNING)  

from src import config                        
from src import ingest, preprocess           

OUT_DIR = ROOT / "monitoring_reports"
OUT_DIR.mkdir(exist_ok=True)

# feature change parameters
TYRELIFE_DELTA     = 20      # change A
NEW_COMPOUND       = "SOFT"  # change B
DEGRADATION_FACTOR = 2.0     # change C

# columns Evidently needs
NUMERICAL_FEATURES = [
    "TyreLife", "LapNumber", "Stint", "Position",
    "LapTime (s)", "LapTime_Delta", "Cumulative_Degradation",
    "RaceProgress", "Position_Change", "PitStop",
]
CATEGORICAL_FEATURES = ["Compound"]

COLUMN_MAPPING = ColumnMapping(
    target="PitNextLap",
    prediction="prediction",
    numerical_features=NUMERICAL_FEATURES,
    categorical_features=CATEGORICAL_FEATURES,
)

# columns kept in monitoring df (raw feature columns + target)
MONITOR_COLS = NUMERICAL_FEATURES + CATEGORICAL_FEATURES + ["PitNextLap"]


# Step 1: ensure processed parquet files exist 

def ensure_processed_data() -> None:
    """Run ingest + preprocess if processed parquets don't exist yet."""
    if not config.PROCESSED_TRAIN.exists() or not config.PROCESSED_TEST.exists():
        print("Processed data not found — running team pipeline ...")
        if not config.RAW_PARQUET.exists():
            print("  ingest.run() ...")
            ingest.run()
        print("  preprocess.run() ...")
        preprocess.run()
        print("  Done.\n")
    else:
        print("Processed data found, skipping pipeline.\n")



# Helpers

def load_model():
    print(f"Loading champion model from {config.CHAMPION_EXPORT_DIR} ...")
    return mlflow.sklearn.load_model(str(config.CHAMPION_EXPORT_DIR))


def load_processed_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load team's processed parquets.
      train.parquet → Year 2022-2024, feature-engineered  → used as reference
      test.parquet  → Year 2025,      feature-engineered  → used as current
    """
    train_df = pd.read_parquet(config.PROCESSED_TRAIN)
    test_df  = pd.read_parquet(config.PROCESSED_TEST)
    print(f"  Reference (train): {len(train_df):,} rows  years={sorted(train_df.Year.unique())}")
    print(f"  Current   (test):  {len(test_df):,} rows   years={sorted(test_df.Year.unique())}")
    return train_df, test_df


def predict(df: pd.DataFrame, model) -> tuple:
    """Run model.predict on a processed dataframe. Model is a full Pipeline."""
    X       = df.drop(columns=[config.TARGET_COL], errors="ignore")
    y_pred  = model.predict(X).astype(int)
    y_score = model.predict_proba(X)[:, 1]
    return y_pred, y_score


def build_monitor_df(processed_df: pd.DataFrame,
                     y_pred, y_score) -> pd.DataFrame:
    """Keep only the columns Evidently needs + add prediction columns."""
    # use raw feature columns that exist in processed df
    keep = [c for c in MONITOR_COLS if c in processed_df.columns]
    out = processed_df[keep].copy()
    out["prediction"] = y_pred
    out["proba_pit"]  = y_score
    return out


def compute_metrics(monitor_df: pd.DataFrame) -> dict:
    y_true  = monitor_df["PitNextLap"].astype(int)
    y_pred  = monitor_df["prediction"].astype(int)
    y_score = monitor_df["proba_pit"]
    return {
        "rows":               int(len(monitor_df)),
        "f1_macro":           round(float(f1_score(y_true, y_pred, average="macro")), 4),
        "roc_auc":            round(float(roc_auc_score(y_true, y_score)), 4),
        "pit_rate_actual":    round(float(y_true.mean()), 4),
        "pit_rate_predicted": round(float(y_pred.mean()), 4),
    }


def build_report(reference: pd.DataFrame,
                 current: pd.DataFrame,
                 out_path: Path,
                 label: str) -> None:
    print(f"  Building Evidently report: {label} ...")
    report = Report(metrics=[
        DataQualityPreset(),
        DataDriftPreset(),
        ClassificationPreset(),
    ])
    report.run(
        reference_data=reference,
        current_data=current,
        column_mapping=COLUMN_MAPPING,
    )
    report.save_html(str(out_path))
    print(f"  Saved -> {out_path.name}")


# feature-change functions (operate on processed DataFrame)

def change_A(df: pd.DataFrame) -> pd.DataFrame:
    """TyreLife += TYRELIFE_DELTA"""
    out = df.copy()
    out["TyreLife"] = out["TyreLife"] + TYRELIFE_DELTA
    return out

def change_B(df: pd.DataFrame) -> pd.DataFrame:
    """Compound -> NEW_COMPOUND"""
    out = df.copy()
    out["Compound"] = NEW_COMPOUND
    return out

def change_C(df: pd.DataFrame) -> pd.DataFrame:
    """Cumulative_Degradation × DEGRADATION_FACTOR"""
    out = df.copy()
    out["Cumulative_Degradation"] = out["Cumulative_Degradation"] * DEGRADATION_FACTOR
    return out



# Main

def main() -> None:
    # setup 
    ensure_processed_data()
    model = load_model()

    print("\nLoading processed data ...")
    train_df, test_df = load_processed_data()

    # sample reference to keep Evidently fast
    ref_df = train_df.copy()
    if len(ref_df) > 50_000:
        ref_df = ref_df.sample(50_000, random_state=42).reset_index(drop=True)
        print(f"  Reference sampled to 50,000 rows")

    # build reference monitor df
    ref_pred, ref_score = predict(ref_df, model)
    ref_monitor = build_monitor_df(ref_df, ref_pred, ref_score)

    results = {}

    # S1: original test data
    print("\n── S1: original test data ──")
    s1_pred, s1_score = predict(test_df, model)
    s1 = build_monitor_df(test_df, s1_pred, s1_score)
    results["S1_original"] = compute_metrics(s1)
    m = results["S1_original"]
    print(f"   F1-macro={m['f1_macro']}   AUC={m['roc_auc']}   pit_pred={m['pit_rate_predicted']:.1%}")
    build_report(ref_monitor, s1, OUT_DIR / "scenario_1_original.html", "S1 original")

    # S2: change A — TyreLife +20
    print("\n── S2: TyreLife +20 ──")
    s2_df = change_A(test_df)
    s2_pred, s2_score = predict(s2_df, model)
    s2 = build_monitor_df(s2_df, s2_pred, s2_score)
    results["S2_tyrelife_plus20"] = compute_metrics(s2)
    m = results["S2_tyrelife_plus20"]
    print(f"   F1-macro={m['f1_macro']}   AUC={m['roc_auc']}   pit_pred={m['pit_rate_predicted']:.1%}")
    build_report(ref_monitor, s2, OUT_DIR / "scenario_2_tyrelife.html", "S2 TyreLife+20")

    # S3: change B — Compound->SOFT 
    print("\n── S3: Compound->SOFT ──")
    s3_df = change_B(test_df)
    s3_pred, s3_score = predict(s3_df, model)
    s3 = build_monitor_df(s3_df, s3_pred, s3_score)
    results["S3_compound"] = compute_metrics(s3)
    m = results["S3_compound"]
    print(f"   F1-macro={m['f1_macro']}   AUC={m['roc_auc']}   pit_pred={m['pit_rate_predicted']:.1%}")
    build_report(ref_monitor, s3, OUT_DIR / "scenario_3_compound.html", "S3 Compound->SOFT")

    # ── S4: change C — Degradation x2 
    print("\n── S4: Degradation x2 ──")
    s4_df = change_C(test_df)
    s4_pred, s4_score = predict(s4_df, model)
    s4 = build_monitor_df(s4_df, s4_pred, s4_score)
    results["S4_degradation"] = compute_metrics(s4)
    m = results["S4_degradation"]
    print(f"   F1-macro={m['f1_macro']}   AUC={m['roc_auc']}   pit_pred={m['pit_rate_predicted']:.1%}")
    build_report(ref_monitor, s4, OUT_DIR / "scenario_4_degradation.html", "S4 Degradation x2")

    # summary JSON 
    summary = {
        "data_source": {
            "reference": "data/processed/train.parquet (Year 2022-2024)",
            "current":   "data/processed/test.parquet  (Year 2025)",
            "note":      "processed by team pipeline: ingest -> preprocess",
        },
        "feature_changes": {
            "S2": f"TyreLife += {TYRELIFE_DELTA}",
            "S3": f"Compound -> '{NEW_COMPOUND}'",
            "S4": f"Cumulative_Degradation x {DEGRADATION_FACTOR}",
        },
        "champion_baseline": {
            "algorithm":     "xgboost",
            "test_macro_f1": 0.7852,
            "test_roc_auc":  0.8945,
        },
        **results,
    }
    p = OUT_DIR / "metrics_summary.json"
    p.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved metrics_summary.json -> {p}")

    # console table
    s1m = results["S1_original"]
    rows = [
        ("Champion (train eval)",    0.7852, 0.8945, "—"),
        ("S1  original test",        s1m["f1_macro"],                          s1m["roc_auc"],                          f"{s1m['pit_rate_predicted']:.1%}"),
        ("S2  TyreLife +20",         results["S2_tyrelife_plus20"]["f1_macro"], results["S2_tyrelife_plus20"]["roc_auc"], f"{results['S2_tyrelife_plus20']['pit_rate_predicted']:.1%}"),
        ("S3  Compound->SOFT",       results["S3_compound"]["f1_macro"],        results["S3_compound"]["roc_auc"],        f"{results['S3_compound']['pit_rate_predicted']:.1%}"),
        ("S4  Degradation x2",       results["S4_degradation"]["f1_macro"],     results["S4_degradation"]["roc_auc"],     f"{results['S4_degradation']['pit_rate_predicted']:.1%}"),
    ]
    print("\n" + "=" * 68)
    print(f"{'Scenario':<35} {'F1-macro':>9} {'AUC':>8} {'Pit%pred':>9}")
    print("-" * 68)
    for label, f1, auc, pit in rows:
        print(f"{label:<35} {str(f1):>9} {str(auc):>8} {pit:>9}")
    print("=" * 68)

    print(f"\nIndividual feature change impact vs S1 baseline (F1-macro):")
    for key, label in [
        ("S2_tyrelife_plus20", "S2 TyreLife+20  "),
        ("S3_compound",        "S3 Compound SOFT"),
        ("S4_degradation",     "S4 Degradation×2"),
    ]:
        delta = results[key]["f1_macro"] - s1m["f1_macro"]
        print(f"  {label}: {s1m['f1_macro']} -> {results[key]['f1_macro']}  (delta {delta:+.4f})")

    print("\nOpen monitoring_reports/*.html in a browser to see Evidently dashboards.")


if __name__ == "__main__":
    main()