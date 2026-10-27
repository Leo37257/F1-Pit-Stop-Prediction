"""Pure-function tests for monitoring/natural_drift.py (synthetic frames, no model)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from monitoring import natural_drift as nd

rng = np.random.default_rng(1)


def _frame(n=3000, year=2024):
    return pd.DataFrame({
        "id": np.arange(n), "Year": year, "Race": rng.choice(["Monaco", "Monza"], n),
        "Compound": rng.choice(["SOFT", "MEDIUM", "HARD"], n), "TyreLife_bucket": rng.choice(["0-5", "6-15"], n),
        "TyreLife": rng.integers(1, 40, n).astype(float), "LapTime (s)": rng.normal(90, 3, n),
        "PitNextLap": rng.integers(0, 2, n),
    })


def test_feature_lists_exclude_ids_year_and_target():
    numeric, categorical = nd.feature_lists(_frame())
    assert "id" not in numeric and "Year" not in numeric and "PitNextLap" not in numeric
    assert set(categorical) == {"Compound", "TyreLife_bucket"}
    assert {"TyreLife", "LapTime (s)"} <= set(numeric)


def test_year_over_year_reports_each_transition():
    df = pd.concat([_frame(year=2023), _frame(year=2024).assign(TyreLife=lambda d: d.TyreLife + 15)])
    numeric, categorical = nd.feature_lists(df)
    yoy = nd.year_over_year(df, numeric, categorical)
    assert [(r["from"], r["to"]) for r in yoy] == [(2023, 2024)]
    assert yoy[0]["features_alert"] >= 1          # the TyreLife shift


def test_slice_performance_flags_a_bad_slice():
    test = _frame(n=4000).reset_index(drop=True)
    y = test["PitNextLap"].to_numpy()
    y_pred = y.copy()
    bad = (test["Race"] == "Monaco").to_numpy()
    y_pred[bad] = 1 - y_pred[bad]                 # model is always wrong on Monaco
    y_score = y_pred.astype(float)
    table = nd.slice_performance(test, y_pred, y_score, overall_f1=0.9)
    monaco = table[(table.slice == "Race") & (table.value == "Monaco")].iloc[0]
    assert monaco["status"] == "alert" and monaco["f1_macro"] < 0.1
    assert set(table.slice) == {"Compound", "Race"}
