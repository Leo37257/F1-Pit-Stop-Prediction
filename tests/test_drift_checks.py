"""Unit tests for monitoring/drift_checks.py. Synthetic data only: no model, no dataset needed."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from monitoring import drift_checks as dc

rng = np.random.default_rng(0)


def test_same_distribution_is_stable():
    a, b = pd.Series(rng.normal(0, 1, 20_000)), pd.Series(rng.normal(0, 1, 20_000))
    psi = dc.psi_numeric(a, b)
    assert psi < 0.02
    assert dc.classify_psi(psi) == "ok"


def test_one_sd_shift_is_an_alert():
    a, b = pd.Series(rng.normal(0, 1, 20_000)), pd.Series(rng.normal(1, 1, 20_000))
    assert dc.psi_numeric(a, b) >= dc.PSI_ALERT
    assert dc.ks_statistic(a, b) > 0.3


def test_small_shift_is_a_warning_not_an_alert():
    a, b = pd.Series(rng.normal(0, 1, 50_000)), pd.Series(rng.normal(0.35, 1, 50_000))
    assert dc.classify_psi(dc.psi_numeric(a, b)) == "warn"


def test_categorical_mix_change():
    ref = pd.Series(["SOFT"] * 300 + ["MEDIUM"] * 500 + ["HARD"] * 200)
    same = ref.sample(frac=1.0, random_state=1)
    all_soft = pd.Series(["SOFT"] * 1000)            # the S3 scenario
    assert dc.psi_categorical(ref, same) == pytest.approx(0.0, abs=1e-9)
    assert dc.classify_psi(dc.psi_categorical(ref, all_soft)) == "alert"


def test_unseen_category_and_missing_values_do_not_crash():
    ref = pd.Series(["SOFT", "HARD", None] * 100)
    cur = pd.Series(["INTERMEDIATE", "SOFT", None] * 100)
    assert np.isfinite(dc.psi_categorical(ref, cur))
    num_ref = pd.Series([1.0, 2.0, np.nan] * 100)
    assert np.isfinite(dc.psi_numeric(num_ref, num_ref))


def test_constant_reference():
    ref = pd.Series([0.0] * 1000)
    assert dc.psi_numeric(ref, ref) == 0.0
    assert dc.classify_psi(dc.psi_numeric(ref, pd.Series([0.0] * 900 + [1.0] * 100))) == "alert"


def test_binary_column_is_compared_by_shares_not_flagged():
    a = pd.Series(rng.binomial(1, 0.25, 20_000).astype(float))
    b = pd.Series(rng.binomial(1, 0.25, 20_000).astype(float))
    assert dc.classify_psi(dc.psi_numeric(a, b)) == "ok"          # used to come out as a false 0.25 alert
    c = pd.Series(rng.binomial(1, 0.60, 20_000).astype(float))
    assert dc.classify_psi(dc.psi_numeric(a, c)) == "alert"


def test_empty_input_is_unknown():
    assert dc.classify_psi(dc.psi_numeric(pd.Series([], dtype=float), pd.Series([1.0]))) == "unknown"


def test_feature_table_sorted_worst_first():
    ref = pd.DataFrame({"x": rng.normal(0, 1, 5000), "y": rng.normal(0, 1, 5000), "c": ["a", "b"] * 2500})
    cur = ref.assign(y=ref["y"] + 2)
    table = dc.feature_drift_table(ref, cur, numeric=["x", "y"], categorical=["c"])
    assert list(table.columns) == ["feature", "kind", "psi", "ks", "status"]
    assert table.iloc[0]["feature"] == "y" and table.iloc[0]["status"] == "alert"
    assert set(table["status"]) <= {"ok", "warn", "alert"}


@pytest.mark.parametrize("ref,cur,expected", [(0.785, 0.782, "ok"), (0.785, 0.751, "warn"), (0.785, 0.666, "alert")])
def test_performance_status_matches_the_four_scenarios(ref, cur, expected):
    assert dc.performance_status(ref, cur)["status"] == expected


def test_worst_status():
    assert dc.worst_status(["ok", "warn", "ok"]) == "warn"
    assert dc.worst_status(["ok", "alert", "warn"]) == "alert"
    assert dc.worst_status([]) == "ok"
