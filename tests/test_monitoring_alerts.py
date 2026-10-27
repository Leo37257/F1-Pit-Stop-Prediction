"""scenario_alerts combines feature drift and F1 cost; check it on synthetic frames (no model, no data)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from monitoring.model_monitoring import scenario_alerts

rng = np.random.default_rng(2)
ref = pd.DataFrame({"TyreLife": rng.integers(1, 40, 5000).astype(float),
                    "Compound": rng.choice(["SOFT", "MEDIUM", "HARD"], 5000)})


def test_drift_without_f1_cost_is_still_an_alert():
    results = {"S1_original": {"f1_macro": 0.785}, "S4": {"f1_macro": 0.784}}
    a = scenario_alerts(ref, {"S4": (ref.assign(TyreLife=ref.TyreLife * 2), "TyreLife")}, results)["S4"]
    assert a["drift_status"] == "alert" and a["performance_status"] == "ok" and a["status"] == "alert"


def test_categorical_feature_uses_category_psi():
    results = {"S1_original": {"f1_macro": 0.785}, "S3": {"f1_macro": 0.751}}
    a = scenario_alerts(ref, {"S3": (ref.assign(Compound="SOFT"), "Compound")}, results)["S3"]
    assert a["feature"] == "Compound" and a["drift_status"] == "alert" and a["performance_status"] == "warn"


def test_no_change_is_ok():
    results = {"S1_original": {"f1_macro": 0.785}, "S0": {"f1_macro": 0.785}}
    assert scenario_alerts(ref, {"S0": (ref.copy(), "TyreLife")}, results)["S0"]["status"] == "ok"
