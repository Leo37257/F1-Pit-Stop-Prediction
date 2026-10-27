"""The served model applies the threshold stored with the champion; the tuning helper picks the F1-best cut-off."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from evaluation.threshold_tuning import best_threshold, reliability
from src import config
from src.inference import DEFAULT_THRESHOLD, apply_threshold, decision_threshold


class _ProbaModel:
    def __init__(self, p):
        self.p = np.asarray(p, float)

    def predict_proba(self, X):
        return np.c_[1 - self.p, self.p]


class _LabelModel:
    def predict(self, X):
        return np.array([1, 0, 1])


def test_champion_threshold_is_read_from_champion_json():
    stored = json.loads((config.CHAMPION_EXPORT_DIR / "CHAMPION.json").read_text())["decision_threshold"]
    assert decision_threshold() == stored == 0.36


def test_missing_metadata_falls_back_to_half(tmp_path):
    assert decision_threshold(tmp_path) == DEFAULT_THRESHOLD == 0.5


def test_apply_threshold_uses_probabilities():
    X = pd.DataFrame({"a": [0, 0, 0, 0]})
    model = _ProbaModel([0.30, 0.36, 0.40, 0.70])
    assert apply_threshold(model, X, threshold=0.36).tolist() == [0, 1, 1, 1]
    assert apply_threshold(model, X, threshold=0.5).tolist() == [0, 0, 0, 1]


def test_label_only_model_is_left_alone():
    assert apply_threshold(_LabelModel(), pd.DataFrame({"a": [0, 0, 0]})).tolist() == [1, 0, 1]


def test_best_threshold_finds_the_separating_cut():
    y = np.array([0] * 50 + [1] * 50)
    p = np.r_[np.full(50, 0.2), np.full(50, 0.3)]          # positives only slightly above negatives
    t, f1 = best_threshold(y, p)
    assert 0.2 < t <= 0.3 and f1 == 1.0


def test_reliability_bins_cover_all_rows():
    rng = np.random.default_rng(3)
    p = rng.uniform(0, 1, 1000)
    y = (rng.uniform(0, 1, 1000) < p).astype(int)
    table = reliability(y, p)
    assert table["rows"].sum() == 1000 and len(table) == 10
