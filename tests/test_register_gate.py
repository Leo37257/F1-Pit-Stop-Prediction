"""Retraining must not silently replace a better champion or drop its decision threshold."""
from __future__ import annotations

import json
import shutil

from evaluation.threshold_tuning import write_champion_threshold
from src import config
from src.register import should_promote


def test_first_model_is_always_promoted():
    assert should_promote(0.70, "run-a", None, None)[0]


def test_challenger_must_beat_the_champion():
    assert should_promote(0.7900, "run-b", 0.7852, "run-a")[0]
    assert not should_promote(0.7852, "run-b", 0.7852, "run-a")[0]     # a tie keeps the champion
    assert not should_promote(0.7700, "run-b", 0.7852, "run-a")[0]


def test_the_current_champion_is_not_promoted_again():
    promote, reason = should_promote(0.7852, "run-a", 0.7852, "run-a")
    assert not promote and "already" in reason


def test_threshold_is_written_back_to_the_champion_card(tmp_path):
    card_path = tmp_path / "CHAMPION.json"
    shutil.copy(config.CHAMPION_EXPORT_DIR / "CHAMPION.json", card_path)
    card = json.loads(card_path.read_text())
    del card["decision_threshold"], card["metrics"]["test_macro_f1_at_decision_threshold"]   # as register leaves it
    card_path.write_text(json.dumps(card))

    summary = {"chosen_threshold": 0.4, "test_2025": {"tuned": {"macro_f1": 0.79}}}
    write_champion_threshold(card_path, summary)

    updated = json.loads(card_path.read_text())
    assert updated["decision_threshold"] == 0.4
    assert updated["metrics"]["test_macro_f1_at_decision_threshold"] == 0.79
    assert updated["run_id"] == card["run_id"]                     # the rest of the card is untouched
