"""Register task: pick the best new run, replace the champion only if it beats it, export standalone.

Run after src.train. Filters to the 4 parent FLAML runs (excludes the ~119 trial
children that FLAML logs automatically) and takes the one with the highest
test_macro_f1 as the challenger. The challenger replaces the current champion only
if it scores higher on the same test season, both at the default 0.5 cut-off (the
challenger has no tuned threshold yet). If it does, it is registered under
f1-pit-stop-classifier, promoted to Production, and exported to models/champion/
(the directory the FastAPI service and the monitoring scripts load) with a
CHAMPION.json metadata file. The decision threshold is chosen afterwards by
evaluation/threshold_tuning.py --write-champion.

If the challenger loses, nothing is changed and the script exits with code 99,
which Airflow's BashOperator treats as "skipped", so the downstream tasks skip too.
`--force` promotes regardless.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.metrics import f1_score

from src import config

logger = logging.getLogger(__name__)

CHALLENGER_LOST_EXIT = 99   # BashOperator's default skip_on_exit_code


def should_promote(challenger_f1: float, challenger_run_id: str,
                   champion_f1: float | None, champion_run_id: str | None) -> tuple[bool, str]:
    """Champion/challenger rule: the challenger must score strictly higher on the same test season."""
    if champion_f1 is None:
        return True, "no current champion"
    if challenger_run_id == champion_run_id:
        return False, "the best run is already the champion"
    if challenger_f1 > champion_f1:
        return True, f"challenger macro-F1 {challenger_f1:.4f} beats champion {champion_f1:.4f}"
    return False, f"challenger macro-F1 {challenger_f1:.4f} does not beat champion {champion_f1:.4f}"


def current_champion() -> tuple[float | None, str | None]:
    """Re-score the exported champion on the current test season at 0.5, so both sides see the same data."""
    card_path = config.CHAMPION_EXPORT_DIR / "CHAMPION.json"
    if not card_path.exists():
        return None, None
    model = mlflow.sklearn.load_model(str(config.CHAMPION_EXPORT_DIR))
    test = pd.read_parquet(config.PROCESSED_TEST)
    y_pred = model.predict(test.drop(columns=[config.TARGET_COL]))
    f1 = float(f1_score(test[config.TARGET_COL].astype(int), y_pred, average="macro"))
    return f1, json.loads(card_path.read_text()).get("run_id")


def run(force: bool = False) -> dict | None:
    config.ensure_dirs()
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    client = mlflow.MlflowClient()

    exp = client.get_experiment_by_name(config.EXPERIMENT_NAME)
    if exp is None:
        raise RuntimeError(
            f"Experiment '{config.EXPERIMENT_NAME}' not found — run `python -m src.train` first."
        )

    all_runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["metrics.test_macro_f1 DESC"],
        max_results=500,
    )
    parent_runs = [r for r in all_runs if (r.info.run_name or "").startswith("flaml_")]
    if not parent_runs:
        raise RuntimeError("No flaml_* parent runs found — run `python -m src.train` first.")

    champion = parent_runs[0]
    algo = champion.data.params["algorithm"]
    macro_f1 = float(champion.data.metrics["test_macro_f1"])
    roc_auc = float(champion.data.metrics["test_roc_auc"])
    logger.info(
        "Challenger: algorithm=%s test_macro_f1=%.4f test_roc_auc=%.4f run_id=%s",
        algo, macro_f1, roc_auc, champion.info.run_id,
    )

    incumbent_f1, incumbent_run_id = current_champion()
    promote, reason = should_promote(macro_f1, champion.info.run_id, incumbent_f1, incumbent_run_id)
    logger.info("Champion/challenger: %s", reason)
    if not (promote or force):
        return None

    model_uri = f"runs:/{champion.info.run_id}/model"
    mv = mlflow.register_model(model_uri=model_uri, name=config.REGISTERED_MODEL_NAME)
    logger.info("Registered %s version %s", config.REGISTERED_MODEL_NAME, mv.version)

    client.transition_model_version_stage(
        name=config.REGISTERED_MODEL_NAME,
        version=mv.version,
        stage="Production",
        archive_existing_versions=True,
    )
    logger.info("Promoted version %s to Production (older versions archived)", mv.version)

    if config.CHAMPION_EXPORT_DIR.exists():
        shutil.rmtree(config.CHAMPION_EXPORT_DIR)
    loaded = mlflow.sklearn.load_model(model_uri)
    mlflow.sklearn.save_model(loaded, str(config.CHAMPION_EXPORT_DIR))
    logger.info("Exported standalone model -> %s", config.CHAMPION_EXPORT_DIR)

    card = {
        "registered_model_name": config.REGISTERED_MODEL_NAME,
        "registered_version": int(mv.version),
        "run_id": champion.info.run_id,
        "algorithm": algo,
        "metrics": {
            "test_macro_f1": macro_f1,
            "test_roc_auc": roc_auc,
            "val_loss": float(champion.data.metrics.get("val_loss", float("nan"))),
        },
        "baseline_to_beat": {"macro_f1": 0.6122, "roc_auc": 0.7394},
        "training_data": {
            "train_parquet_sha256": _sha256(config.PROCESSED_TRAIN),
            "test_parquet_sha256": _sha256(config.PROCESSED_TEST),
        },
        "replaced": {"run_id": incumbent_run_id,
                     "test_macro_f1_rescored": None if incumbent_f1 is None else round(incumbent_f1, 4)},
        "trained_at_utc": _iso_now(),
        "git_sha": _git_sha(),
        "best_hyperparams": {
            k.removeprefix("best_"): v
            for k, v in champion.data.params.items()
            if k.startswith("best_")
        },
    }
    card_path = config.CHAMPION_EXPORT_DIR / "CHAMPION.json"
    card_path.write_text(json.dumps(card, indent=2))
    logger.info("Wrote champion card -> %s", card_path)

    return card


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=config.PROJECT_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(description="Promote the best new run if it beats the current champion")
    parser.add_argument("--force", action="store_true", help="promote the best new run even if it does not win")
    card = run(force=parser.parse_args().force)
    if card is None:
        print("\nChampion kept: the challenger did not beat it (see the log line above).")
        sys.exit(CHALLENGER_LOST_EXIT)
    print("\n=== Champion card ===")
    print(json.dumps(card, indent=2))
    print(f"\nStandalone export: {config.CHAMPION_EXPORT_DIR}")
    print("Next: python evaluation/threshold_tuning.py --write-champion  (picks the decision threshold)")
    print("Load via the registry (needs the sqlite DB):")
    print(f"  mlflow.sklearn.load_model('models:/{config.REGISTERED_MODEL_NAME}/Production')")
    print("Load standalone (no MLflow tracking server needed):")
    print(f"  mlflow.sklearn.load_model('{config.CHAMPION_EXPORT_DIR.relative_to(config.PROJECT_ROOT)}')")
