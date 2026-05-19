"""Register task: pick champion by macro F1, register in MLflow Registry, export standalone.

Run after src.train. Filters to the 4 parent FLAML runs (excludes the ~119 trial
children that FLAML logs automatically), picks the one with highest
test_macro_f1, registers it under f1-pit-stop-classifier, promotes that version
to Production, and exports a standalone MLflow model directory at
models/champion/ for Member C's FastAPI service plus a CHAMPION.json metadata
file for Member D's drift baseline.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import mlflow.sklearn

from src import config

logger = logging.getLogger(__name__)


def run() -> dict:
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
        "Champion picked: algorithm=%s test_macro_f1=%.4f test_roc_auc=%.4f run_id=%s",
        algo, macro_f1, roc_auc, champion.info.run_id,
    )

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
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    card = run()
    print("\n=== Champion card ===")
    print(json.dumps(card, indent=2))
    print(f"\nStandalone export: {config.CHAMPION_EXPORT_DIR}")
    print("\nMember C — load via the registry (needs sqlite DB on path):")
    print(f"  mlflow.pyfunc.load_model('models:/{config.REGISTERED_MODEL_NAME}/Production')")
    print("Member C — load standalone (no MLflow tracking server needed):")
    print(f"  mlflow.pyfunc.load_model('{config.CHAMPION_EXPORT_DIR.relative_to(config.PROJECT_ROOT)}')")
