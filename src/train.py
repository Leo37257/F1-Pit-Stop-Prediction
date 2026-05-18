"""Train task: FLAML AutoML per algorithm, MLflow logging for every candidate.

We run one FLAML AutoML invocation per algorithm so each produces a clean MLflow
run with that algorithm's best hyperparameters, test metrics, and a saved
sklearn Pipeline (preprocessor + estimator). Champion selection happens in
src.register based on test_macro_f1.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import mlflow
import mlflow.sklearn
import pandas as pd
from flaml import AutoML
from mlflow.models.signature import infer_signature
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.pipeline import Pipeline

from src import config
from src.preprocess import build_preprocessor

logger = logging.getLogger(__name__)


def run(time_budget_total: int | None = None) -> dict[str, Any]:
    """Train one FLAML AutoML per algorithm, log each to MLflow.

    Returns a dict keyed by algorithm with test metrics for quick inspection.
    """
    config.ensure_dirs()
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.EXPERIMENT_NAME)

    total = time_budget_total if time_budget_total is not None else config.AUTOML_TIME_BUDGET
    per_algo = max(15, total // len(config.AUTOML_ESTIMATORS))
    logger.info("Total budget=%ds, per-algorithm budget=%ds", total, per_algo)

    train_df = pd.read_parquet(config.PROCESSED_TRAIN)
    test_df = pd.read_parquet(config.PROCESSED_TEST)
    X_train = train_df.drop(columns=[config.TARGET_COL])
    y_train = train_df[config.TARGET_COL].astype(int)
    X_test = test_df.drop(columns=[config.TARGET_COL])
    y_test = test_df[config.TARGET_COL].astype(int)
    logger.info("Train shape %s, Test shape %s", X_train.shape, X_test.shape)

    results: dict[str, dict[str, float]] = {}

    for algo in config.AUTOML_ESTIMATORS:
        with mlflow.start_run(run_name=f"flaml_{algo}") as run:
            t0 = time.perf_counter()
            pipe = Pipeline([
                ("pre", build_preprocessor()),
                ("clf", AutoML()),
            ])
            pipe.fit(
                X_train, y_train,
                clf__time_budget=per_algo,
                clf__metric=config.AUTOML_METRIC,
                clf__task="classification",
                clf__seed=config.RANDOM_SEED,
                clf__estimator_list=[algo],
                clf__verbose=0,
                clf__eval_method="holdout",
            )
            fit_seconds = time.perf_counter() - t0

            y_pred = pipe.predict(X_test)
            y_proba = pipe.predict_proba(X_test)[:, 1]
            test_macro_f1 = float(f1_score(y_test, y_pred, average="macro"))
            test_roc_auc = float(roc_auc_score(y_test, y_proba))

            clf = pipe.named_steps["clf"]
            mlflow.set_tag("algorithm", algo)
            mlflow.log_param("algorithm", algo)
            mlflow.log_param("time_budget_seconds", per_algo)
            mlflow.log_param("random_seed", config.RANDOM_SEED)
            for k, v in (clf.best_config or {}).items():
                mlflow.log_param(f"best_{k}", v)
            mlflow.log_metrics({
                "test_macro_f1": test_macro_f1,
                "test_roc_auc": test_roc_auc,
                "val_loss": float(clf.best_loss) if clf.best_loss is not None else float("nan"),
                "fit_seconds": fit_seconds,
                "flaml_best_iteration": int(clf.best_iteration or 0),
            })

            signature = infer_signature(X_test.head(5), y_pred[:5])
            mlflow.sklearn.log_model(
                pipe,
                artifact_path="model",
                signature=signature,
                input_example=X_test.head(5),
            )

            results[algo] = {
                "run_id": run.info.run_id,
                "test_macro_f1": test_macro_f1,
                "test_roc_auc": test_roc_auc,
                "fit_seconds": fit_seconds,
            }
            logger.info(
                "[%s] test_macro_f1=%.4f test_roc_auc=%.4f fit=%.1fs run_id=%s",
                algo, test_macro_f1, test_roc_auc, fit_seconds, run.info.run_id,
            )

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    budget_env = os.environ.get("TIME_BUDGET")
    budget = int(budget_env) if budget_env else None
    results = run(time_budget_total=budget)
    print("\n=== Per-algorithm results ===")
    for algo, r in sorted(results.items(), key=lambda kv: -kv[1]["test_macro_f1"]):
        print(f"  {algo:<16} test_macro_f1={r['test_macro_f1']:.4f}  "
              f"test_roc_auc={r['test_roc_auc']:.4f}  fit={r['fit_seconds']:.1f}s  "
              f"run_id={r['run_id']}")
