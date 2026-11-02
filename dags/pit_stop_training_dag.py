"""F1 pit-stop training pipeline.

ingest -> preprocess -> train -> register -> tune_threshold -> [monitor_drift, stress_test]

Every task is a BashOperator that runs a fresh `python` process inside the
Airflow container. We use Bash rather than PythonOperator because the train
task pulls in catboost/xgboost/lightgbm + MLflow, and Airflow's standard
task runner forks the task from a multi-threaded parent process which
crashes those native libraries silently. A fresh Python subprocess per task
sidesteps that completely and also matches the standalone `python -m src.X`
commands used during development.

Schedule is None (manual trigger). DVC versioning and git tagging happen
outside the DAG.

register is a champion/challenger gate: the new best run replaces the champion only if it scores
higher on the test season. Otherwise register exits 99, Airflow marks it skipped, and every task after
it is skipped too, so the served model and its threshold are left alone.

When the challenger wins:
  tune_threshold  picks the decision threshold out-of-fold and writes it to CHAMPION.json
  monitor_drift   natural drift + slice performance; fails the run if macro-F1 on the current data drops past
                  the alert line versus the macro-F1 recorded for the champion (feature drift alone only warns).
                  On this static dataset that is the same season, so it guards against a broken export or
                  threshold rather than real decay; it becomes a decay check once a new labelled season is scored.
  stress_test     the four injected-drift scenarios with Evidently reports; informational, never fails

Triggering with `{"time_budget": 80}` in the run config gives a ~2 min smoke
run; the default 600s gives the production training (~10 min).
"""

from __future__ import annotations

from datetime import datetime

from airflow import DAG
from airflow.models.param import Param
from airflow.operators.bash import BashOperator


with DAG(
    dag_id="pit_stop_training",
    description="F1 pit-stop classifier: ingest -> preprocess -> train -> register (gate) -> tune_threshold -> monitoring",
    schedule=None,
    start_date=datetime(2026, 5, 1),
    catchup=False,
    default_args={"owner": "f1-pit-stop", "retries": 0},
    params={
        "time_budget": Param(
            600,
            type="integer",
            minimum=60,
            maximum=3600,
            title="FLAML AutoML total budget (seconds)",
            description="Total time across all 4 algorithms. 80 = smoke run (~2min), 600 = production (~10min).",
        ),
    },
    tags=["mlops", "f1"],
) as dag:

    ingest = BashOperator(
        task_id="ingest",
        bash_command="cd /opt/airflow && python -m src.ingest",
    )

    preprocess = BashOperator(
        task_id="preprocess",
        bash_command="cd /opt/airflow && python -m src.preprocess",
    )

    train = BashOperator(
        task_id="train",
        bash_command=(
            "cd /opt/airflow && "
            "TIME_BUDGET={{ params.time_budget }} python -m src.train"
        ),
    )

    register = BashOperator(
        task_id="register",
        bash_command="cd /opt/airflow && python -m src.register",
        skip_on_exit_code=99,      # challenger lost: skip the rest, keep the champion
    )

    tune_threshold = BashOperator(
        task_id="tune_threshold",
        bash_command="cd /opt/airflow && python evaluation/threshold_tuning.py --write-champion",
    )

    monitor_drift = BashOperator(
        task_id="monitor_drift",
        bash_command="cd /opt/airflow && python monitoring/natural_drift.py --fail-on alert",
    )

    stress_test = BashOperator(
        task_id="stress_test",
        bash_command="cd /opt/airflow && python monitoring/model_monitoring.py",
    )

    ingest >> preprocess >> train >> register >> tune_threshold >> [monitor_drift, stress_test]
