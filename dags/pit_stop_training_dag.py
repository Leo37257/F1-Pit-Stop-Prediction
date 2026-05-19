"""F1 pit-stop training pipeline.

Four sequential BashOperators that invoke `python -m src.<task>` inside the
Airflow container. We use Bash rather than PythonOperator because the train
task pulls in catboost/xgboost/lightgbm + MLflow, and Airflow's standard
task runner forks the task from a multi-threaded parent process which
crashes those native libraries silently. A fresh Python subprocess per task
sidesteps that completely and also matches the standalone `python -m src.X`
commands used during development.

Schedule is None (manual trigger). DVC versioning and git tagging happen
outside the DAG.

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
    description="F1 pit-stop binary classifier: ingest -> preprocess -> train -> register",
    schedule=None,
    start_date=datetime(2026, 5, 1),
    catchup=False,
    default_args={"owner": "member-b", "retries": 0},
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
    tags=["mlops", "f1", "member-b"],
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
    )

    ingest >> preprocess >> train >> register
