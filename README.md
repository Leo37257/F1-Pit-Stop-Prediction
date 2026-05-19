# F1 Pit-Stop Prediction — Training Pipeline

Binary classifier predicting whether an F1 driver will pit on the next lap. Built on the Kaggle Playground Series S5E6 dataset (2022–2025).

This repo owns the **training pipeline** (Member B's deliverable): ingest → preprocess → train → register. The trained champion lives at [`models/champion/`](models/champion/) and is the contract for Member C's FastAPI `/predict` service and Member D's drift monitoring.

## Project layout

```
F1-Pit-Stop-Prediction/
├── data/                          # Raw Kaggle CSVs (train.csv, test.csv) + DVC-tracked outputs
├── notebooks/                     # Member A's EDA + feature engineering source of truth
├── src/                           # Importable pipeline modules
│   ├── config.py                  # Paths, MLflow URI, RANDOM_SEED, AutoML config
│   ├── ingest.py                  # CSV → parquet (validates the 16-column schema)
│   ├── feature_engineering.py     # add_features() — verbatim port of Member A's EDA cell
│   ├── preprocess.py              # FE + year split + build_preprocessor()
│   ├── train.py                   # FLAML AutoML per algorithm, MLflow per-candidate logs
│   └── register.py                # Champion selection, MLflow Registry, standalone export
├── dags/pit_stop_training_dag.py  # Airflow DAG — 4 BashOperators wrapping src.*
├── tests/                         # Import + config-path smoke tests
├── models/champion/               # ⭐ Handoff payload for Member C and D
├── docker-compose.yaml            # Airflow LocalExecutor + Postgres
├── Dockerfile                     # apache/airflow:2.9.3-python3.12 + ML deps
├── dvc.yaml / dvc.lock            # DVC pipeline (ingest, preprocess stages)
└── requirements.txt               # Pinned versions
```

## Pipeline at a glance

```
data/train.csv  ─[ingest]─▶  data/interim/raw.parquet
                                       │
                            ─[preprocess]─▶  data/processed/{train,test}.parquet  +  ColumnTransformer factory
                                       │
                              ─[train]─▶  4 MLflow runs (lgbm, xgboost, xgb_limitdepth, catboost)
                                       │       each = Pipeline(preprocessor → AutoML).fit(...)
                                       │
                           ─[register]─▶  Pick best test_macro_f1 → register as f1-pit-stop-classifier
                                                                  → promote to Production
                                                                  → export models/champion/
                                                                  → write CHAMPION.json baseline
```

## Quick start (clone → predict)

```bash
git clone https://github.com/EdwardHuang777/F1-Pit-Stop-Prediction.git
cd F1-Pit-Stop-Prediction

python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest tests/                          # 2/2 should pass

# The trained champion is committed to git — load and predict on raw rows:
python -c "
import mlflow.pyfunc, pandas as pd
m = mlflow.pyfunc.load_model('models/champion')
sample = pd.read_parquet('data/processed/test.parquet').drop(columns=['PitNextLap']).head(5)
print(m.predict(sample))
"
```

If `data/processed/test.parquet` isn't on disk yet, run `python -m src.ingest && python -m src.preprocess` (or `dvc repro`) first.

## For Member C — FastAPI `/predict`

The champion is a self-contained MLflow model directory. **No MLflow tracking server, no sqlite DB required** — just load the directory.

```python
import mlflow.pyfunc
model = mlflow.pyfunc.load_model("models/champion")
predictions = model.predict(features_df)   # numpy array of int (0 / 1)
```

**Input shape:** a pandas DataFrame with the 35 feature columns from the processed schema (anything except `PitNextLap`). Preprocessing is bundled in the saved `Pipeline`:
- Numeric features pass through (tree models handle NaN natively)
- `Compound` and `TyreLife_bucket` get one-hot encoded
- High-cardinality / temporal-leak columns (`id`, `Driver`, `Race`, `Year`) are dropped automatically

The full column list with dtypes is in [src/preprocess.py:18-43](src/preprocess.py#L18-L43). The MLflow signature in `models/champion/MLmodel` is the authoritative schema — generate a request example with `mlflow models predict --model-uri models/champion --content-type csv --input-path …` or just read `model.metadata.get_input_schema()` in Python.

**Refresh cadence:** when Member B retrains, the new champion is committed to `models/champion/`. A `git pull` + model reload gives you the new version. The directory layout never changes; only the algorithm and weights do.

**If you want raw probabilities** instead of class labels: load with `mlflow.sklearn.load_model("models/champion")` (sklearn flavor) and call `.predict_proba()`.

## For Member D — drift monitoring

Two artifacts in `models/champion/`:

1. **`CHAMPION.json`** — drift baseline:
   - `metrics.test_macro_f1`, `metrics.test_roc_auc` — performance baseline to alert against
   - `training_data.train_parquet_sha256` / `test_parquet_sha256` — data-identity hashes; comparing to incoming hashes tells you if the model was retrained on different data
   - `algorithm`, `best_hyperparams` — what's currently in production
   - `trained_at_utc`, `git_sha`, `registered_version` — provenance

2. **`data/processed/test.parquet`** — the 2025 reference distribution for feature-drift detection (Evidently AI's reference dataset, etc.)

**Pre-empt the 2023 anomaly** in your dashboard. Year 2023 has ~0.96% pit-next-lap rate vs ~28% in 2022/2024/2025 — a Kaggle Playground artifact, not real drift. Member A flagged it in [notebooks/EDA.ipynb](notebooks/EDA.ipynb) §7. If you slice by year, label it as such.

## Running the pipeline

### A) Standalone Python (fastest for development, ~12 min)

```bash
source .venv/bin/activate
python -m src.ingest        # ~2s
python -m src.preprocess    # ~30s
python -m src.train         # ~12 min (4 × FLAML @ 150s/algo + overhead)
python -m src.register      # ~5s
```

Override the AutoML budget with `TIME_BUDGET=80 python -m src.train` (80s total = ~2 min smoke run).

### B) Airflow DAG via Docker (production orchestration, ~20–25 min)

```bash
mkdir -p mlflow              # Pre-create bind-mount sources (one-time after a fresh clone)
docker compose build         # ~10 min one-time; cached after
docker compose up -d
# Open http://localhost:8080  → admin / admin → trigger pit_stop_training
# Default budget {"time_budget": 600}. Use {"time_budget": 80} for a smoke run.
docker compose down          # when done
```

The DAG runs each `src.X` task as a `BashOperator` subprocess (not PythonOperator — that hits a fork-after-thread bug with native ML libs). Bind-mounted volumes mean updated `models/champion/` appears on host immediately after register completes.

### C) DVC reproduce (data only)

```bash
dvc repro
```

Replays `ingest` then `preprocess` if any of their deps changed. Doesn't run train/register — those are Airflow's job.

## MLflow

- **Backend:** SQLite at `mlflow/mlflow.db`, artifacts at `mlflow/artifacts/`
- **Tracking URI:** `sqlite:///mlflow/mlflow.db` (set in [src/config.py:21](src/config.py#L21))
- **UI:** `mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db --port 5001`

**Host ↔ Docker caveat.** MLflow stores absolute artifact paths in the sqlite DB at experiment-creation time, so the DB created from a host run won't work for a Docker run (and vice versa). When switching environments: `rm -rf mlflow/ mlruns/` first. [src/train.py:31-52](src/train.py#L31-L52) detects stale `artifact_location` and fails fast with a clear hint. `models/champion/` is the durable handoff and is unaffected.

## Champion history

| Version | Algorithm | Test F1-macro | Test ROC-AUC | Notes |
|---|---|---:|---:|---|
| 1 | catboost | 0.7834 | 0.8944 | Host-trained, 600s budget. v1 register. |
| 2 | xgboost | **0.7852** | 0.8945 | First DAG-trained champion. |

**Baseline to beat (Member A):** F1-macro ≥ 0.6122, ROC-AUC ≥ 0.7394 (TyreLife≥25 heuristic). Both champion versions clear it by ~+0.17 / +0.15.

## Reproducibility

- **Python**: 3.12 on host and inside Docker
- **Random seed**: 42, set in [src/config.py](src/config.py) and threaded into FLAML via `clf__seed`
- **Pinned deps**: [requirements.txt](requirements.txt)
- **Data identity**: SHA256 of `data/processed/*.parquet` stored in `CHAMPION.json`
- **Code identity**: git SHA stored in `CHAMPION.json`

## Gotchas

- **First Docker build is ~10 min** (catboost + xgboost + lightgbm + arm64 wheels). Cached after that.
- **catboost is slow inside arm64 Docker.** A 600s-budget DAG run can take 20–25 min real-time mostly waiting on catboost. The host venv is ~2× faster (no bind-mount I/O penalty).
- **Don't `rm -rf mlflow/` while Docker is up.** Bring `docker compose down` first; the bind mount can go into a weird state otherwise.
- **Switching host ↔ Docker for training:** wipe `mlflow/` and `mlruns/` between environments. (`models/champion/` survives.)
- **GitHub flags `data/train.csv` as >50 MB.** It's a soft warning, not a block — committed for grader convenience.

## Pipeline contract for Member B (retrain workflow)

1. Pull latest data if changed
2. `python -m src.train` (host) or trigger Airflow DAG (Docker)
3. `python -m src.register` runs automatically as the last DAG task; otherwise run it manually
4. Inspect `models/champion/CHAMPION.json` — verify `test_macro_f1` beats the prior champion
5. `git add models/champion/ && git commit -m "Champion v{N}: {algo}, F1={X}"`
6. `git push` — Member C reloads, Member D updates the drift baseline
