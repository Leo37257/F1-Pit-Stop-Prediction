![F1 Pit-Stop Predictor — banner](docs/banner.jpg)

# F1 Pit-Stop Prediction

<p align="center">
  <a href="README.zh-CN.md">简体中文</a>
</p>

> Binary classifier predicting whether an F1 driver will pit on the **next** lap. Built on Kaggle Playground Series **S5E6** (2022–2025) — full MLOps loop from raw CSV to a live, deployable inference UI.

<p align="center">
  <a href="https://t0myyy-f1-pit-predictor.hf.space/"><img alt="Try it live" src="https://img.shields.io/badge/Try%20it%20live-▶-1f7a3d?style=flat-square&labelColor=2a323d"></a>
  <a href="https://huggingface.co/spaces/T0MYYY/f1-pit-predictor"><img alt="HF Space" src="https://img.shields.io/badge/🤗%20Space-running-FFD21E?style=flat-square&labelColor=2a323d"></a>
  <img alt="macro-F1" src="https://img.shields.io/badge/macro--F1-0.785-00aa55?style=flat-square&labelColor=2a323d">
  <img alt="ROC-AUC" src="https://img.shields.io/badge/ROC--AUC-0.894-00aa55?style=flat-square&labelColor=2a323d">
  <img alt="Status" src="https://img.shields.io/badge/Status-v2%20shipped-ec5800?style=flat-square&labelColor=2a323d">
  <img alt="Monitoring" src="https://img.shields.io/badge/Monitoring-Evidently%200.4.33-7B68EE?style=flat-square&labelColor=2a323d">
  <img alt="PRs" src="https://img.shields.io/badge/PRs-welcome-1f7a3d?style=flat-square&labelColor=2a323d">
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white&labelColor=2a323d">
  <img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-0.136-009688?style=flat-square&logo=fastapi&logoColor=white&labelColor=2a323d">
  <img alt="Pydantic" src="https://img.shields.io/badge/Pydantic-2-E92063?style=flat-square&logo=pydantic&logoColor=white&labelColor=2a323d">
  <img alt="Uvicorn" src="https://img.shields.io/badge/Uvicorn-0.34-499848?style=flat-square&logo=gunicorn&logoColor=white&labelColor=2a323d">
  <img alt="React" src="https://img.shields.io/badge/React-18-61DAFB?style=flat-square&logo=react&logoColor=black&labelColor=2a323d">
  <img alt="Docker" src="https://img.shields.io/badge/Docker-Ready-2496ED?style=flat-square&logo=docker&logoColor=white&labelColor=2a323d">
</p>

<p align="center">
  <img alt="pandas" src="https://img.shields.io/badge/pandas-2.2-150458?style=flat-square&logo=pandas&logoColor=white&labelColor=2a323d">
  <img alt="scikit-learn" src="https://img.shields.io/badge/scikit--learn-1.5-F7931E?style=flat-square&logo=scikitlearn&logoColor=white&labelColor=2a323d">
  <img alt="XGBoost" src="https://img.shields.io/badge/XGBoost-2.1-ec5800?style=flat-square&labelColor=2a323d">
  <img alt="MLflow" src="https://img.shields.io/badge/MLflow-2.17-0194E2?style=flat-square&logo=mlflow&logoColor=white&labelColor=2a323d">
  <img alt="FLAML" src="https://img.shields.io/badge/FLAML-2.3-7B68EE?style=flat-square&labelColor=2a323d">
  <img alt="Airflow" src="https://img.shields.io/badge/Airflow-2.9-017CEE?style=flat-square&logo=apacheairflow&logoColor=white&labelColor=2a323d">
  <img alt="DVC" src="https://img.shields.io/badge/DVC-3.55-13ADC7?style=flat-square&logo=dvc&logoColor=white&labelColor=2a323d">
</p>

---

## 🔗 Live demo

| | URL |
|---|---|
| **Dashboard (HF Space)** | <https://huggingface.co/spaces/T0MYYY/f1-pit-predictor> |
| **Direct app URL** | <https://t0myyy-f1-pit-predictor.hf.space> |
| **CSV batch inference UI** | <https://t0myyy-f1-pit-predictor.hf.space/predict.html> |
| **Health check** | <https://t0myyy-f1-pit-predictor.hf.space/health> |

`GET /` is the race-playback dashboard (a `REAL MODEL` toggle swaps the synthetic predictions for live `predict_proba` calls). `GET /predict.html` is the drag-and-drop CSV inference page. `POST /predict`, `POST /predict/dashboard`, `POST /predict/csv` are JSON / multipart endpoints — see [§ API](#-api-endpoints).

---

## 🚦 Handoff status

```mermaid
flowchart LR
    A["<b>Member A</b><br/>EDA · features<br/>baseline<br/>✅ done"]
    B["<b>Member B</b><br/>Training pipeline<br/>Airflow · MLflow<br/>✅ done"]
    C["<b>Member C</b><br/>FastAPI · Dashboard<br/>HF Space deploy<br/>✅ done"]
    D["<b>Member D</b><br/>Drift monitoring<br/>Evidently · Reports<br/>✅ done"]

    A --> B --> C --> D

    classDef done fill:#0a3d1f,stroke:#1f7a3d,color:#dff5e7,stroke-width:1px;
    classDef todo fill:#3d2a0a,stroke:#7a5b1f,color:#f5e7c7,stroke-width:1px;
    class A,B,C,D done
```

All four members are done. Full MLOps loop: EDA → training pipeline → deployment → drift monitoring.

---

## 🏗 Architecture

```mermaid
flowchart TB
    subgraph TRAIN ["Training — Member B (host or Airflow)"]
        direction LR
        raw["data/train.csv<br/><i>16-col Kaggle schema</i>"] --> ingest[["src.ingest"]]
        ingest --> parq[("data/interim/raw.parquet")]
        parq --> prep[["src.preprocess<br/><i>feature engineering · year split</i>"]]
        prep --> trainparq[("processed/{train,test}.parquet")]
        trainparq --> trn[["src.train<br/><i>FLAML × {lgbm, xgb, xgb_ld, catboost}</i>"]]
        trn --> mlflow[("MLflow runs<br/>sqlite + artifacts")]
        mlflow --> reg[["src.register<br/><i>pick best macro-F1 · export</i>"]]
        reg --> champ[["<b>models/champion/</b><br/>mlflow pyfunc dir<br/>+ CHAMPION.json baseline"]]
    end

    subgraph SERVE ["Serving — Member C (HF Docker Space)"]
        direction LR
        champ --> api[["FastAPI<br/>src.api:app"]]
        api --> ep1["POST /predict<br/><i>raw JSON records</i>"]
        api --> ep2["POST /predict/dashboard<br/><i>dashboard rows</i>"]
        api --> ep3["POST /predict/csv<br/><i>multipart CSV upload</i>"]
        api --> ui[["dashboard/<br/>React + Babel SPA<br/>+ predict.html"]]
    end

    subgraph MONITOR ["Monitoring — Member D"]
        direction LR
        trainparq -.->|"reference Year 2022-2024"| mon[["monitoring/<br/>model_monitoring.py<br/><i>Evidently AI</i>"]]
        champ -.->|"champion XGBoost Pipeline"| mon
        mon --> s1["S1 · original test<br/><i>F1=0.785  AUC=0.894</i>"]
        mon --> s2["S2 · TyreLife +20<br/><i>F1=0.666 ↓  pit%=55%</i>"]
        mon --> s3["S3 · Compound→SOFT<br/><i>F1=0.751 ↓</i>"]
        mon --> s4["S4 · Degradation ×2<br/><i>F1=0.782 ≈</i>"]
        s1 & s2 & s3 & s4 --> rep[/"monitoring_reports/<br/>4 × HTML + metrics_summary.json"/]
    end

    classDef done fill:#0a3d1f,stroke:#1f7a3d,color:#dff5e7;
```
---

## ⚡ Quick start

### Predict with the trained champion (no training needed)

The champion is committed at [`models/champion/`](models/champion/) (5.4 MB, XGBoost via FLAML). A clean clone is enough to serve predictions.

```bash
git clone https://github.com/EdwardHuang777/F1-Pit-Stop-Prediction.git
cd F1-Pit-Stop-Prediction

python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest tests/                                            # 2/2 smoke tests
python -m src.inference --input data/train.csv --year 2025   # batch predict on 2025 holdout
```

### Run the full inference UI locally (mirror of the HF Space)

```bash
pip install -r requirements.txt
uvicorn src.api:app --host 0.0.0.0 --port 7860
# Dashboard:   http://localhost:7860/
# CSV upload:  http://localhost:7860/predict.html
# Health:      http://localhost:7860/health
```

### Reproduce the Docker deployment

```bash
docker build -f deploy/hf/Dockerfile -t f1-pit-predictor .
docker run -p 7860:7860 f1-pit-predictor
```

Deploy to your own HF Space: `python deploy/hf/push_space.py` (needs `hf auth login` first, and the `REPO_ID` constant updated).

### Retrain

```bash
python -m src.ingest         # ~2s
python -m src.preprocess     # ~30s
python -m src.train          # ~12 min (4 × FLAML @ 150s/algo)
python -m src.register       # ~5s · writes models/champion/
```

Or via Airflow (`docker compose up -d` → trigger `pit_stop_training` at <http://localhost:8080>, login `admin/admin`).

---

## 🔌 API endpoints

[`src/api.py`](src/api.py) exposes:

| Endpoint | Body | Use case |
|---|---|---|
| `GET /health` | — | Liveness probe |
| `POST /predict` | `{"records":[{...raw 15 cols}]}` | Original int-label endpoint, kept for back-compat |
| `POST /predict/dashboard` | `{race, year, rows:[{driver, lap, compound, …}]}` | Per-driver request fired by the dashboard's `REAL MODEL` toggle |
| `POST /predict/csv` | `multipart/form-data file=…csv` | Drag-and-drop CSV inference for `/predict.html` |
| `GET /` | — | StaticFiles mount — serves `dashboard/index.html` |

**Loading the champion directly in Python** (no API needed):

```python
import mlflow.sklearn
model = mlflow.sklearn.load_model("models/champion")
probs = model.predict_proba(features_df)[:, 1]   # raw probabilities
```

**Input shape.** A DataFrame with the 15 raw columns from [`src/config.py:RAW_COLUMNS`](src/config.py) **except** `PitNextLap`. All feature engineering — lag, rolling means, tyre-life buckets, early/late-race flags — is rebuilt inside `prepare_features()` ([`src/inference.py`](src/inference.py)). Feature engineering groups by `(Year, Race, Driver, Stint)`, so **send a full driver-stint history per request** for lag/rolling features to populate correctly.

**Refresh cadence.** When the training pipeline retrains, `models/champion/` is updated in-place. A `git pull` and process restart on the HF Space picks it up — no schema changes, no client work.

---

## Model Monitoring

Drift monitoring is implemented in [`monitoring/model_monitoring.py`](monitoring/model_monitoring.py) using **Evidently AI**.

### How it works

The script reuses the team's existing pipeline (`src.ingest` + `src.preprocess`) to generate processed data, then runs four monitoring scenarios against the deployed champion model:

| Scenario | Change |
|---|---|
| S1 | Original test data (Year 2025), no changes |
| S2 | `TyreLife += 20` |
| S3 | `Compound → SOFT` |
| S4 | `Cumulative_Degradation × 2` |

S2 / S3 / S4 are **independent** (each starts from the original test data), so you can see the isolated impact of each feature change.

### Results

| Scenario | F1-macro | ROC-AUC | Pit% predicted |
|---|---:|---:|---:|
| Champion (train eval) | 0.7852 | 0.8945 | — |
| S1 original test | 0.7852 | 0.8945 | 25.3% |
| S2 TyreLife +20 | 0.6662 | 0.8292 | 54.7% |
| S3 Compound→SOFT | 0.7509 | 0.8562 | 25.5% |
| S4 Degradation ×2 | 0.7823 | 0.8912 | 25.6% |

**Key finding:** `TyreLife` is the most sensitive feature. A +20 shift causes F1 to drop by **−0.119** and pit prediction rate to jump from 25% to 55%. `Compound` has a moderate effect (−0.034). `Cumulative_Degradation` has minimal impact (−0.003), which means the model compensates via correlated features.

### Run

```bash
pip install evidently==0.4.33
python monitoring/model_monitoring.py
```

Reports are saved to `monitoring_reports/` (4 HTML files + `metrics_summary.json`). Open any HTML in a browser to see the full Evidently dashboard with Data Quality, Data Drift, and Classification panels.

### Note on the 2023 anomaly

Year 2023 has ~0.96% pit-next-lap rate vs ~28% in 2022/2024/2025 — a **Kaggle Playground synthetic artifact**, not actual drift. Member A flagged this in [`notebooks/EDA.ipynb`](notebooks/EDA.ipynb) §7. If your dashboard slices by year, label it as `data-source artifact, not model drift` so the on-call doesn't chase a ghost.

### Repository layout

```
F1-Pit-Stop-Prediction/
├── data/                          · Raw Kaggle CSVs + DVC-tracked outputs
├── notebooks/                     · Member A's EDA + feature-engineering source of truth
├── src/                           · Importable pipeline modules
│   ├── config.py                  · Paths, MLflow URI, RANDOM_SEED, AutoML config
│   ├── ingest.py                  · CSV → parquet (16-column schema check)
│   ├── feature_engineering.py     · add_features() — verbatim port of Member A's cell
│   ├── preprocess.py              · FE + year split + build_preprocessor()
│   ├── train.py                   · FLAML per-algorithm AutoML, MLflow per-run logs
│   ├── register.py                · Champion selection, MLflow Registry, export
│   ├── inference.py               · prepare_features() · load_*_model() · CLI
│   └── api.py                     · FastAPI app — /predict, /predict/dashboard, /predict/csv
├── dashboard/                     · React + Babel SPA (CDN, no build) + predict.html CSV page
├── deploy/hf/                     · HF Docker Space — Dockerfile, requirements, push_space.py
├── dags/pit_stop_training_dag.py  · Airflow DAG — 4 BashOperators wrapping src.*
├── tests/                         · Import + config-path smoke tests
├── models/champion/               · Handoff payload (consumed by API and monitoring)
├── monitoring/
│   └── model_monitoring.py        · Evidently AI drift monitoring (4 scenarios)
├── monitoring_reports/            · Generated HTML reports + metrics_summary.json
├── docker-compose.yaml            · Airflow LocalExecutor + Postgres
├── Dockerfile                     · Training image (apache/airflow:2.9.3-python3.12 + ML deps)
├── dvc.yaml / dvc.lock            · DVC pipeline (ingest, preprocess stages)
└── requirements.txt               · Pinned versions
```

---

## 🏆 Champion history

| Version | Algorithm | Test macro-F1 | Test ROC-AUC | Notes |
|---|---|---:|---:|---|
| 1 | catboost | 0.7834 | 0.8944 | Host-trained, 600s budget. v1 register. |
| **2** | **xgboost** | **0.7852** | **0.8945** | First DAG-trained champion. Currently deployed. |


---

## 🛠 Detailed pipeline reference

### A) Standalone Python (fastest for development, ~12 min)

```bash
source .venv/bin/activate
python -m src.ingest        # ~2s
python -m src.preprocess    # ~30s
python -m src.train         # ~12 min (4 × FLAML @ 150s/algo + overhead)
python -m src.register      # ~5s
```

Override the AutoML budget with `TIME_BUDGET=80 python -m src.train` (80s total ≈ 2 min smoke run).

### B) Airflow DAG via Docker (production orchestration, ~20–25 min)

```bash
mkdir -p mlflow              # pre-create bind-mount sources (one-time after fresh clone)
docker compose build         # ~10 min one-time; cached after
docker compose up -d
# Open http://localhost:8080  → admin / admin → trigger pit_stop_training
# Default budget {"time_budget": 600}.  Use {"time_budget": 80} for smoke.
docker compose down          # when done
```

Each `src.*` runs as a `BashOperator` subprocess (not PythonOperator — that hits a fork-after-thread bug with native ML libs).

### C) DVC reproduce (data stages only)

```bash
dvc repro                   # replays ingest + preprocess; doesn't run train/register
```

### MLflow

- **Backend:** SQLite at `mlflow/mlflow.db`, artifacts at `mlflow/artifacts/`
- **Tracking URI:** `sqlite:///mlflow/mlflow.db` ([`src/config.py:21`](src/config.py#L21))
- **UI:** `mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db --port 5001`

**Host ↔ Docker caveat.** MLflow stores **absolute** artifact paths in sqlite at experiment-creation time. The DB from a host run won't work in Docker (and vice versa). When switching environments: `rm -rf mlflow/ mlruns/` first. [`src/train.py:31-52`](src/train.py#L31-L52) detects stale `artifact_location` and fails fast with a hint. `models/champion/` is the durable handoff and is unaffected.

---

## ♻️ Reproducibility

- **Python:** 3.12 on host and inside Docker
- **Random seed:** 42, set in [`src/config.py`](src/config.py) and threaded into FLAML via `clf__seed`
- **Pinned deps:** [`requirements.txt`](requirements.txt)
- **Data identity:** SHA256 of `data/processed/*.parquet` stored in `CHAMPION.json`
- **Code identity:** git SHA stored in `CHAMPION.json`

---

## ⚠️ Gotchas

- **First Docker build is ~10 min** (catboost + xgboost + lightgbm + arm64 wheels). Cached afterward.
- **catboost is slow inside arm64 Docker.** A 600s-budget DAG run can take 20–25 min real-time mostly waiting on catboost. The host venv is ~2× faster (no bind-mount I/O penalty).
- **Don't `rm -rf mlflow/` while Docker is up.** Bring `docker compose down` first; the bind mount can go into a weird state otherwise.
- **Switching host ↔ Docker for training:** wipe `mlflow/` and `mlruns/` between environments. `models/champion/` survives.
- **GitHub flags `data/train.csv` as >50 MB.** Soft warning, not a block — committed for grader convenience.
- **HF Space deploy needs `flaml` and `lightgbm`** even though the champion is XGBoost — the FLAML training wrapper leaves references in the pickle. See [`deploy/hf/requirements.txt`](deploy/hf/requirements.txt).

---

## 🤝 Retrain workflow

1. Pull latest data if changed.
2. `python -m src.train` (host) or trigger Airflow DAG (Docker).
3. `python -m src.register` runs as the last DAG task; otherwise run manually.
4. Inspect `models/champion/CHAMPION.json` — verify `test_macro_f1` beats the prior champion.
5. `git add models/champion/ && git commit -m "Champion v{N}: {algo}, F1={X}"`.
6. `git push` — re-run `python deploy/hf/push_space.py` to ship the new champion to the live Space; Member D updates the drift baseline.

---

<sub>Banner: Pirelli F1 tyre range (Soft · Medium · Hard · Intermediate · Wet). Photo via Wikimedia Commons, CC BY-SA.</sub>
