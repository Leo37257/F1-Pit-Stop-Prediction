---
title: F1 Pit Predictor
emoji: 🏎
colorFrom: red
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# F1 Pit-Stop Predictor

XGBoost champion model (macro-F1 **0.785**, ROC-AUC **0.894**) served via FastAPI behind a dark-mode engineer dashboard.

- **`/`** — race-playback dashboard with a `REAL MODEL` toggle that overlays live inference on the synthetic race.
- **`/predict.html`** — upload a CSV of raw F1 lap rows and get `pPit` + `pred` per row (with optional macro-F1 if `PitNextLap` is included).
- **`POST /predict/csv`** — JSON API behind the upload page.
- **`POST /predict/dashboard`** — per-row JSON inference used by the dashboard.
- **`POST /predict`** — original int-label endpoint.
- **`GET /health`** — liveness probe.
