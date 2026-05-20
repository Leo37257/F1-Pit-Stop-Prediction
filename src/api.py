"""FastAPI app for F1 pit-stop model inference."""

from __future__ import annotations

import io
from typing import Any

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src import config
from src.inference import (
    RAW_INPUT_COLUMNS,
    load_sklearn_model,
    predict_records,
    prepare_features,
)


class PredictRequest(BaseModel):
    records: list[dict[str, Any]] = Field(..., min_length=1)


class PredictResponse(BaseModel):
    predictions: list[int]


class DashboardRow(BaseModel):
    driver: str
    lap: int
    compound: str
    stint: int
    tyreLife: int
    lapTime: float
    lapDelta: float
    cumDeg: float
    raceProg: float
    position: int
    posChange: int
    isStintStart: bool


class DashboardPredictRequest(BaseModel):
    race: str
    year: int
    rows: list[DashboardRow] = Field(..., min_length=1)


class DashboardPrediction(BaseModel):
    driver: str
    lap: int
    pPit: float
    pred: int


class DashboardPredictResponse(BaseModel):
    predictions: list[DashboardPrediction]


app = FastAPI(title="F1 Pit-Stop Prediction API")

_sklearn_model: Any = None

COMPOUND_MAP = {"S": "SOFT", "M": "MEDIUM", "H": "HARD", "I": "INTERMEDIATE", "W": "WET"}


def _get_sklearn_model() -> Any:
    global _sklearn_model
    if _sklearn_model is None:
        _sklearn_model = load_sklearn_model()
    return _sklearn_model


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    try:
        predictions = predict_records(request.records)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return PredictResponse(predictions=predictions)


@app.post("/predict/dashboard", response_model=DashboardPredictResponse)
def predict_dashboard(request: DashboardPredictRequest) -> DashboardPredictResponse:
    rows = request.rows
    raw = pd.DataFrame({
        "id":                     [f"{r.driver}_{r.lap}" for r in rows],
        "Driver":                 [r.driver for r in rows],
        "Compound":               [COMPOUND_MAP.get(r.compound, r.compound) for r in rows],
        "Race":                   [request.race] * len(rows),
        "Year":                   [request.year] * len(rows),
        "PitStop":                [int(r.isStintStart) for r in rows],
        "LapNumber":              [r.lap for r in rows],
        "Stint":                  [r.stint for r in rows],
        "TyreLife":               [r.tyreLife for r in rows],
        "Position":               [r.position for r in rows],
        "LapTime (s)":            [r.lapTime for r in rows],
        "LapTime_Delta":          [r.lapDelta for r in rows],
        "Cumulative_Degradation": [r.cumDeg for r in rows],
        "RaceProgress":           [r.raceProg for r in rows],
        "Position_Change":        [r.posChange for r in rows],
    })
    try:
        model = _get_sklearn_model()
        features = prepare_features(raw)
        probs = model.predict_proba(features)[:, 1]
        preds = (probs >= 0.5).astype(int)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return DashboardPredictResponse(predictions=[
        DashboardPrediction(driver=r.driver, lap=r.lap, pPit=float(p), pred=int(d))
        for r, p, d in zip(rows, probs, preds)
    ])


@app.post("/predict/csv")
async def predict_csv(file: UploadFile = File(...)) -> dict[str, Any]:
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty file")
    try:
        raw = pd.read_csv(io.BytesIO(contents))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {exc}") from exc

    try:
        model = _get_sklearn_model()
        features = prepare_features(raw)
        probs = model.predict_proba(features)[:, 1]
        preds = (probs >= 0.5).astype(int)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    out = raw.copy()
    out["pPit"] = probs
    out["pred"] = preds

    f1 = None
    if config.TARGET_COL in raw.columns:
        from sklearn.metrics import f1_score
        try:
            f1 = float(f1_score(raw[config.TARGET_COL].astype(int), preds, average="macro"))
        except Exception:
            f1 = None

    return {
        "count": int(len(out)),
        "f1_macro": f1,
        "expected_columns": RAW_INPUT_COLUMNS,
        "rows": out.to_dict(orient="records"),
    }


app.mount("/", StaticFiles(directory="dashboard", html=True), name="static")
