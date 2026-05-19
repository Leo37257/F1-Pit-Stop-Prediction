"""FastAPI app for F1 pit-stop model inference."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.inference import predict_records


class PredictRequest(BaseModel):
    records: list[dict[str, Any]] = Field(..., min_length=1)


class PredictResponse(BaseModel):
    predictions: list[int]


app = FastAPI(title="F1 Pit-Stop Prediction API")


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
