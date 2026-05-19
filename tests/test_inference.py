from __future__ import annotations

import pandas as pd


def _raw_rows() -> list[dict]:
    return [
        {
            "id": 2,
            "Driver": "VER",
            "Compound": "MEDIUM",
            "Race": "Dutch Grand Prix",
            "Year": 2025,
            "PitStop": 0,
            "LapNumber": 12,
            "Stint": 1,
            "TyreLife": 12.0,
            "Position": 1,
            "LapTime (s)": 75.2,
            "LapTime_Delta": 0.2,
            "Cumulative_Degradation": 1.1,
            "RaceProgress": 0.24,
            "Position_Change": 0.0,
            "PitNextLap": 0,
        },
        {
            "id": 1,
            "Driver": "VER",
            "Compound": "SOFT",
            "Race": "Dutch Grand Prix",
            "Year": 2025,
            "PitStop": 0,
            "LapNumber": 11,
            "Stint": 1,
            "TyreLife": 11.0,
            "Position": 1,
            "LapTime (s)": 75.0,
            "LapTime_Delta": 0.0,
            "Cumulative_Degradation": 0.9,
            "RaceProgress": 0.22,
            "Position_Change": 0.0,
            "PitNextLap": 0,
        },
    ]


def test_prepare_features_uses_processed_schema_and_preserves_input_order():
    from src.inference import prepare_features
    from src.preprocess import FEATURE_COLUMNS

    features = prepare_features(pd.DataFrame(_raw_rows()))

    assert list(features.columns) == FEATURE_COLUMNS
    assert features["Compound"].tolist() == ["MEDIUM", "SOFT"]
    assert features["TyreLife_bucket"].tolist() == ["11_15", "11_15"]


def test_score_raw_dataframe_requires_target_column():
    import pytest

    from src.inference import score_raw_dataframe

    raw = pd.DataFrame(_raw_rows()).drop(columns=["PitNextLap"])

    with pytest.raises(ValueError, match="PitNextLap"):
        score_raw_dataframe(raw, model=object())


def test_predict_endpoint_returns_predictions(monkeypatch):
    from fastapi.testclient import TestClient

    from src import api

    monkeypatch.setattr(api, "predict_records", lambda records: [1, 0])
    client = TestClient(api.app)

    response = client.post("/predict", json={"records": _raw_rows()})

    assert response.status_code == 200
    assert response.json() == {"predictions": [1, 0]}


def test_predict_endpoint_rejects_empty_records():
    from fastapi.testclient import TestClient

    from src.api import app

    response = TestClient(app).post("/predict", json={"records": []})

    assert response.status_code == 422
