"""
Unit tests for FastAPI prediction endpoints.
Uses a mock predictor so no trained model is required.
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# Patch prediction service BEFORE importing the app
MOCK_RESULT = {
    "prediction": 1,
    "prediction_label": "Churn",
    "churn_probability": 0.82,
    "no_churn_probability": 0.18,
}

SAMPLE_PAYLOAD = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "No",
    "MultipleLines": "No phone service",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85,
    "TotalCharges": 29.85,
}


@pytest.fixture
def client():
    with patch("churn.api.prediction_service.get_predictor") as mock_get:
        mock_predictor = MagicMock()
        mock_predictor.predict.return_value = MOCK_RESULT
        mock_predictor.predict_batch.return_value = [MOCK_RESULT, MOCK_RESULT]
        mock_predictor._load_artifacts.return_value = None
        mock_get.return_value = mock_predictor

        from churn.api.main import app
        with TestClient(app) as c:
            yield c


def test_root_returns_200(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "service" in resp.json()


def test_health_endpoint(client):
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data
    assert "model_loaded" in data


def test_predict_returns_200(client):
    resp = client.post("/api/v1/predict", json=SAMPLE_PAYLOAD)
    assert resp.status_code == 200


def test_predict_response_schema(client):
    resp = client.post("/api/v1/predict", json=SAMPLE_PAYLOAD)
    data = resp.json()
    assert "prediction" in data
    assert "churn_probability" in data
    assert "prediction_label" in data
    assert data["prediction"] in (0, 1)
    assert 0.0 <= data["churn_probability"] <= 1.0


def test_predict_with_high_risk_customer(client):
    resp = client.post("/api/v1/predict", json=SAMPLE_PAYLOAD)
    assert resp.status_code == 200
    data = resp.json()
    assert data["prediction_label"] in ("Churn", "No Churn")


def test_predict_missing_field_returns_422(client):
    bad_payload = {k: v for k, v in SAMPLE_PAYLOAD.items() if k != "tenure"}
    resp = client.post("/api/v1/predict", json=bad_payload)
    assert resp.status_code == 422


def test_batch_predict_endpoint(client):
    batch_payload = {"records": [SAMPLE_PAYLOAD, SAMPLE_PAYLOAD]}
    resp = client.post("/api/v1/predict/batch", json=batch_payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 2
    assert "churn_rate" in data
    assert len(data["results"]) == 2


def test_batch_predict_too_large_returns_422(client):
    batch_payload = {"records": [SAMPLE_PAYLOAD] * 1001}
    resp = client.post("/api/v1/predict/batch", json=batch_payload)
    assert resp.status_code == 422
