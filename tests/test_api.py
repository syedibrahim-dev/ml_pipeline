"""
Unit tests for the FastAPI inference server.
Tests /health, /predict, /predict/batch, /metrics, /drift/check endpoints.
Uses TestClient (in-process, no real server needed).
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ------------------------------------------------------------------ #
# Client fixture                                                        #
# ------------------------------------------------------------------ #

@pytest.fixture(scope="module")
def client():
    """Return a FastAPI TestClient for the fraud detection API."""
    try:
        from fastapi.testclient import TestClient
        from api.main import app
        return TestClient(app)
    except ImportError:
        pytest.skip("fastapi or httpx not installed")


def sample_features(n_features: int = 20) -> dict:
    """Generate a sample feature dict for prediction requests."""
    rng = np.random.default_rng(42)
    return {f"feat_{i}": float(rng.standard_normal(1)[0]) for i in range(n_features)}


# ------------------------------------------------------------------ #
# /health                                                              #
# ------------------------------------------------------------------ #

class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        r = client.get("/health")
        assert r.status_code == 200

    def test_health_has_status_field(self, client):
        r = client.get("/health")
        data = r.json()
        assert "status" in data
        assert data["status"] == "healthy"

    def test_health_has_model_loaded_field(self, client):
        r = client.get("/health")
        data = r.json()
        assert "model_loaded" in data
        assert isinstance(data["model_loaded"], bool)

    def test_health_has_model_type_field(self, client):
        r = client.get("/health")
        data = r.json()
        assert "model_type" in data


# ------------------------------------------------------------------ #
# /predict                                                             #
# ------------------------------------------------------------------ #

class TestPredictEndpoint:
    def test_predict_returns_200(self, client):
        r = client.post("/predict", json={"features": sample_features()})
        assert r.status_code == 200

    def test_predict_returns_fraud_probability(self, client):
        r = client.post("/predict", json={"features": sample_features()})
        data = r.json()
        assert "fraud_probability" in data

    def test_fraud_probability_in_range(self, client):
        r = client.post("/predict", json={"features": sample_features()})
        prob = r.json()["fraud_probability"]
        assert 0.0 <= prob <= 1.0

    def test_predict_returns_is_fraud_bool(self, client):
        r = client.post("/predict", json={"features": sample_features()})
        data = r.json()
        assert "is_fraud" in data
        assert isinstance(data["is_fraud"], bool)

    def test_predict_returns_confidence_field(self, client):
        r = client.post("/predict", json={"features": sample_features()})
        data = r.json()
        assert "confidence" in data
        assert data["confidence"] in ("high", "medium", "low")

    def test_predict_returns_latency_ms(self, client):
        r = client.post("/predict", json={"features": sample_features()})
        data = r.json()
        assert "latency_ms" in data
        assert data["latency_ms"] >= 0

    def test_predict_echoes_transaction_id(self, client):
        r = client.post("/predict", json={
            "features": sample_features(),
            "transaction_id": "TX-999"
        })
        data = r.json()
        assert data.get("transaction_id") == "TX-999"

    def test_predict_empty_features_returns_200(self, client):
        """Empty feature dict should still return a prediction (model uses defaults)."""
        r = client.post("/predict", json={"features": {}})
        assert r.status_code == 200

    def test_predict_with_none_values(self, client):
        """None values in features should be handled gracefully."""
        features = {f"feat_{i}": None for i in range(5)}
        r = client.post("/predict", json={"features": features})
        assert r.status_code == 200


# ------------------------------------------------------------------ #
# /predict/batch                                                       #
# ------------------------------------------------------------------ #

class TestBatchPredictEndpoint:
    def test_batch_predict_returns_200(self, client):
        txns = [{"features": sample_features()} for _ in range(5)]
        # Flatten features for batch endpoint
        batch = [sample_features() for _ in range(5)]
        r = client.post("/predict/batch", json={"transactions": batch})
        assert r.status_code == 200

    def test_batch_predict_correct_count(self, client):
        n = 7
        batch = [sample_features() for _ in range(n)]
        r = client.post("/predict/batch", json={"transactions": batch})
        data = r.json()
        assert data["n_transactions"] == n

    def test_batch_predict_fraud_rate_in_range(self, client):
        batch = [sample_features() for _ in range(10)]
        r = client.post("/predict/batch", json={"transactions": batch})
        rate = r.json()["fraud_rate"]
        assert 0.0 <= rate <= 1.0


# ------------------------------------------------------------------ #
# /metrics                                                             #
# ------------------------------------------------------------------ #

class TestMetricsEndpoint:
    def test_metrics_returns_200(self, client):
        r = client.get("/metrics")
        assert r.status_code == 200

    def test_metrics_content_type_is_prometheus(self, client):
        r = client.get("/metrics")
        # Prometheus format uses text/plain
        assert "text/plain" in r.headers.get("content-type", "")

    def test_metrics_contains_request_counter(self, client):
        # Make a predict request first, then check counter
        client.post("/predict", json={"features": sample_features()})
        r = client.get("/metrics")
        assert "fraud_api_requests_total" in r.text

    def test_metrics_contains_latency_histogram(self, client):
        r = client.get("/metrics")
        assert "fraud_api_request_duration_seconds" in r.text


# ------------------------------------------------------------------ #
# /model/info                                                          #
# ------------------------------------------------------------------ #

class TestModelInfoEndpoint:
    def test_model_info_returns_200(self, client):
        r = client.get("/model/info")
        assert r.status_code == 200

    def test_model_info_has_model_type(self, client):
        r = client.get("/model/info")
        data = r.json()
        assert "model_type" in data

    def test_model_info_has_is_loaded(self, client):
        r = client.get("/model/info")
        data = r.json()
        assert "is_loaded" in data


# ------------------------------------------------------------------ #
# /drift/check                                                         #
# ------------------------------------------------------------------ #

class TestDriftCheckEndpoint:
    def test_drift_check_returns_200(self, client):
        batch = [sample_features() for _ in range(20)]
        r = client.post("/drift/check", json={"batch": batch})
        assert r.status_code == 200

    def test_drift_check_has_drift_detected_field(self, client):
        batch = [sample_features() for _ in range(20)]
        r = client.post("/drift/check", json={"batch": batch})
        data = r.json()
        assert "drift_detected" in data
        assert isinstance(data["drift_detected"], bool)

    def test_drift_check_empty_batch_returns_200(self, client):
        r = client.post("/drift/check", json={"batch": []})
        assert r.status_code == 200
