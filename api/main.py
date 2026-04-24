"""
Fraud Detection Inference API
=================================
FastAPI server with:
  - POST /predict        – fraud probability prediction
  - GET  /health         – health check
  - GET  /metrics        – Prometheus metrics
  - GET  /model/info     – model metadata
  - POST /drift/check    – check current batch for data drift

Prometheus metrics tracked:
  - fraud_api_requests_total        (counter, by endpoint + status)
  - fraud_api_request_duration_secs (histogram, latency)
  - fraud_prediction_probability    (histogram, score distribution)
  - fraud_detected_total            (counter)
  - fraud_recall_current            (gauge, updated per batch)
  - fraud_false_positive_rate       (gauge)
  - feature_drift_score             (gauge, per feature)
"""

import os
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from prometheus_client import (
    Counter, Gauge, Histogram,
    generate_latest, CONTENT_TYPE_LATEST,
    CollectorRegistry, REGISTRY,
)
from pydantic import BaseModel

# ------------------------------------------------------------------ #
# Prometheus metrics                                                   #
# ------------------------------------------------------------------ #

REQUEST_COUNT = Counter(
    "fraud_api_requests_total",
    "Total API requests",
    ["method", "endpoint", "status"],
)
REQUEST_LATENCY = Histogram(
    "fraud_api_request_duration_seconds",
    "Request latency in seconds",
    ["endpoint"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)
FRAUD_PROBABILITY = Histogram(
    "fraud_prediction_probability",
    "Distribution of predicted fraud probabilities",
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)
FRAUD_DETECTED = Counter(
    "fraud_detected_total",
    "Total number of fraud cases detected by the model",
)
RECALL_GAUGE = Gauge(
    "fraud_recall_current",
    "Current model recall on recent predictions (rolling estimate)",
)
FPR_GAUGE = Gauge(
    "fraud_false_positive_rate",
    "Current false positive rate (rolling estimate)",
)
FEATURE_DRIFT_GAUGE = Gauge(
    "feature_drift_score",
    "KS statistic for feature drift detection",
    ["feature"],
)
MISSING_VALUE_RATE = Gauge(
    "feature_missing_value_rate",
    "Rate of missing values in incoming data",
    ["feature"],
)

# ------------------------------------------------------------------ #
# Pydantic models                                                      #
# ------------------------------------------------------------------ #

class PredictRequest(BaseModel):
    """Single transaction prediction request."""
    features: Dict[str, Any]
    transaction_id: Optional[str] = None


class BatchPredictRequest(BaseModel):
    """Batch prediction request."""
    transactions: List[Dict[str, Any]]


class DriftCheckRequest(BaseModel):
    """Drift detection request for a batch of feature vectors."""
    batch: List[Dict[str, Any]]


# ------------------------------------------------------------------ #
# Application lifespan                                                 #
# ------------------------------------------------------------------ #

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    from api.model_loader import loader
    loader.load()
    # Initialise recall / FPR gauges with neutral values
    RECALL_GAUGE.set(0.0)
    FPR_GAUGE.set(0.0)
    print("[API] Fraud Detection Inference API started.")
    yield
    print("[API] Fraud Detection Inference API shutting down.")


app = FastAPI(
    title="Fraud Detection API",
    description=(
        "Production inference API for IEEE CIS Fraud Detection model. "
        "Exposes /predict, /health, /metrics (Prometheus), and /model/info endpoints."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ------------------------------------------------------------------ #
# Helper: feature dict → numpy array                                  #
# ------------------------------------------------------------------ #

def features_to_array(features: Dict[str, Any]) -> np.ndarray:
    """Convert feature dict to a float32 numpy row vector."""
    values = []
    for val in features.values():
        try:
            values.append(float(val) if val is not None else 0.0)
        except (TypeError, ValueError):
            values.append(0.0)
    return np.array(values, dtype=np.float32).reshape(1, -1)


# ------------------------------------------------------------------ #
# Endpoints                                                            #
# ------------------------------------------------------------------ #

@app.get("/health")
def health_check():
    """Health check endpoint for load balancer and monitoring."""
    from api.model_loader import loader
    return {
        "status": "healthy",
        "model_loaded": loader.is_loaded,
        "model_type": loader.model_type,
    }


@app.post("/predict")
def predict(request: PredictRequest):
    """
    Predict fraud probability for a single transaction.
    Returns:
      - fraud_probability: float [0, 1]
      - is_fraud: bool (threshold = 0.5)
      - confidence: "high" | "medium" | "low"
      - transaction_id: str (echoed back)
    """
    from api.model_loader import loader

    start = time.time()
    endpoint = "/predict"

    try:
        # Track missing values
        for feat_name, val in request.features.items():
            if val is None:
                MISSING_VALUE_RATE.labels(feature=feat_name).inc()

        X = features_to_array(request.features)
        fraud_prob = float(loader.predict_proba(X)[0])

        is_fraud = fraud_prob >= 0.5
        if fraud_prob >= 0.8:
            confidence = "high"
        elif fraud_prob >= 0.5:
            confidence = "medium"
        else:
            confidence = "low"

        # Update Prometheus metrics
        FRAUD_PROBABILITY.observe(fraud_prob)
        REQUEST_COUNT.labels(method="POST", endpoint=endpoint, status="200").inc()
        if is_fraud:
            FRAUD_DETECTED.inc()

        latency = time.time() - start
        REQUEST_LATENCY.labels(endpoint=endpoint).observe(latency)

        return {
            "transaction_id":   request.transaction_id,
            "fraud_probability": round(fraud_prob, 6),
            "is_fraud":         bool(is_fraud),
            "confidence":       confidence,
            "latency_ms":       round(latency * 1000, 2),
        }

    except Exception as e:
        REQUEST_COUNT.labels(method="POST", endpoint=endpoint, status="500").inc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch")
def predict_batch(request: BatchPredictRequest):
    """Batch prediction for multiple transactions."""
    from api.model_loader import loader

    start = time.time()
    endpoint = "/predict/batch"

    try:
        results = []
        for i, feat_dict in enumerate(request.transactions):
            X = features_to_array(feat_dict)
            fraud_prob = float(loader.predict_proba(X)[0])
            is_fraud = fraud_prob >= 0.5
            FRAUD_PROBABILITY.observe(fraud_prob)
            if is_fraud:
                FRAUD_DETECTED.inc()
            results.append({
                "index": i,
                "fraud_probability": round(fraud_prob, 6),
                "is_fraud": bool(is_fraud),
            })

        REQUEST_COUNT.labels(method="POST", endpoint=endpoint, status="200").inc()
        REQUEST_LATENCY.labels(endpoint=endpoint).observe(time.time() - start)

        fraud_rate = sum(1 for r in results if r["is_fraud"]) / max(len(results), 1)
        return {
            "n_transactions": len(results),
            "fraud_rate":     round(fraud_rate, 4),
            "predictions":    results,
        }

    except Exception as e:
        REQUEST_COUNT.labels(method="POST", endpoint=endpoint, status="500").inc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/info")
def model_info():
    """Return model metadata."""
    from api.model_loader import loader
    return {
        "model_type":  loader.model_type,
        "is_loaded":   loader.is_loaded,
        "metadata":    loader.metadata,
        "registry":    os.environ.get("MODEL_REGISTRY", "/tmp/fraud_model_registry"),
    }


@app.post("/drift/check")
def drift_check(request: DriftCheckRequest):
    """
    Check a batch of incoming transactions for data drift.
    Updates Prometheus feature_drift_score gauges.
    """
    import pandas as pd
    from api.drift_detector import detector

    start = time.time()

    try:
        df = pd.DataFrame(request.batch)
        result = detector.detect(df)

        # Update Prometheus gauges for each feature
        for feat, scores in result.get("features", {}).items():
            FEATURE_DRIFT_GAUGE.labels(feature=feat).set(scores["ks_statistic"])

        REQUEST_COUNT.labels(method="POST", endpoint="/drift/check", status="200").inc()
        REQUEST_LATENCY.labels(endpoint="/drift/check").observe(time.time() - start)

        return result

    except Exception as e:
        REQUEST_COUNT.labels(method="POST", endpoint="/drift/check", status="500").inc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/metrics/update")
def update_model_metrics(recall: float, fpr: float):
    """
    Endpoint for updating model performance gauges.
    Called by monitoring / evaluation scripts.
    """
    RECALL_GAUGE.set(recall)
    FPR_GAUGE.set(fpr)
    return {"status": "updated", "recall": recall, "fpr": fpr}


@app.get("/metrics")
def prometheus_metrics():
    """Expose Prometheus metrics in text format."""
    return Response(content=generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)


# ------------------------------------------------------------------ #
# AlertManager webhook → GitHub Actions trigger                       #
# ------------------------------------------------------------------ #

ALERT_TRIGGER_COUNT = Counter(
    "alert_github_trigger_total",
    "Number of times a GitHub Actions retraining was triggered via alert",
    ["alert_name", "status"],
)


class AlertManagerWebhook(BaseModel):
    """AlertManager webhook payload."""
    alerts: List[Dict[str, Any]] = []
    status: Optional[str] = None


@app.post("/alert/webhook")
def alert_webhook(payload: AlertManagerWebhook):
    """
    Receives AlertManager webhooks and triggers GitHub Actions
    retraining workflow when FraudRecallDrop or FeatureDriftDetected fires.

    Requires env vars:
      GITHUB_TOKEN  – Personal Access Token with repo scope
      GITHUB_REPO   – e.g. syedibrahim-dev/ml_pipeline
    """
    import urllib.request
    import json as json_lib

    github_token = os.environ.get("GITHUB_TOKEN", "")
    github_repo  = os.environ.get("GITHUB_REPO", "")

    TRIGGER_ALERTS = {"FraudRecallDrop", "FeatureDriftDetected"}

    triggered = []
    for alert in payload.alerts:
        alert_name = alert.get("labels", {}).get("alertname", "")
        alert_status = alert.get("status", "firing")

        if alert_name not in TRIGGER_ALERTS or alert_status != "firing":
            continue

        event_type = (
            "model-performance-drop"
            if alert_name == "FraudRecallDrop"
            else "drift-detected"
        )

        if not github_token or not github_repo:
            print(f"[alert/webhook] {alert_name} fired — GITHUB_TOKEN/GITHUB_REPO not set, skipping trigger")
            ALERT_TRIGGER_COUNT.labels(alert_name=alert_name, status="skipped").inc()
            continue

        # Call GitHub repository_dispatch API
        url  = f"https://api.github.com/repos/{github_repo}/dispatches"
        body = json_lib.dumps({"event_type": event_type}).encode()
        req  = urllib.request.Request(
            url,
            data=body,
            headers={
                "Authorization": f"token {github_token}",
                "Accept": "application/vnd.github.v3+json",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                print(f"[alert/webhook] Triggered GitHub Actions ({event_type}) — HTTP {resp.status}")
                ALERT_TRIGGER_COUNT.labels(alert_name=alert_name, status="triggered").inc()
                triggered.append({"alert": alert_name, "event_type": event_type, "http_status": resp.status})
        except Exception as e:
            print(f"[alert/webhook] Failed to trigger GitHub Actions: {e}")
            ALERT_TRIGGER_COUNT.labels(alert_name=alert_name, status="error").inc()

    return {
        "received_alerts": len(payload.alerts),
        "triggered": triggered,
    }
