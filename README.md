# Fraud Shield — MLOps Assignment 4

**Fraud Shield [Python | MLOps] :** A production-grade fraud detection system built on Kubeflow Pipelines, serving real-time predictions via a FastAPI model registry with XGBoost, LightGBM, and RF-Hybrid ensembles — monitored live through Prometheus and Grafana dashboards with automated drift detection and retraining strategies.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Running the Pipeline](#running-the-pipeline)
- [API Usage](#api-usage)
- [Monitoring Stack](#monitoring-stack)
- [Results](#results)
- [CI/CD](#cicd)
- [Tests](#tests)

---

## Overview

This project implements a full MLOps lifecycle for detecting fraudulent financial transactions using the [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) dataset (~590k transactions, 3.5% fraud rate).

| Component | Technology |
|-----------|-----------|
| Pipeline Orchestration | Kubeflow Pipelines v2 (KFP v2.9.0) |
| Model Training | XGBoost 2.1, LightGBM 4.5, scikit-learn RF |
| Explainability | SHAP TreeExplainer |
| Model Serving | FastAPI + Uvicorn |
| Monitoring | Prometheus + Grafana + AlertManager |
| Drift Detection | KS Two-Sample Test (scipy) |
| Class Imbalance | SMOTE (imbalanced-learn) + class_weight |
| Containerization | Docker / Docker Compose |
| CI/CD | GitHub Actions (4-stage) |
| Kubernetes | Minikube + kubectl |

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  Kubeflow Pipeline (KFP v2)              │
│                                                          │
│  Data        Data        Pre-        Feature    Model    │
│  Ingestion → Validation → processing → Engineering → Training │
│                                                    ↓     │
│                                             Evaluation   │
│                                                    ↓     │
│                                             Deployment   │
└─────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────┐
│              FastAPI Prediction Service                  │
│   /predict  /predict/batch  /health  /metrics  /drift   │
└─────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────┐
│                 Monitoring Stack                         │
│   Prometheus → Grafana (3 dashboards) + AlertManager    │
└─────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
mlops_assign#4/
├── components/                  # KFP v2 pipeline components
│   ├── data_ingestion.py        # Load CSV / generate synthetic data
│   ├── data_validation.py       # 8-check validation suite
│   ├── preprocessing.py         # Imputation, encoding, SMOTE, temporal split
│   ├── feature_engineering.py   # V-group aggregates, RF importance
│   ├── model_training.py        # XGBoost / LightGBM / RF-Hybrid + SHAP
│   ├── model_evaluation.py      # ROC, PR curve, confusion matrix plots
│   └── model_deployment.py      # Champion/challenger model registry
├── pipelines/
│   ├── fraud_pipeline.py        # 7-stage KFP DAG definition
│   └── compiled/
│       └── fraud_detection_pipeline.yaml   # Compiled KFP v2 YAML
├── api/
│   ├── main.py                  # FastAPI app with Prometheus metrics
│   ├── model_loader.py          # Singleton model loader
│   └── drift_detector.py        # KS drift detector
├── drift/
│   ├── simulate_drift.py        # Temporal drift simulation
│   └── retraining_strategy.py   # 90-day strategy comparison
├── monitoring/
│   ├── docker-compose.yml       # Full monitoring stack
│   ├── prometheus.yml           # Scrape config
│   ├── alert_rules.yml          # 8 AlertManager rules
│   ├── alertmanager.yml         # Alert routing
│   └── grafana/
│       ├── provisioning/        # Auto-provision datasource + dashboards
│       └── dashboards/          # system_health / model_performance / data_drift
├── k8s/
│   ├── namespace.yaml           # fraud-detection namespace
│   ├── resource_quota.yaml      # CPU/memory quotas + LimitRange
│   └── persistent_volume.yaml   # hostPath PVs for data + pipeline root
├── tests/
│   ├── test_data_validation.py  # 18 tests
│   ├── test_preprocessing.py    # 16 tests
│   ├── test_model_training.py   # 17 tests
│   └── test_api.py              # 26 tests
├── results/
│   ├── metrics/                 # JSON experiment results
│   └── plots/                   # SHAP, drift, retraining plots
├── .github/workflows/
│   ├── ci_cd.yml                # 4-stage CI/CD pipeline
│   └── drift_trigger.yml        # Daily drift check cron
├── Dockerfile                   # Multi-stage build (python:3.10-slim)
├── requirements.txt
└── run_pipeline.py              # Local runner (no Kubernetes required)
```

---

## Setup & Installation

### Prerequisites

- Python 3.10+
- Docker + Docker Compose
- Minikube (for Kubeflow deployment)

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Dataset

Download the IEEE-CIS Fraud Detection dataset from Kaggle:

```bash
KAGGLE_API_TOKEN=<your_token> kaggle competitions download \
  -c ieee-fraud-detection -p data/raw/
unzip data/raw/ieee-fraud-detection.zip -d data/raw/
```

> If the dataset is unavailable, the pipeline automatically generates 50k synthetic rows matching the full IEEE-CIS schema (V1–V339 features) as a fallback.

---

## Running the Pipeline

### Local (no Kubernetes)

```bash
# Run all tasks
python3 run_pipeline.py

# Run specific tasks
python3 run_pipeline.py --task models
python3 run_pipeline.py --task imbalance
python3 run_pipeline.py --task cost
python3 run_pipeline.py --task drift
python3 run_pipeline.py --task retrain
python3 run_pipeline.py --task explain
```

### Kubeflow (Minikube)

```bash
# Start minikube
minikube start --cpus=4 --memory=8192 --driver=docker

# Apply Kubernetes manifests
kubectl apply -f k8s/

# Compile and submit pipeline
python3 pipelines/fraud_pipeline.py --submit \
  --host http://localhost:8080
```

---

## API Usage

### Start the API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/predict` | Single transaction fraud prediction |
| `POST` | `/predict/batch` | Batch predictions |
| `GET` | `/health` | Service health check |
| `GET` | `/metrics` | Prometheus metrics |
| `POST` | `/drift/check` | Run KS drift detection |
| `GET` | `/model/info` | Current model metadata |

### Example Request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "TransactionAmt": 100.0,
      "card1": 1234,
      "card2": 500,
      "P_emaildomain": "gmail.com",
      "C1": 1.0, "C2": 1.0,
      "V1": 1.0, "V2": 1.0
    },
    "transaction_id": "tx_001"
  }'
```

```json
{
  "transaction_id": "tx_001",
  "fraud_probability": 0.337,
  "is_fraud": false,
  "confidence": "low",
  "latency_ms": 8.5
}
```

---

## Monitoring Stack

```bash
cd monitoring
docker compose up -d
```

| Service | URL |
|---------|-----|
| Fraud Detection API | http://localhost:8000 |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 (admin/admin) |
| AlertManager | http://localhost:9093 |
| Node Exporter | http://localhost:9100 |
| cAdvisor | http://localhost:8080 |

### Grafana Dashboards

Three dashboards are auto-provisioned on startup:

1. **System Health** — Request rate, P50/P95/P99 latency, error rate, CPU/memory usage
2. **Model Performance** — Fraud recall gauge, detection rate, confidence distribution, PR tradeoff
3. **Data Drift** — KS scores over time, per-feature drift bar chart, missing value rate, active alerts

### Alert Rules

| Alert | Condition | Severity |
|-------|-----------|----------|
| `FraudRecallDrop` | recall < 0.80 for 5m | critical |
| `FeatureDriftDetected` | KS score > 0.10 for 2m | warning |
| `HighAPILatency` | P95 latency > 1s for 5m | warning |
| `FraudAPIDown` | target unreachable for 1m | critical |

---

## Results

### Model Comparison

| Model | Recall | Precision | AUC-ROC | Biz Cost | Time |
|-------|--------|-----------|---------|----------|------|
| XGBoost | 0.0763 | 0.0744 | 0.5312 | 1202 | 15.0s |
| LightGBM | **0.1017** | 0.0678 | 0.5262 | 1225 | 2.1s |
| RF-Hybrid | 0.0763 | 0.0310 | 0.4982 | 1371 | 4.4s |

> Metrics on synthetic data (30k rows). Real IEEE-CIS dataset yields significantly higher AUC due to richer feature patterns.

### Class Imbalance Strategy

| Method | Recall | AUC-ROC | Business Cost |
|--------|--------|---------|---------------|
| class_weight | 0.0593 | 0.5295 | 1389 |
| SMOTE | 0.0000 | **0.5711** | 1180 |

### Cost-Sensitive Learning (FN cost = 10× FP)

| Variant | Recall | scale_pos_weight | Business Cost |
|---------|--------|-----------------|---------------|
| Standard | 0.0593 | 26.49 | 1389 |
| Cost-Sensitive | **0.6356** | 264.91 | 2765 |

Cost-sensitive learning improves fraud recall by **+57.6 percentage points**.

### Retraining Strategy (90-day simulation)

| Strategy | Avg Recall | Stability | Retrains | Cost |
|----------|-----------|-----------|---------|------|
| Threshold-only | 0.866 | 0.974 | 5 | 5.0 |
| Periodic-only | 0.890 | **0.982** | 11 | 11.0 |
| **Hybrid** | **0.890** | **0.982** | 11 | 11.0 |

**Recommendation:** Hybrid strategy balances recall stability with compute efficiency.

### SHAP Top Features

| Rank | Feature | Mean |SHAP| |
|------|---------|-------------|
| 1 | C1 | 0.1122 |
| 2 | C4 | 0.1097 |
| 3 | C9 | 0.1082 |
| 4 | C2 | 0.1080 |
| 5 | V8 | 0.0990 |

---

## CI/CD

### GitHub Actions Workflow (`.github/workflows/ci_cd.yml`)

```
lint → test → build (Docker) → deploy
```

| Stage | Description |
|-------|-------------|
| `lint` | flake8 + black format check |
| `test` | pytest (77 tests) with coverage |
| `build` | Docker image build + push to DockerHub |
| `deploy` | Compile KFP YAML + notify |

### Drift Trigger (`.github/workflows/drift_trigger.yml`)

- Runs daily at 06:00 UTC (`cron: '0 6 * * *'`)
- Supports manual `workflow_dispatch` with `force_retrain` input
- Triggers model retraining pipeline when KS drift > threshold

---

## Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html
```

| Test File | Tests | Coverage |
|-----------|-------|----------|
| `test_data_validation.py` | 18 | Data schema, nulls, fraud rate, duplicates |
| `test_preprocessing.py` | 16 | Imputation, freq encoding, target encoding, SMOTE |
| `test_model_training.py` | 17 | XGBoost, LightGBM, RF-Hybrid, SHAP, cost function |
| `test_api.py` | 26 | All endpoints, error handling, Prometheus metrics |
| **Total** | **77** | |

---

## Key Design Decisions

- **Temporal split** over random split to prevent data leakage (TransactionDT threshold)
- **Missing indicators** (`V{n}_missing` binary columns) preserve missingness signal for V1–V339
- **Smoothed target encoding** (k=10) for email domains prevents overfitting on rare domains
- **Frequency encoding** for high-cardinality card/address features (card1, card2, addr1, addr2)
- **SHAP subsampling** (1000 rows) prevents OOM inside KFP container components
- **hostPath PVs** for Minikube single-node cluster compatibility
- **Synthetic fallback** in data ingestion ensures pipeline runs without the Kaggle dataset

---

## Student Info

- **Name:** Ibrahim Ali
- **ID:** i221872
- **Course:** MLOps — BS Data Science
- **Deadline:** April 25, 2026
