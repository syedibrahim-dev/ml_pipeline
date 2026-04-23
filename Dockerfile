# ---- Build stage ----
FROM python:3.10-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user \
    fastapi==0.115.0 \
    uvicorn==0.32.1 \
    prometheus-client==0.21.0 \
    scikit-learn==1.6.1 \
    xgboost==2.1.3 \
    lightgbm==4.5.0 \
    pandas==2.2.3 \
    numpy==1.26.4 \
    scipy==1.14.1 \
    joblib==1.4.2 \
    pydantic==2.9.2

# ---- Runtime stage ----
FROM python:3.10-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Install libgomp for LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy application code
COPY api/     ./api/
COPY results/ ./results/

# Model registry will be mounted as a volume
RUN mkdir -p /tmp/fraud_model_registry/champion

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
