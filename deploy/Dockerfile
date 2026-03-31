# IoMT IDS — Production container
# Multi-stage build: dependencies → app

FROM python:3.10-slim AS base

WORKDIR /app

# System dependencies for TensorFlow + scikit-learn
RUN apt-get update && apt-get install -y --no-install-recommends \
    libhdf5-dev gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY . .

# Create directories
RUN mkdir -p data/production data/audit logs

# Default: run API server
EXPOSE 8000
CMD ["uvicorn", "src.production.api:app", "--host", "0.0.0.0", "--port", "8000"]
