FROM python:3.11-slim AS base

LABEL version="3.0" \
      description="IDS Healthcare CIP — Phase 0 + Phase 1 + Phase 2 pipeline" \
      maintainer="analyst"

# ── GPU / CPU fallback ────────────────────────────────────────────────
# Default to CPU; override with `docker run --gpus all -e CUDA_VISIBLE_DEVICES=0`
ENV CUDA_VISIBLE_DEVICES=""
ENV TF_DETERMINISTIC_OPS=1

# ── Non-root user (OWASP A01) ────────────────────────────────────────
RUN useradd -m -s /bin/bash analyst
WORKDIR /home/analyst/app

# ── Dependencies (Phase 0 + Phase 1) ────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Source + config + tests (no data, models, logs, results) ─────────
COPY src/ src/
COPY config/ config/
COPY tests/ tests/

# ── Writable output directories for pipeline execution ───────────────
RUN mkdir -p data/processed data/phase2 models/scalers results/phase0_analysis \
    && chown -R analyst:analyst /home/analyst/app

USER analyst

# ── Health check: verify Phase 0 + Phase 1 + Phase 2 imports ─────────
HEALTHCHECK --interval=30s --timeout=10s \
  CMD python -c "\
from src.phase0_dataset_analysis.phase0 import Phase0Config, DataLoader; \
from src.phase1_preprocessing.phase1 import Phase1Config, PreprocessingPipeline; \
from src.phase2_detection_engine.phase2 import Phase2Config, DetectionPipeline; \
import tensorflow as tf; \
print(f'OK — TF {tf.__version__}, GPU: {len(tf.config.list_physical_devices(\"GPU\"))}')"

ENTRYPOINT ["python", "-m"]
CMD ["pytest", \
     "src/phase0_dataset_analysis/phase0/tests/", \
     "src/phase1_preprocessing/phase1/tests/", \
     "src/phase2_detection_engine/phase2/tests/", \
     "-v"]
