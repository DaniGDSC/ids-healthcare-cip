FROM python:3.11-slim AS base

LABEL version="6.0" \
      description="IDS Healthcare CIP — Phase 0 + Phase 1 + Phase 2 + Phase 3 + Phase 4 + Phase 5 pipeline" \
      maintainer="analyst"

# ── GPU / CPU fallback ────────────────────────────────────────────────
# Default to CPU; override with `docker run --gpus all -e CUDA_VISIBLE_DEVICES=0`
ENV CUDA_VISIBLE_DEVICES=""
ENV TF_DETERMINISTIC_OPS=1
ENV MPLBACKEND=Agg

# ── Non-root user (OWASP A01) ────────────────────────────────────────
RUN useradd -m -s /bin/bash analyst
WORKDIR /home/analyst/app

# ── Dependencies (Phase 0–5: includes shap, matplotlib, plotly) ──────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Source + config + tests (no data, models, logs, results) ─────────
COPY src/ src/
COPY config/ config/
COPY tests/ tests/

# ── Writable output directories for pipeline execution ───────────────
RUN mkdir -p data/processed data/phase2 data/phase3 data/phase4 \
             data/phase5 data/phase5/charts \
             models/scalers results/phase0_analysis \
    && chown -R analyst:analyst /home/analyst/app

USER analyst

# ── Health check: verify Phase 0–5 imports + shap importable ─────────
HEALTHCHECK --interval=30s --timeout=10s \
  CMD python -c "\
from src.phase0_dataset_analysis.phase0 import Phase0Config, DataLoader; \
from src.phase1_preprocessing.phase1 import Phase1Config, PreprocessingPipeline; \
from src.phase2_detection_engine.phase2 import Phase2Config, DetectionPipeline; \
from src.phase3_classification_engine.phase3 import Phase3Config, ClassificationPipeline; \
from src.phase4_risk_engine.phase4 import Phase4Config, RiskAdaptivePipeline; \
from src.phase5_explanation_engine.phase5.config import Phase5Config; \
import shap; import tensorflow as tf; import json, pathlib; \
m = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=(5,))]); \
m.save_weights('/tmp/_healthcheck.weights.h5'); \
pathlib.Path('/tmp/_baseline.json').write_text(json.dumps({'threshold': 0.2})); \
print(f'OK — TF {tf.__version__}, shap {shap.__version__}, GPU: {len(tf.config.list_physical_devices(\"GPU\"))}')"

ENTRYPOINT ["python", "-m"]
CMD ["pytest", \
     "src/phase0_dataset_analysis/phase0/tests/", \
     "src/phase1_preprocessing/phase1/tests/", \
     "src/phase2_detection_engine/phase2/tests/", \
     "src/phase3_classification_engine/phase3/tests/", \
     "src/phase4_risk_engine/phase4/tests/", \
     "src/phase5_explanation_engine/phase5/tests/", \
     "-v"]
