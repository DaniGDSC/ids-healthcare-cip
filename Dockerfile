FROM python:3.11-slim AS base

LABEL version="2.0" \
      description="IDS Healthcare CIP — Phase 0 + Phase 1 pipeline" \
      maintainer="analyst"

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
RUN mkdir -p data/processed models/scalers results/phase0_analysis \
    && chown -R analyst:analyst /home/analyst/app

USER analyst

# ── Health check: verify Phase 0 + Phase 1 imports ───────────────────
HEALTHCHECK --interval=30s --timeout=5s \
  CMD python -c "\
from src.phase0_dataset_analysis.phase0 import Phase0Config, DataLoader; \
from src.phase1_preprocessing.phase1 import Phase1Config, PreprocessingPipeline; \
print('OK')"

ENTRYPOINT ["python", "-m"]
CMD ["pytest", \
     "src/phase0_dataset_analysis/phase0/tests/", \
     "src/phase1_preprocessing/phase1/tests/", \
     "-v"]
