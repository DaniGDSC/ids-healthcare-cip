FROM python:3.11-slim AS base

LABEL org.opencontainers.image.source="https://github.com/example/ids-healthcare-cip"
LABEL org.opencontainers.image.description="Healthcare IoMT IDS — runtime image"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

RUN groupadd --system --gid 1000 ids \
 && useradd --system --uid 1000 --gid ids --create-home --shell /bin/bash ids

WORKDIR /app

COPY --chown=ids:ids requirements.txt requirements.lock* ./

# Install with --require-hashes when a lockfile is present; otherwise fall
# back to the unpinned requirements.txt (development convenience). CI is
# expected to ship the lockfile so the runtime image is reproducible.
RUN if [ -s requirements.lock ]; then \
        pip install --require-hashes -r requirements.lock ; \
    else \
        pip install -r requirements.txt ; \
    fi

COPY --chown=ids:ids common/ ./common/
COPY --chown=ids:ids module0_analysis/ ./module0_analysis/
COPY --chown=ids:ids module1_preprocessing/ ./module1_preprocessing/
COPY --chown=ids:ids module2_detection/ ./module2_detection/
COPY --chown=ids:ids module3_risk_scoring/ ./module3_risk_scoring/
COPY --chown=ids:ids module4_explanations/ ./module4_explanations/
COPY --chown=ids:ids module5_responses/ ./module5_responses/
COPY --chown=ids:ids module6_evaluation/ ./module6_evaluation/
COPY --chown=ids:ids detection_engine/ ./detection_engine/
COPY --chown=ids:ids src/ ./src/
COPY --chown=ids:ids configs/ ./configs/
COPY --chown=ids:ids config/ ./config/
COPY --chown=ids:ids run_all_modules.py pyproject.toml ./

USER ids

ENTRYPOINT ["python3"]
CMD ["-c", "from module0_analysis import Phase0Config, DataLoader, IntegrityVerifier, PathValidator; print('OK')"]
