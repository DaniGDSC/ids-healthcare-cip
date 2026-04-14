"""Model Registry — process-scoped singleton loader for all ML artefacts.

All pipeline modules (module3, module4_online, module4_explanations) previously
loaded the same 4 model objects independently via loads_signed() and
DAEDetector.from_artefacts().  In a single pipeline run this caused:
  - 3 × disk reads + deserializations of Track A classifiers
  - 3 × DAE JSON + Keras HDF5 weight loads

This module provides @lru_cache-backed loaders that deserialize each artefact
exactly once per process and reuse the cached object on every subsequent call.
No caller-side changes are required beyond swapping the load call to use these
functions.

Usage:
    from common.model_registry import get_track_a_classifiers, get_dae

    classifiers = get_track_a_classifiers()   # dict: name → fitted clf
    dae = get_dae()                            # DAEDetector instance
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Canonical model paths — single source of truth for all pipeline modules
_TRACK_A_PATHS = {
    "xgboost":       "results/models/xgboost_final_pipeline.pkl",
    "random_forest": "results/models/random_forest_final_pipeline.pkl",
    "decision_tree": "results/models/decision_tree_final_pipeline.pkl",
}
_DAE_JSON    = PROJECT_ROOT / "results/models/dae_detector.json"
_DAE_WEIGHTS = PROJECT_ROOT / "results/models/dae_model.weights.h5"


@lru_cache(maxsize=None)
def get_track_a_classifiers() -> dict:
    """Load all three Track A classifiers once and cache for the process lifetime.

    Returns:
        dict mapping model name → fitted sklearn classifier (bare, not Pipeline).
        The bare classifier is extracted when the artefact is a full sklearn
        Pipeline with a named 'classifier' step (legacy format); otherwise the
        loaded object is used directly.

    Raises:
        FileNotFoundError: if any model artefact is missing.
        RuntimeError: if loads_signed() rejects the signature.
    """
    from common import loads_signed

    classifiers = {}
    for name, rel_path in _TRACK_A_PATHS.items():
        path = PROJECT_ROOT / rel_path
        logger.info("ModelRegistry: loading %s from %s", name, path)
        obj = loads_signed(path)
        # Extract bare classifier if artefact is a full Pipeline
        classifiers[name] = (
            obj.named_steps["classifier"]
            if hasattr(obj, "named_steps") else obj
        )
    logger.info("ModelRegistry: Track A classifiers loaded (3 models)")
    return classifiers


@lru_cache(maxsize=None)
def get_dae():
    """Load DAE detector once and cache for the process lifetime.

    Returns:
        DAEDetector instance with weights loaded and threshold calibrated.

    Raises:
        FileNotFoundError: if JSON sidecar or weights file is missing.
    """
    from module2_detection.models.DAE import DAEDetector

    logger.info("ModelRegistry: loading DAE from %s", _DAE_JSON)
    dae = DAEDetector.from_artefacts(
        json_path=_DAE_JSON,
        weights_path=_DAE_WEIGHTS,
    )
    logger.info("ModelRegistry: DAE loaded (threshold=%.6f)", dae.threshold)
    return dae


@lru_cache(maxsize=None)
def get_track_a_thresholds() -> dict:
    """Load optimal thresholds for each Track A model once.

    Returns:
        dict mapping model name → float threshold.
    """
    import json

    _REPORT_PATHS = {
        "xgboost":       "results/models/xgboost_final_report.json",
        "random_forest": "results/models/random_forest_final_report.json",
        "decision_tree": "results/models/decision_tree_final_report.json",
    }
    thresholds = {}
    for name, rel_path in _REPORT_PATHS.items():
        path = PROJECT_ROOT / rel_path
        with open(path) as f:
            thresholds[name] = json.load(f)["optimal_threshold"]
    logger.info("ModelRegistry: Track A thresholds loaded")
    return thresholds


def invalidate_cache() -> None:
    """Clear all cached model objects (test utility — not for production use)."""
    get_track_a_classifiers.cache_clear()
    get_dae.cache_clear()
    get_track_a_thresholds.cache_clear()
    logger.info("ModelRegistry: cache cleared")
