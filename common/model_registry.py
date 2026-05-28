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

# Canonical model paths — single source of truth for all pipeline modules.
# Runtime path uses only XGBoost (decision driver). RandomForest and
# DecisionTree remain on disk as RQ1 R2 ablation baselines and are loaded
# lazily by get_baseline_classifiers() — never by the runtime engine, the
# online explainer, or any M4/M5/M6 module.
_RUNTIME_TRACK_A_PATHS = {
    "xgboost":       "results/models/xgboost_final_pipeline.pkl",
}
_BASELINE_TRACK_A_PATHS = {
    "random_forest": "results/models/random_forest_final_pipeline.pkl",
    "decision_tree": "results/models/decision_tree_final_pipeline.pkl",
}
_DAE_JSON    = PROJECT_ROOT / "results/models/dae_detector.json"
_DAE_WEIGHTS = PROJECT_ROOT / "results/models/dae_model.weights.h5"


def _load_classifiers(paths: dict[str, str]) -> dict:
    """Load + verify each classifier and stash the signed metadata.

    Tier 0 F5: when the signed sidecar carries a ``metadata.optimal_threshold``
    field we cache it in ``_SIGNED_THRESHOLDS`` so :func:`_load_thresholds`
    can read it without re-touching the (unsigned) report JSON.
    """
    from common.signed_pickle import loads_signed_with_metadata
    classifiers: dict = {}
    for name, rel_path in paths.items():
        path = PROJECT_ROOT / rel_path
        logger.info("ModelRegistry: loading %s from %s", name, path)
        obj, meta = loads_signed_with_metadata(path)
        # Extract bare classifier if artefact is a full Pipeline
        classifiers[name] = (
            obj.named_steps["classifier"]
            if hasattr(obj, "named_steps") else obj
        )
        if "optimal_threshold" in meta:
            t = float(meta["optimal_threshold"])
            if not (0.0 <= t <= 1.0):
                raise ValueError(
                    f"signed threshold metadata for {name} is out of "
                    f"range [0,1]: {t!r}. Refusing to load."
                )
            _SIGNED_THRESHOLDS[name] = t
            logger.info(
                "ModelRegistry: bound optimal_threshold=%.6f for %s from "
                "signed sidecar metadata",
                t, name,
            )
    return classifiers


# Process-scoped cache of signed-metadata thresholds. Populated by
# `_load_classifiers` whenever a sidecar carries `metadata.optimal_threshold`.
_SIGNED_THRESHOLDS: dict[str, float] = {}


@lru_cache(maxsize=None)
def get_track_a_classifiers() -> dict:
    """Load runtime Track A classifiers (XGBoost only) once per process.

    Returns:
        dict ``{"xgboost": fitted_classifier}``. The bare classifier is
        extracted when the artefact is a full sklearn Pipeline with a
        named 'classifier' step; otherwise the loaded object is used
        directly.

    Raises:
        FileNotFoundError: if the artefact is missing.
        RuntimeError: if loads_signed() rejects the signature.
    """
    classifiers = _load_classifiers(_RUNTIME_TRACK_A_PATHS)
    logger.info("ModelRegistry: Track A runtime classifiers loaded (%d)",
                len(classifiers))
    return classifiers


@lru_cache(maxsize=None)
def get_baseline_classifiers() -> dict:
    """Load RQ1 R2 baseline classifiers (RandomForest, DecisionTree).

    These are NOT consulted at runtime — the engine, online explainer,
    and all M4/M5/M6 builders use only :func:`get_track_a_classifiers`.
    Returned objects are intended for offline ablation tooling
    (``tools/rq1_compute_metrics.compute_track_a_ablation``).

    Returns:
        dict ``{"random_forest": clf, "decision_tree": clf}``.
    """
    classifiers = _load_classifiers(_BASELINE_TRACK_A_PATHS)
    logger.info("ModelRegistry: baseline classifiers loaded (%d)",
                len(classifiers))
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


_RUNTIME_THRESHOLD_PATHS = {
    "xgboost": "results/models/xgboost_final_report.json",
}
_BASELINE_THRESHOLD_PATHS = {
    "random_forest": "results/models/random_forest_final_report.json",
    "decision_tree": "results/models/decision_tree_final_report.json",
}


def _load_thresholds(paths: dict[str, str]) -> dict:
    """Resolve the per-model threshold.

    Resolution order (tier 0 F5):
      1. Signed sidecar metadata cached in ``_SIGNED_THRESHOLDS`` —
         populated by ``_load_classifiers`` when it deserialises the
         pickle. This is the integrity-bound path.
      2. Fallback: the unsigned ``<name>_final_report.json``. Logged
         as a WARNING because the value is not bound to the pickle
         signature.

    Either path bounds the loaded value to [0, 1]; out-of-range
    thresholds are refused.
    """
    import json
    thresholds: dict = {}
    for name, rel_path in paths.items():
        if name in _SIGNED_THRESHOLDS:
            thresholds[name] = _SIGNED_THRESHOLDS[name]
            continue
        path = PROJECT_ROOT / rel_path
        logger.warning(
            "ModelRegistry: %s has no signed-sidecar threshold; falling "
            "back to unsigned %s. Re-train the model so the threshold is "
            "embedded in the signed pickle.",
            name, path.name,
        )
        with open(path) as f:
            value = float(json.load(f)["optimal_threshold"])
        if not (0.0 <= value <= 1.0):
            raise ValueError(
                f"unsigned threshold for {name} is out of range [0,1]: {value!r}"
            )
        thresholds[name] = value
    return thresholds


@lru_cache(maxsize=None)
def get_track_a_thresholds() -> dict:
    """Load runtime Track A thresholds (XGBoost only).

    Returns:
        dict ``{"xgboost": optimal_threshold}``.
    """
    # Force-load classifiers first so signed-sidecar metadata is in
    # `_SIGNED_THRESHOLDS` before _load_thresholds runs (tier 0 F5).
    # @lru_cache makes this a no-op once primed.
    get_track_a_classifiers()
    thresholds = _load_thresholds(_RUNTIME_THRESHOLD_PATHS)
    logger.info("ModelRegistry: Track A runtime thresholds loaded (%d)",
                len(thresholds))
    return thresholds


@lru_cache(maxsize=None)
def get_baseline_thresholds() -> dict:
    """Load RQ1 R2 baseline thresholds (RandomForest, DecisionTree)."""
    get_baseline_classifiers()
    thresholds = _load_thresholds(_BASELINE_THRESHOLD_PATHS)
    logger.info("ModelRegistry: baseline thresholds loaded (%d)",
                len(thresholds))
    return thresholds


def invalidate_cache() -> None:
    """Clear all cached model objects (test utility — not for production use)."""
    get_track_a_classifiers.cache_clear()
    get_baseline_classifiers.cache_clear()
    get_dae.cache_clear()
    get_track_a_thresholds.cache_clear()
    get_baseline_thresholds.cache_clear()
    logger.info("ModelRegistry: cache cleared")
