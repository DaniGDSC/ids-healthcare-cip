"""SHAP + DAE compute layer.

Single canonical implementation of:
  - ``compute_tree_shap`` for Track A models
  - ``compute_dae_feature_errors`` for Track B (cascaded)
  - ``compute_global_importance`` (mean |SHAP|)
  - ``_top_features_shap`` / ``_top_features_dae`` (per-sample top-k)

Previously these lived in 2 separate files with subtle divergence; this
module is the only source after the cleanup.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


# ── Shape-normalisation helper ───────────────────────────────────────


def _normalise_shap_output(sv) -> np.ndarray:
    """Coerce TreeSHAP output to a 2-D ``(n_samples, n_features)`` array
    representing the attack class.

    Handles three observed return shapes:
      - ``list`` of two ndarrays → take index 1 (attack class)
      - 3-D array of shape ``(n, features, 2)`` → slice ``[:, :, 1]``
      - 2-D array → returned as-is
    """
    if isinstance(sv, list):
        return sv[1]
    if isinstance(sv, np.ndarray) and sv.ndim == 3:
        return sv[:, :, 1]
    return sv


def _normalise_expected_value(expected) -> float:
    """Coerce TreeExplainer ``expected_value`` to a scalar."""
    if isinstance(expected, (list, np.ndarray)):
        expected = np.atleast_1d(expected)
        return float(expected[1]) if len(expected) > 1 else float(expected[0])
    return float(expected)


# ── TreeSHAP ─────────────────────────────────────────────────────────


def compute_tree_shap(
    model_name: str,
    pipeline_path: Path,
    X_test: np.ndarray,
    feat_names: list,
) -> tuple[np.ndarray, float]:
    """Compute TreeSHAP values for a Track A model.

    Args:
        model_name: model identifier (logged for context).
        pipeline_path: ECDSA-signed pickle path.
        X_test: ``(n_samples, n_features)`` feature matrix.
        feat_names: ordered feature names. Must match ``X_test.shape[1]``
            — otherwise SHAP values would be silently mis-attributed.

    Returns:
        ``(shap_values, expected_value)`` — SHAP values normalised to
        2-D ``(n, n_features)`` for the attack class; expected value as
        a scalar.

    Raises:
        ValueError: if ``feat_names`` length doesn't match ``X_test`` width.
    """
    if len(feat_names) != X_test.shape[1]:
        raise ValueError(
            f"compute_tree_shap: feat_names has {len(feat_names)} entries "
            f"but X_test has {X_test.shape[1]} columns. Refusing to run — "
            f"SHAP values would be silently mis-attributed to wrong feature names."
        )

    logger.info("Computing TreeSHAP for %s...", model_name)
    t0 = time.perf_counter()

    # Lazy import: shap pulls in a lot of optional deps.
    import shap

    # Signed-pickle load: refuses tampered/unsigned files. Phase 2's
    # final-training writes the bare classifier (not a full Pipeline).
    from common import loads_signed

    obj = loads_signed(pipeline_path)
    clf = obj.named_steps["classifier"] if hasattr(obj, "named_steps") else obj

    explainer = shap.TreeExplainer(clf)
    sv = explainer.shap_values(X_test)
    sv = _normalise_shap_output(sv)
    expected = _normalise_expected_value(explainer.expected_value)

    elapsed = time.perf_counter() - t0
    logger.info(
        "  %s TreeSHAP done: shape=%s, expected=%.4f (%.1fs)",
        model_name, sv.shape, expected, elapsed,
    )
    return sv, expected


def compute_global_importance(sv: np.ndarray, feat_names: list) -> list:
    """Compute ranked global feature importance from SHAP values."""
    if len(feat_names) != sv.shape[1]:
        raise ValueError(
            f"compute_global_importance: feat_names has {len(feat_names)} "
            f"entries but SHAP values have {sv.shape[1]} columns."
        )
    mean_abs = np.mean(np.abs(sv), axis=0)
    ranked = sorted(zip(feat_names, mean_abs), key=lambda x: -x[1])
    return [
        {"rank": i + 1, "feature": name, "mean_abs_shap": round(float(val), 6)}
        for i, (name, val) in enumerate(ranked)
    ]


# ── DAE per-feature error ───────────────────────────────────────────


def compute_dae_feature_errors(
    X_test: np.ndarray,
    feat_names: list,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decompose DAE reconstruction error into per-feature contributions.

    Uses the public ``DetectionEngine.get_dae()`` accessor (Y2 fix) —
    the previous version of this function reached into ``engine._dae``,
    which was private API.

    Returns ``(sq_err, weighted_err, feat_weights)`` sliced to the raw
    feature width so downstream code keyed on the raw ``feat_names``
    list stays consistent. The proba columns the DAE consumes are kept
    inside the engine and not exposed here.
    """
    logger.info("Computing DAE per-feature error decomposition...")
    from detection_engine import DetectionEngine

    engine = DetectionEngine()
    X_aug = engine.build_augmented(X_test)
    det = engine.get_dae()   # Y2 fix — public accessor

    X_norm = det._normalise(X_aug)
    recon = det._forward(X_norm)
    sq_err_full = (X_norm - recon) ** 2
    weighted_err_full = sq_err_full * det._feat_weights

    n_raw = X_test.shape[1]
    sq_err = sq_err_full[:, :n_raw]
    weighted_err = weighted_err_full[:, :n_raw]
    feat_weights = det._feat_weights[:n_raw]

    logger.info(
        "  DAE decomposition done: shape=%s (sliced from %s)",
        weighted_err.shape, weighted_err_full.shape,
    )
    return sq_err, weighted_err, feat_weights


# ── Top-k helpers (single canonical implementation) ─────────────────


def _top_features_shap(sv_row: np.ndarray, feat_names: list, k: int = 3) -> list:
    """Top-k features by ``|SHAP|`` for one sample.

    Uses ``np.argpartition`` (O(F)) to find candidate indices, then
    sorts only the k of them — faster than full ``argsort``. ``k`` is
    clamped to ``len(sv_row)`` so callers requesting more features than
    exist (e.g. a stakeholder test fixture with 2 features asking for
    top-5) get whatever is available instead of crashing.
    """
    abs_vals = np.abs(sv_row)
    k = min(k, len(abs_vals))
    if k <= 0:
        return []
    top_i_unsorted = np.argpartition(abs_vals, -k)[-k:]
    top_i = top_i_unsorted[np.argsort(abs_vals[top_i_unsorted])[::-1]]
    return [
        {
            "feature": feat_names[i],
            "shap_value": round(float(sv_row[i]), 6),
            "direction": "increases_risk" if sv_row[i] > 0 else "decreases_risk",
        }
        for i in top_i
    ]


def _top_features_dae(werr_row: np.ndarray, feat_names: list, k: int = 3) -> list:
    """Top-k features by weighted error for one DAE sample. ``k`` clamped
    to array length (see ``_top_features_shap`` for rationale).
    """
    total = werr_row.sum()
    k = min(k, len(werr_row))
    if k <= 0:
        return []
    top_i_unsorted = np.argpartition(werr_row, -k)[-k:]
    top_i = top_i_unsorted[np.argsort(werr_row[top_i_unsorted])[::-1]]
    return [
        {
            "feature": feat_names[i],
            "weighted_error": round(float(werr_row[i]), 8),
            "pct_contribution": (
                round(float(werr_row[i] / total * 100), 1) if total > 0 else 0.0
            ),
        }
        for i in top_i
    ]


__all__ = [
    "compute_tree_shap",
    "compute_global_importance",
    "compute_dae_feature_errors",
    "_top_features_shap",
    "_top_features_dae",
    "_normalise_shap_output",
    "_normalise_expected_value",
]
