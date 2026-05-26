"""Per-feature risk components: D_crit, S_data, D_clinical_tier.

These three components feed into the composite risk formula
``R = w1·C_detect + w2·D_crit + w3·S_data + w4·D_clinical_tier``.
The fourth component (``C_detect``) is produced by
``detection_engine.DetectionEngine`` — Module 3 only consumes it.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from .config import (
    BIOMETRIC_FEATURES,
    CIA_SCORE,
    DATA_SENSITIVITY,
    DEFAULT_CIA_SCORE,
    FEATURE_ACTIVE_EPSILON,
    SIGMA_THRESHOLD,
)


# ── Biometric-index cache (lru_cache, tuple key) ─────────────────────
@lru_cache(maxsize=4)
def _bio_idx_cached(feat_tuple: tuple) -> np.ndarray:
    """Cached lookup of biometric feature indices.

    Keyed by *tuple* of feat_names — value-based comparison so equal-but-
    distinct lists hit the same cache entry. Replaces the prior global
    mutable cache (identity-checked via ``is``) that didn't reset
    between tests.
    """
    return np.array(
        [feat_tuple.index(f) for f in BIOMETRIC_FEATURES if f in feat_tuple],
        dtype=np.intp,
    )


def _get_bio_idx(feat_names: list) -> np.ndarray:
    return _bio_idx_cached(tuple(feat_names))


# ── compute_d_crit ────────────────────────────────────────────────────
def compute_d_crit(attack_cats: np.ndarray) -> np.ndarray:
    """Device criticality from tier + CIA threat interaction.

    Vectorised Pandas map; pre-computed CIA_SCORE means no per-row
    ``max(C,I,A)`` calls.
    """
    scores = (
        pd.Series(attack_cats, dtype=str)
        .map(CIA_SCORE)
        .fillna(DEFAULT_CIA_SCORE)
        .values.astype(np.float64)
    )
    return np.clip(scores, 0.0, 1.0)


# ── compute_s_data (Y3: row-density refactor) ────────────────────────
def compute_s_data(X_test: np.ndarray, feat_names: list) -> np.ndarray:
    """Data sensitivity: weighted mix of PHI (biometric) vs telemetry features.

    Per-sample score combines:
      • ``bio_active`` — fraction of biometric features whose magnitude
        exceeds FEATURE_ACTIVE_EPSILON (PHI signal density)
      • ``net_active`` — analogous fraction for network features
        (device-telemetry signal density)

    Formula:
        s_data = (phi_w · bio_active + net_w · net_active) / (phi_w + net_w)
        phi_w  = DATA_SENSITIVITY['phi_realtime']     = 1.0
        net_w  = DATA_SENSITIVITY['device_telemetry'] = 0.4

    Y3 fix — previously ``net_active`` was hardcoded to ``ones(...)`` so
    rows with all-zero network features still scored at ``s_data ≥ 0.286``,
    a hidden floor. The fix replaces the constant with a row-density
    measurement symmetric to ``bio_active``. Downstream R values for
    sparse-network samples now drop by up to ``w3 × 0.286 ≈ 0.043``.
    """
    bio_idx = _get_bio_idx(feat_names)
    n_feats = len(feat_names)
    n_bio = len(bio_idx)
    n_net = n_feats - n_bio

    # Network indices = numeric features minus the biometric set.
    net_mask = np.ones(n_feats, dtype=bool)
    net_mask[bio_idx] = False
    net_idx = np.where(net_mask)[0]

    phi_weight = DATA_SENSITIVITY["phi_realtime"]
    net_weight = DATA_SENSITIVITY["device_telemetry"]

    # Fraction of biometric / network features that are non-zero
    if n_bio:
        bio_active = (np.abs(X_test[:, bio_idx]) > FEATURE_ACTIVE_EPSILON).sum(axis=1) / n_bio
    else:
        bio_active = np.zeros(len(X_test))
    if n_net:
        net_active = (np.abs(X_test[:, net_idx]) > FEATURE_ACTIVE_EPSILON).sum(axis=1) / n_net
    else:
        net_active = np.zeros(len(X_test))

    s_data = (phi_weight * bio_active + net_weight * net_active) / (phi_weight + net_weight)
    return np.clip(s_data, 0.0, 1.0)


# ── compute_d_clinical_tier ──────────────────────────────────────────
def compute_d_clinical_tier(X_test: np.ndarray, feat_names: list) -> np.ndarray:
    """Patient acuity: fraction of biometric features exceeding SIGMA_THRESHOLD.

    Uses the cached biometric-feature index lookup. Denominator is the
    full canonical biometric count so the metric stays comparable across
    runs even if a particular dataset drops some biometric channels.
    """
    bio_idx = _get_bio_idx(feat_names)
    bio_vals = X_test[:, bio_idx]
    abnormal_count = (np.abs(bio_vals) > SIGMA_THRESHOLD).sum(axis=1)
    return abnormal_count / len(BIOMETRIC_FEATURES)


__all__ = [
    "compute_d_crit",
    "compute_s_data",
    "compute_d_clinical_tier",
    "_get_bio_idx",
]
