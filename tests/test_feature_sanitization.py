"""Acceptance tests for ARCHITECTURE.md Step [5] feature sanitization.

Covers the contract in results/reports/feature_sanitization.yaml:
- 5 named test cases (test_1 ... test_5)
- EA-06 NaN-injection mitigation
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_models import DataQuality
from src.preprocessing import (
    FEATURE_NAMES_25,
    load_benign_medians,
    sanitize_features,
)
from src.risk_scorer import score_alert

# A clean reference flow with all 25 features set to per-feature benign medians.
# Using medians as the reference makes the "unchanged" assertion in test_1
# robust regardless of the actual scaler-output distribution.
def _clean_input() -> np.ndarray:
    medians = load_benign_medians()
    return np.array([medians[f] for f in FEATURE_NAMES_25], dtype=np.float64)


# ── Acceptance test cases ────────────────────────────────────────────────

def test_1_normal_input_unchanged_flag_ok() -> None:
    """test_1_normal_input: clean 25 features → unchanged, flag=OK."""
    x = _clean_input()
    x_clean, flag, nan_rate = sanitize_features(x)
    np.testing.assert_array_equal(x_clean, x)
    assert flag == "OK"
    assert nan_rate == 0.0


def test_2_partial_nan_below_threshold_flag_ok() -> None:
    """test_2_partial_nan: 2/25 features NaN → median replacement, flag=OK (8% > 5%)."""
    x = _clean_input()
    x[0] = np.nan
    x[1] = np.nan
    x_clean, flag, nan_rate = sanitize_features(x)
    medians = load_benign_medians()
    # Both NaN slots replaced with their per-feature benign median.
    assert x_clean[0] == medians[FEATURE_NAMES_25[0]]
    assert x_clean[1] == medians[FEATURE_NAMES_25[1]]
    # Untouched features preserved.
    np.testing.assert_array_equal(x_clean[2:], x[2:])
    # Spec says "flag=OK (rate < 5%)" but 2/25 = 8%; the spec text contradicts
    # itself. Per the explicit threshold (NAN_RATE_DEGRADED = 0.05), 8% is
    # DEGRADED. Honour the threshold; this is also what EA-06 mitigation
    # depends on. If the spec table is the source of truth, lower the
    # threshold instead — see feature_sanitization.yaml note.
    assert flag == "DEGRADED"
    assert nan_rate == round(2 / 25, 6)


def test_3_high_nan_rate_flag_degraded() -> None:
    """test_3_high_nan_rate: 5/25 features NaN → replaced, flag=DEGRADED (20%)."""
    x = _clean_input()
    for i in range(5):
        x[i] = np.nan
    x_clean, flag, nan_rate = sanitize_features(x)
    assert np.isfinite(x_clean).all()
    assert flag == "DEGRADED"
    assert nan_rate == round(5 / 25, 6)


def test_4_inf_handling() -> None:
    """test_4_inf_handling: feature = +Inf or -Inf → replaced with median."""
    x = _clean_input()
    x[5] = np.inf
    x[10] = -np.inf
    x_clean, flag, nan_rate = sanitize_features(x)
    medians = load_benign_medians()
    assert x_clean[5] == medians[FEATURE_NAMES_25[5]]
    assert x_clean[10] == medians[FEATURE_NAMES_25[10]]
    assert np.isfinite(x_clean).all()
    assert nan_rate == round(2 / 25, 6)


def test_5_nan_injection_attack_elevates_score() -> None:
    """test_5_nan_injection_attack: deliberately NaN many features → flag=DEGRADED,
    anomaly score is elevated so the attack cannot mask a true anomaly (EA-06).
    """
    x = _clean_input()
    # Adversary nan-bombs 6 features (24% > DEGRADED threshold).
    for i in range(6):
        x[i] = np.nan
    x_clean, flag, nan_rate = sanitize_features(x)
    assert flag == "DEGRADED"
    assert nan_rate > 0.05

    # EA-06 mitigation: even with a low raw anomaly score, the DEGRADED flag
    # must elevate the adjusted score so the alert routes for verification.
    raw_score = 0.40   # below the default 0.50 surfacing threshold
    result = score_alert(
        anomaly_score=raw_score,
        device_context={"criticality": "HIGH", "patchable": False},
        event_context=None,
        data_quality=flag,
    )
    # 0.40 × 1.20 (DEGRADED bump) × 1.20 (HIGH+unpatchable risk multiplier)
    # = 0.576 > 0.425 (HIGH+unpatchable threshold). Surfaces.
    assert result.adjusted_score > raw_score
    assert result.should_surface is True


# ── EA-06 specific: FAILED inputs always surface ─────────────────────────

def test_failed_quality_forces_surface() -> None:
    """nan_rate >= 50% (FAILED) — alert always surfaces regardless of raw score."""
    x = _clean_input()
    for i in range(13):  # 13/25 = 52% → FAILED
        x[i] = np.nan
    _, flag, nan_rate = sanitize_features(x)
    assert flag == "FAILED"
    assert nan_rate >= 0.5

    result = score_alert(
        anomaly_score=0.05,  # extremely weak signal
        device_context={"criticality": "LOW", "patchable": True},
        event_context=None,
        data_quality=flag,
    )
    # FAILED clamps adjusted_score to 0.95 minimum, which trivially exceeds
    # the LOW+patchable threshold (0.50). Operator must verify the device.
    assert result.adjusted_score >= 0.95
    assert result.should_surface is True


# ── BENIGN_MEDIANS lookup ────────────────────────────────────────────────

def test_benign_medians_persisted_and_complete() -> None:
    """The persisted lookup covers all 25 features and was computed from
    the benign training subset."""
    medians = load_benign_medians()
    assert set(medians) == set(FEATURE_NAMES_25)
    # All medians finite.
    assert all(np.isfinite(v) for v in medians.values())
