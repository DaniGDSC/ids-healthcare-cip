"""Module 3 components — D_crit / S_data / D_clinical_tier.

Includes Y3 verification: s_data should hit 0 when both bio and net
features are all-zero (was previously floor at 0.286).
"""
from __future__ import annotations

import numpy as np
import pytest

from module3_risk_scoring.components import (
    _get_bio_idx,
    compute_d_clinical_tier,
    compute_d_crit,
    compute_s_data,
)
from module3_risk_scoring.config import (
    BIOMETRIC_FEATURES,
    CIA_SCORE,
    DEFAULT_CIA_SCORE,
    DEFAULT_DEVICE_TIER,
    DEVICE_TIERS,
    SIGMA_THRESHOLD,
)


# ── compute_d_crit ────────────────────────────────────────────────────


def test_compute_d_crit_known_category():
    """Spoofing has max(C, I, A) = 0.9; D_crit = 0.8 * 0.9 = 0.72."""
    cats = np.array(["Spoofing", "Spoofing"])
    out = compute_d_crit(cats)
    expected = DEVICE_TIERS[DEFAULT_DEVICE_TIER] * 0.9
    np.testing.assert_array_almost_equal(out, [expected, expected])


def test_compute_d_crit_data_alteration():
    """Data Alteration has max(C, I, A) = 1.0; D_crit = 0.8 * 1.0 = 0.80."""
    cats = np.array(["Data Alteration"])
    out = compute_d_crit(cats)
    expected = DEVICE_TIERS[DEFAULT_DEVICE_TIER] * 1.0
    np.testing.assert_array_almost_equal(out, [expected])


def test_compute_d_crit_unknown_category_falls_back():
    """Unknown category maps to DEFAULT_CIA_SCORE."""
    cats = np.array(["RandomNewAttack"])
    out = compute_d_crit(cats)
    np.testing.assert_array_almost_equal(out, [DEFAULT_CIA_SCORE])


def test_compute_d_crit_mixed_categories():
    cats = np.array(["Spoofing", "Data Alteration", "normal"])
    out = compute_d_crit(cats)
    assert out[0] == pytest.approx(CIA_SCORE["Spoofing"])
    assert out[1] == pytest.approx(CIA_SCORE["Data Alteration"])
    assert out[2] == pytest.approx(DEFAULT_CIA_SCORE)


def test_compute_d_crit_clipped_to_unit_interval():
    """Output always in [0, 1] regardless of categories."""
    cats = np.array(["X"] * 100)
    out = compute_d_crit(cats)
    assert (out >= 0).all() and (out <= 1).all()


# ── compute_s_data (Y3 verification) ──────────────────────────────────


def test_compute_s_data_zero_when_all_features_inactive():
    """Y3 fix: s_data must reach 0 when no features are active (previously
    floor at 0.286 because net_present was hardcoded to 1)."""
    feat_names = BIOMETRIC_FEATURES + ["net_feat_1", "net_feat_2"]
    X = np.zeros((1, len(feat_names)), dtype=np.float32)
    out = compute_s_data(X, feat_names)
    assert out[0] == pytest.approx(0.0, abs=1e-9)


def test_compute_s_data_max_when_all_features_active():
    feat_names = BIOMETRIC_FEATURES + ["net_feat_1", "net_feat_2"]
    X = np.ones((1, len(feat_names)), dtype=np.float32)
    out = compute_s_data(X, feat_names)
    # Both bio_active and net_active = 1 → s_data = (1.0 + 0.4) / 1.4 = 1.0
    assert out[0] == pytest.approx(1.0, abs=1e-9)


def test_compute_s_data_only_biometric_active():
    """bio_active=1, net_active=0 → s_data = 1.0 / 1.4 ≈ 0.714."""
    feat_names = BIOMETRIC_FEATURES + ["net_feat_1", "net_feat_2"]
    X = np.zeros((1, len(feat_names)), dtype=np.float32)
    bio_count = len(BIOMETRIC_FEATURES)
    X[0, :bio_count] = 1.0  # only biometric features active
    out = compute_s_data(X, feat_names)
    assert out[0] == pytest.approx(1.0 / 1.4, abs=1e-9)


def test_compute_s_data_only_network_active():
    """bio_active=0, net_active=1 → s_data = 0.4 / 1.4 ≈ 0.286."""
    feat_names = BIOMETRIC_FEATURES + ["net_feat_1", "net_feat_2"]
    X = np.zeros((1, len(feat_names)), dtype=np.float32)
    bio_count = len(BIOMETRIC_FEATURES)
    X[0, bio_count:] = 1.0  # only network features active
    out = compute_s_data(X, feat_names)
    assert out[0] == pytest.approx(0.4 / 1.4, abs=1e-9)


def test_compute_s_data_output_in_unit_interval():
    feat_names = BIOMETRIC_FEATURES + ["net_feat_1", "net_feat_2"]
    rng = np.random.default_rng(0)
    X = rng.uniform(-2, 2, (100, len(feat_names))).astype(np.float32)
    out = compute_s_data(X, feat_names)
    assert (out >= 0).all() and (out <= 1).all()


# ── compute_d_clinical_tier ──────────────────────────────────────────


def test_clinical_tier_zero_when_all_biometric_in_range():
    """Values within ±SIGMA_THRESHOLD produce zero abnormal count."""
    feat_names = BIOMETRIC_FEATURES + ["net_feat"]
    X = np.zeros((1, len(feat_names)), dtype=np.float32)
    out = compute_d_clinical_tier(X, feat_names)
    assert out[0] == 0.0


def test_clinical_tier_high_when_all_biometric_abnormal():
    """All biometric features at +3σ → fraction = 1.0."""
    feat_names = BIOMETRIC_FEATURES + ["net_feat"]
    X = np.zeros((1, len(feat_names)), dtype=np.float32)
    bio_count = len(BIOMETRIC_FEATURES)
    X[0, :bio_count] = 3.0  # all biometric features beyond SIGMA_THRESHOLD
    out = compute_d_clinical_tier(X, feat_names)
    assert out[0] == pytest.approx(1.0)


def test_clinical_tier_proportional_to_abnormal_count():
    """Half of biometric features abnormal → fraction = 0.5."""
    feat_names = BIOMETRIC_FEATURES + ["net_feat"]
    X = np.zeros((1, len(feat_names)), dtype=np.float32)
    bio_count = len(BIOMETRIC_FEATURES)
    # Set first half of biometric features beyond threshold
    half = bio_count // 2
    X[0, :half] = 3.0
    out = compute_d_clinical_tier(X, feat_names)
    expected = half / bio_count
    assert out[0] == pytest.approx(expected)


def test_clinical_tier_negative_values_count_as_abnormal():
    """|x| > SIGMA_THRESHOLD applies to both signs."""
    feat_names = BIOMETRIC_FEATURES + ["net_feat"]
    X = np.zeros((1, len(feat_names)), dtype=np.float32)
    bio_count = len(BIOMETRIC_FEATURES)
    X[0, :bio_count] = -(SIGMA_THRESHOLD + 0.5)
    out = compute_d_clinical_tier(X, feat_names)
    assert out[0] == pytest.approx(1.0)


# ── _get_bio_idx caching ─────────────────────────────────────────────


def test_get_bio_idx_returns_consistent_indices():
    """Same feat_names tuple → identical cached output."""
    feat_names = BIOMETRIC_FEATURES + ["a", "b"]
    idx1 = _get_bio_idx(feat_names)
    idx2 = _get_bio_idx(feat_names)
    np.testing.assert_array_equal(idx1, idx2)


def test_get_bio_idx_value_based_cache():
    """Equal-but-distinct lists hit the same cache entry via tuple key."""
    feat_names_a = list(BIOMETRIC_FEATURES) + ["a", "b"]
    feat_names_b = list(BIOMETRIC_FEATURES) + ["a", "b"]
    idx_a = _get_bio_idx(feat_names_a)
    idx_b = _get_bio_idx(feat_names_b)
    np.testing.assert_array_equal(idx_a, idx_b)


def test_get_bio_idx_handles_missing_biometric():
    """If some biometric features missing from feat_names, only present ones indexed."""
    feat_names = list(BIOMETRIC_FEATURES[:2]) + ["net_a", "net_b"]
    idx = _get_bio_idx(feat_names)
    assert len(idx) == 2
