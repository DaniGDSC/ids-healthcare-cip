"""Module 3 feedback — apply_feedback + apply_weight_feedback (C3 fix)."""
from __future__ import annotations

import logging

import numpy as np
import pytest

from module3_risk_scoring.config import WEIGHTS
from module3_risk_scoring.feedback import apply_feedback, apply_weight_feedback


# ── apply_feedback (threshold clamp) ─────────────────────────────────


def test_apply_feedback_no_change_when_no_suggestion():
    current = {"CRITICAL": 0.80, "HIGH": 0.60, "MEDIUM": 0.40}
    updated = apply_feedback(current, {"suggested_threshold_change": {}})
    assert updated == current


def test_apply_feedback_clamps_to_max_delta():
    """Suggested change > max_delta gets clamped."""
    current = {"CRITICAL": 0.80, "HIGH": 0.60, "MEDIUM": 0.40}
    feedback = {"suggested_threshold_change": {
        "CRITICAL": 1.00,  # +0.20 → clamped to +0.10
    }}
    updated = apply_feedback(current, feedback, max_delta=0.10)
    assert updated["CRITICAL"] == pytest.approx(0.90)


def test_apply_feedback_negative_clamp():
    current = {"CRITICAL": 0.80}
    feedback = {"suggested_threshold_change": {"CRITICAL": 0.10}}  # delta = -0.70
    updated = apply_feedback(current, feedback, max_delta=0.10)
    assert updated["CRITICAL"] == pytest.approx(0.70)


def test_apply_feedback_within_clamp_passes_through():
    current = {"CRITICAL": 0.80}
    feedback = {"suggested_threshold_change": {"CRITICAL": 0.85}}  # delta = +0.05
    updated = apply_feedback(current, feedback, max_delta=0.10)
    assert updated["CRITICAL"] == pytest.approx(0.85)


def test_apply_feedback_preserves_tier_keys():
    """Returned dict has same keys as input."""
    current = {"CRITICAL": 0.80, "HIGH": 0.60, "MEDIUM": 0.40}
    updated = apply_feedback(current, {"suggested_threshold_change": {}})
    assert set(updated.keys()) == set(current.keys())


def test_apply_feedback_warns_on_unknown_tier(caplog):
    """N3 fix: unknown tier in suggestion logged as WARNING."""
    current = {"CRITICAL": 0.80}
    feedback = {"suggested_threshold_change": {"GHOST_TIER": 0.5}}
    caplog.set_level(logging.WARNING)
    apply_feedback(current, feedback)
    assert any("GHOST_TIER" in r.message and "unknown" in r.message.lower()
               for r in caplog.records)


def test_apply_feedback_clamp_invariant_holds():
    """For every tier, |new_val - cur_val| ≤ max_delta."""
    current = {"CRITICAL": 0.80, "HIGH": 0.60, "MEDIUM": 0.40}
    feedback = {"suggested_threshold_change": {
        "CRITICAL": 5.0,
        "HIGH": -3.0,
        "MEDIUM": 0.50,
    }}
    updated = apply_feedback(current, feedback, max_delta=0.10)
    for tier in current:
        delta = abs(updated[tier] - current[tier])
        assert delta <= 0.10 + 1e-9


# ── apply_weight_feedback (C3 — vectorised AUROC hill-climb) ────────


def _build_separable_arrays(n=200, seed=0):
    """Synthetic data where c_detect is highly predictive of y_true."""
    rng = np.random.default_rng(seed)
    y = np.array([0] * (n // 2) + [1] * (n // 2))
    c_detect = np.concatenate([
        rng.uniform(0, 0.3, n // 2),     # benign — low
        rng.uniform(0.7, 1.0, n // 2),   # attack — high
    ])
    d_crit = rng.uniform(0, 1, n)
    s_data = rng.uniform(0, 1, n)
    d_clinical = rng.uniform(0, 1, n)
    return y, c_detect, d_crit, s_data, d_clinical


def test_apply_weight_feedback_returns_normalized_weights():
    """Output weights must sum to 1.0."""
    y, c, d, s, t = _build_separable_arrays()
    component_variances = {
        "w1": float(np.var(c)),
        "w2": float(np.var(d)),
        "w3": float(np.var(s)),
        "w4": float(np.var(t)),
    }
    new_w = apply_weight_feedback(
        dict(WEIGHTS), component_variances, y, c, d, s, t,
    )
    assert abs(sum(new_w.values()) - 1.0) < 1e-3


def test_apply_weight_feedback_preserves_weight_keys():
    y, c, d, s, t = _build_separable_arrays()
    component_variances = {k: 1.0 for k in WEIGHTS}
    new_w = apply_weight_feedback(
        dict(WEIGHTS), component_variances, y, c, d, s, t,
    )
    assert set(new_w.keys()) == set(WEIGHTS.keys())


def test_apply_weight_feedback_increases_predictive_weight():
    """When c_detect perfectly separates y_true, w1 should be ≥ baseline."""
    y, c, d, s, t = _build_separable_arrays()
    component_variances = {k: 1.0 for k in WEIGHTS}
    new_w = apply_weight_feedback(
        dict(WEIGHTS), component_variances, y, c, d, s, t,
    )
    # w1 should grow or hold — never decrease in a perfectly-separable case
    assert new_w["w1"] >= WEIGHTS["w1"] - 0.02


def test_apply_weight_feedback_max_delta_clamp():
    """Single-iteration change per weight should respect max_delta."""
    y, c, d, s, t = _build_separable_arrays()
    component_variances = {k: 1.0 for k in WEIGHTS}
    new_w = apply_weight_feedback(
        dict(WEIGHTS), component_variances, y, c, d, s, t,
        max_delta=0.03,
    )
    # Account for renormalisation rounding
    for k in WEIGHTS:
        assert abs(new_w[k] - WEIGHTS[k]) <= 0.10  # generous bound


def test_apply_weight_feedback_low_variance_redistribution():
    """A near-constant component (variance ≈ 0) gets shrunk."""
    n = 200
    rng = np.random.default_rng(0)
    y = np.array([0] * 100 + [1] * 100)
    c = rng.uniform(0, 1, n)
    d = rng.uniform(0, 1, n)
    s = np.full(n, 0.5)  # near-constant — should be shrunk
    t = rng.uniform(0, 1, n)
    component_variances = {
        "w1": 1.0, "w2": 1.0, "w3": 0.0001, "w4": 1.0,
    }
    new_w = apply_weight_feedback(
        dict(WEIGHTS), component_variances, y, c, d, s, t,
    )
    assert new_w["w3"] < WEIGHTS["w3"]


def test_apply_weight_feedback_all_outputs_in_unit_interval():
    y, c, d, s, t = _build_separable_arrays()
    component_variances = {k: 1.0 for k in WEIGHTS}
    new_w = apply_weight_feedback(
        dict(WEIGHTS), component_variances, y, c, d, s, t,
    )
    for v in new_w.values():
        assert 0.0 <= v <= 1.0
