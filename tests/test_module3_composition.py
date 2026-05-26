"""Module 3 composition — compute_composite_risk + assign_risk_levels."""
from __future__ import annotations

import numpy as np
import pytest

from module3_risk_scoring.composition import (
    assign_risk_levels,
    compute_composite_risk,
)
from module3_risk_scoring.config import RISK_THRESHOLDS, WEIGHTS


# ── compute_composite_risk ───────────────────────────────────────────


def test_composite_formula_matches_weighted_sum():
    """R = w1·c_detect + w2·d_crit + w3·s_data + w4·d_clinical_tier."""
    c = np.array([0.5])
    d = np.array([0.6])
    s = np.array([0.7])
    t = np.array([0.8])
    R = compute_composite_risk(c, d, s, t)
    w = WEIGHTS
    expected = (
        w["w1"] * 0.5 + w["w2"] * 0.6 + w["w3"] * 0.7 + w["w4"] * 0.8
    )
    assert R[0] == pytest.approx(expected)


def test_composite_clipped_to_unit_interval():
    """Even with malformed inputs, output ∈ [0, 1]."""
    c = np.array([1.5, -0.5])
    d = np.array([1.5, -0.5])
    s = np.array([1.5, -0.5])
    t = np.array([1.5, -0.5])
    R = compute_composite_risk(c, d, s, t)
    assert (R >= 0).all() and (R <= 1).all()


def test_composite_custom_weights():
    """Override WEIGHTS via parameter."""
    c = np.array([1.0])
    d = np.array([0.0])
    s = np.array([0.0])
    t = np.array([0.0])
    R = compute_composite_risk(c, d, s, t, weights={"w1": 1.0, "w2": 0.0, "w3": 0.0, "w4": 0.0})
    assert R[0] == pytest.approx(1.0)


def test_composite_zero_inputs_yield_zero():
    n = 10
    R = compute_composite_risk(
        np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n),
    )
    assert (R == 0).all()


def test_composite_preserves_input_length():
    n = 50
    rng = np.random.default_rng(0)
    R = compute_composite_risk(
        rng.uniform(0, 1, n), rng.uniform(0, 1, n),
        rng.uniform(0, 1, n), rng.uniform(0, 1, n),
    )
    assert R.shape == (n,)


# ── assign_risk_levels ────────────────────────────────────────────────


def test_assign_risk_levels_default_boundaries():
    """0.80 / 0.60 / 0.40 boundary mapping."""
    R = np.array([0.99, 0.80, 0.79, 0.61, 0.60, 0.59, 0.41, 0.40, 0.39, 0.0])
    levels = assign_risk_levels(R)
    expected = [
        "CRITICAL", "CRITICAL", "HIGH", "HIGH", "HIGH",
        "MEDIUM", "MEDIUM", "MEDIUM", "LOW", "LOW",
    ]
    assert levels.tolist() == expected


def test_assign_risk_levels_custom_thresholds():
    R = np.array([0.95, 0.85, 0.75, 0.65, 0.50])
    levels = assign_risk_levels(
        R, thresholds={"CRITICAL": 0.90, "HIGH": 0.80, "MEDIUM": 0.70},
    )
    expected = ["CRITICAL", "HIGH", "MEDIUM", "LOW", "LOW"]
    assert levels.tolist() == expected


def test_assign_risk_levels_handles_empty_input():
    R = np.array([])
    levels = assign_risk_levels(R)
    assert len(levels) == 0


def test_assign_risk_levels_uses_module_constant():
    """Boundaries must match module-level RISK_THRESHOLDS."""
    expected_boundaries = sorted([t for t, _ in RISK_THRESHOLDS], reverse=True)
    assert expected_boundaries == [0.80, 0.60, 0.40]
