"""Tests for ``tools.formula_comparison``.

Covers tier assignment for each option, metric computation, scoring,
and the JSON-safety of the output dict.
"""
from __future__ import annotations

import numpy as np

from tools.formula_comparison import (
    OPTIMAL_GATE,
    OPTIMAL_T_NORMAL,
    _v1_phase_a_tier,
    _v1_phase_ab_tier,
    _v1_phase_b_tier,
    _v1_tier,
    _prec_recall_f1,
    _per_tier_distribution,
    _counterfactual_coverage_estimate,
    evaluate_options,
)


# ── Tier assignment ────────────────────────────────────────────────


def test_v1_tier_boundaries():
    R = np.array([0.99, 0.80, 0.79, 0.61, 0.60, 0.59, 0.41, 0.40, 0.39, 0.10, 0.0])
    expected = ["CRITICAL", "CRITICAL", "HIGH", "HIGH", "HIGH",
                "MEDIUM", "MEDIUM", "MEDIUM", "LOW", "LOW", "LOW"]
    assert _v1_tier(R).tolist() == expected


def test_v1_phase_a_emits_normal_below_t_normal():
    R = np.array([0.85, 0.50, 0.30, 0.20, 0.19, 0.10, 0.0])
    out = _v1_phase_a_tier(R, t_normal=0.20).tolist()
    assert out == ["CRITICAL", "MEDIUM", "LOW", "LOW", "NORMAL", "NORMAL", "NORMAL"]


def test_v1_phase_a_with_custom_t_normal():
    R = np.array([0.30, 0.25, 0.20])
    out = _v1_phase_a_tier(R, t_normal=0.30).tolist()
    # Note: R==0.30 is >= 0.30 so still LOW (boundary inclusive)
    assert out == ["LOW", "NORMAL", "NORMAL"]


def test_v1_phase_b_demotes_low_detection_to_normal():
    R = np.array([0.85, 0.55, 0.45, 0.30])
    c = np.array([0.50, 0.02, 0.10, 0.50])  # idx 1 below gate
    out = _v1_phase_b_tier(R, c, gate=0.05).tolist()
    # idx 0: CRITICAL kept (c >= 0.05)
    # idx 1: HIGH → NORMAL (c < 0.05)
    # idx 2: MEDIUM kept
    # idx 3: LOW kept (c >= 0.05)
    assert out == ["CRITICAL", "NORMAL", "MEDIUM", "LOW"]


def test_v1_phase_ab_combines_both_gates():
    R = np.array([0.85, 0.30, 0.15, 0.50])
    c = np.array([0.50, 0.02, 0.50, 0.50])
    out = _v1_phase_ab_tier(R, c, t_normal=0.20, gate=0.05).tolist()
    # idx 0: CRITICAL kept
    # idx 1: dropped by gate (c < 0.05)
    # idx 2: dropped by t_normal (R < 0.20)
    # idx 3: MEDIUM kept
    assert out == ["CRITICAL", "NORMAL", "NORMAL", "MEDIUM"]


def test_optimal_defaults_constants_match_sweep_winner():
    """The hard-coded optimal params must match what the sweep selected
    (t_normal=0.30, gate=0.02). If anyone changes them, the comparison
    output stops corresponding to the documented winning row."""
    assert OPTIMAL_T_NORMAL == 0.30
    assert OPTIMAL_GATE     == 0.02


# ── Metric helpers ─────────────────────────────────────────────────


def test_prec_recall_f1_perfect():
    y = np.array([1, 1, 0, 0])
    mask = y.astype(bool)
    out = _prec_recall_f1(mask, y)
    assert out["precision"] == 1.0
    assert out["recall"]    == 1.0
    assert out["f1"]        == 1.0


def test_prec_recall_f1_zero_alerts():
    y = np.array([1, 0])
    mask = np.zeros(2, dtype=bool)
    out = _prec_recall_f1(mask, y)
    assert out["alert_volume"] == 0
    assert out["precision"]    == 0.0
    assert out["recall"]       == 0.0


def test_per_tier_distribution_counts():
    tiers = np.array(["CRITICAL", "HIGH", "HIGH", "LOW", "NORMAL"])
    y = np.array([1, 1, 0, 0, 0])
    out = _per_tier_distribution(tiers, y)
    assert out["CRITICAL"] == {"total": 1, "attacks": 1, "benign": 0}
    assert out["HIGH"]     == {"total": 2, "attacks": 1, "benign": 1}
    assert out["LOW"]      == {"total": 1, "attacks": 0, "benign": 1}
    assert out["NORMAL"]   == {"total": 1, "attacks": 0, "benign": 1}
    assert out["MEDIUM"]   == {"total": 0, "attacks": 0, "benign": 0}


def test_counterfactual_estimate_uses_proba_threshold():
    tiers = np.array(["HIGH", "MEDIUM", "LOW", "NORMAL"])
    y_proba = np.array([0.9, 0.4, 0.005, 0.001])
    out = _counterfactual_coverage_estimate(tiers, y_proba, threshold=0.5)
    # actionable = HIGH + MEDIUM = 2 samples; only HIGH has proba ≥ 0.5
    assert out["actionable_feasible_est"] == {"seen": 2, "feasible_est": 1, "rate": 0.5}
    assert out["any_alert_feasible_est"]["seen"] == 3  # HIGH + MEDIUM + LOW


# ── Full evaluate_options integration ──────────────────────────────


def test_evaluate_options_returns_all_four_options():
    n = 10
    R = np.linspace(0.1, 0.9, n)
    c = np.linspace(0.0, 0.5, n)
    y = (R > 0.5).astype(int)
    y_proba = R.copy()  # plausible stand-in
    out = evaluate_options(R, c, y, y_proba, threshold=0.5)
    assert set(out["per_option"].keys()) == {
        "v1_baseline", "v1_phase_a", "v1_phase_b", "v1_phase_a_plus_b",
    }


def test_evaluate_options_winner_is_well_defined():
    n = 100
    rng = np.random.default_rng(42)
    R = rng.uniform(0.0, 1.0, n)
    c = rng.uniform(0.0, 1.0, n)
    y = (R > 0.5).astype(int)
    y_proba = R.copy()
    out = evaluate_options(R, c, y, y_proba, threshold=0.5)
    assert out["winner"] in out["per_option"]
    # Winner's score must equal the top of ranking
    top_name, top_score = out["ranking"][0]
    assert out["winner"] == top_name
    assert out["per_option"][top_name]["score"]["total_score"] == top_score


def test_evaluate_options_baseline_is_unscored_reference():
    """``v1_baseline`` is included in the table for reference but should
    not dominate scoring — it has zero noise reduction by construction."""
    n = 50
    rng = np.random.default_rng(1)
    R = rng.uniform(0.0, 1.0, n)
    c = rng.uniform(0.0, 1.0, n)
    y = (R > 0.5).astype(int)
    y_proba = R.copy()
    out = evaluate_options(R, c, y, y_proba, threshold=0.5)
    base_score = out["per_option"]["v1_baseline"]["score"]["components"]
    # noise_reduction component is exactly 0 for the baseline (alert_volume
    # equals baseline alert_volume).
    assert base_score["noise_reduction"] == 0.0


def test_evaluate_options_json_safe():
    """The output must serialise without numpy scalars in it — the
    driver writes it to disk verbatim."""
    import json
    n = 20
    R = np.linspace(0.1, 0.9, n)
    c = np.linspace(0.0, 0.5, n)
    y = (R > 0.5).astype(int)
    y_proba = R.copy()
    out = evaluate_options(R, c, y, y_proba, threshold=0.5)
    json.dumps(out)  # must not raise
