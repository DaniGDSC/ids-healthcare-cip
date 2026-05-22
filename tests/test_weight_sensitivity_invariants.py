"""Invariant tests for Stage 5B weight sensitivity (Fix 1).

These tests assert structural and safety invariants on the analysis script
without requiring the full run. They are fast (no real analysis run; small
synthetic components fixture) and catch regressions in the script's
contract.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.compute_weight_sensitivity import (  # noqa: E402
    MAGNITUDES,
    N_PERTURBATIONS,
    RANDOM_SEED,
    TIER_BOUNDARIES,
    TIER_CRITICAL,
    _agreement_exact_tier_match,
    _assign_tier,
    _baseline_c_detect_only,
    _baseline_equal_weights,
    _compute_multiplicative_R,
    _fnr_critical_delta,
    _perturb_weights,
    _run_named_baselines,
    _run_perturbations_for_magnitude,
)


# ----- Synthetic fixture (small, deterministic) -----
@pytest.fixture
def small_components():
    """50-alert fixture with mixed-tier baseline R."""
    rng = np.random.default_rng(0)
    c_detect = rng.uniform(0.1, 0.95, size=50)
    d_crit = rng.uniform(0.0, 1.0, size=50)
    s_data = rng.uniform(0.0, 1.0, size=50)
    d_clinical_tier = rng.uniform(0.0, 1.0, size=50)
    y_true = rng.choice([0, 1], size=50, p=[0.95, 0.05])  # 5% positives
    return c_detect, d_crit, s_data, d_clinical_tier, y_true


# ----- Test 1: sum-to-1.0 invariant on perturbations -----
def test_perturbations_sum_to_one(small_components):
    """Every perturbed weight dict must sum to 1.0 within 1e-6.

    Matches the production invariant at
    module3_risk_scoring/module3_risk_scores.py:86-90 (Session 8 Q-W3).
    """
    rng = np.random.default_rng(RANDOM_SEED)
    base = {"w1": 0.4, "w2": 0.25, "w3": 0.15, "w4": 0.2}
    for mag in MAGNITUDES:
        for _ in range(100):  # 100 trials per magnitude
            pert = _perturb_weights(rng, base, mag)
            s = sum(pert.values())
            assert abs(s - 1.0) <= 1e-6, (
                f"Perturbation sum {s} != 1.0; weights {pert}"
            )


# ----- Test 2: tier assignment matches expected semantics -----
def test_tier_boundaries_correct():
    """_assign_tier must produce LOW/MEDIUM/HIGH/CRITICAL per boundaries."""
    R = np.array([0.1, 0.45, 0.65, 0.85, 0.95])
    tiers = _assign_tier(R, TIER_BOUNDARIES)
    # LOW, MEDIUM, HIGH, CRITICAL, CRITICAL
    assert tiers.tolist() == [0, 1, 2, 3, 3]


# ----- Test 3: agreement metric reflexivity -----
def test_agreement_reflexive():
    """Identical tier vectors must yield agreement == 1.0."""
    tiers = np.array([0, 1, 2, 3, 0, 1])
    assert _agreement_exact_tier_match(tiers, tiers) == 1.0


# ----- Test 4: agreement bounded in [0, 1] -----
def test_agreement_bounds(small_components):
    """All per-magnitude agreement summary stats must be in [0, 1]."""
    rng = np.random.default_rng(RANDOM_SEED)
    c_detect, d_crit, s_data, d_clinical_tier, y_true = small_components
    base = {"w1": 0.4, "w2": 0.25, "w3": 0.15, "w4": 0.2}
    from analysis.compute_weight_sensitivity import compute_composite_risk
    R_base = compute_composite_risk(
        c_detect, d_crit, s_data, d_clinical_tier, base
    )
    tiers_base = _assign_tier(R_base, TIER_BOUNDARIES)
    res = _run_perturbations_for_magnitude(
        rng, 0.10, c_detect, d_crit, s_data, d_clinical_tier,
        y_true, base, tiers_base,
    )
    assert 0.0 <= res["agreement_min"] <= res["agreement_max"] <= 1.0


# ----- Test 5: multiplicative baseline formula -----
def test_multiplicative_formula():
    """R_mult = c_detect * max(d_crit, s_data, d_clinical_tier) verbatim.

    Per analysis/compute_rq1.py:357-359 (Session 11 section 4).
    """
    c = np.array([0.5, 0.8])
    d = np.array([0.4, 0.1])
    s = np.array([0.3, 0.9])
    t = np.array([0.2, 0.5])
    R = _compute_multiplicative_R(c, d, s, t)
    expected = c * np.maximum.reduce([d, s, t])
    np.testing.assert_allclose(R, expected)


# ----- Test 6: equal-weights baseline -----
def test_equal_weights_baseline_sum():
    """Equal weights must sum to 1.0 and be symmetric."""
    w = _baseline_equal_weights()
    assert all(v == 0.25 for v in w.values())
    assert abs(sum(w.values()) - 1.0) <= 1e-6


# ----- Test 7: c-detect-only baseline -----
def test_c_detect_only_baseline():
    """c_detect_only must be (1.0, 0, 0, 0)."""
    w = _baseline_c_detect_only()
    assert w == {"w1": 1.0, "w2": 0.0, "w3": 0.0, "w4": 0.0}


# ----- Test 8: deterministic with fixed seed -----
def test_deterministic_under_seed(small_components):
    """Two runs with the same seed must produce identical results."""
    c_detect, d_crit, s_data, d_clinical_tier, y_true = small_components
    base = {"w1": 0.4, "w2": 0.25, "w3": 0.15, "w4": 0.2}
    from analysis.compute_weight_sensitivity import compute_composite_risk
    R_base = compute_composite_risk(
        c_detect, d_crit, s_data, d_clinical_tier, base
    )
    tiers_base = _assign_tier(R_base, TIER_BOUNDARIES)

    rng1 = np.random.default_rng(RANDOM_SEED)
    rng2 = np.random.default_rng(RANDOM_SEED)
    res1 = _run_perturbations_for_magnitude(
        rng1, 0.10, c_detect, d_crit, s_data, d_clinical_tier,
        y_true, base, tiers_base,
    )
    res2 = _run_perturbations_for_magnitude(
        rng2, 0.10, c_detect, d_crit, s_data, d_clinical_tier,
        y_true, base, tiers_base,
    )
    assert res1["agreement_mean"] == res2["agreement_mean"]
    assert res1["histogram_counts"] == res2["histogram_counts"]


# ----- Test 9: number of perturbations matches constant -----
def test_n_perturbations_constant():
    """N_PERTURBATIONS == 30 per design memo D2 (Phase 1 pick)."""
    assert N_PERTURBATIONS == 30


# ----- Test 10: magnitudes match design memo D1 -----
def test_magnitudes_constant():
    """MAGNITUDES == (0.10, 0.20) per design memo D1 (Phase 1 pick)."""
    assert MAGNITUDES == (0.10, 0.20)


# ----- Test 11: random seed matches legacy -----
def test_random_seed_constant():
    """RANDOM_SEED == 42 (matches legacy artifact's provenance)."""
    assert RANDOM_SEED == 42


# ----- Test 12: named baselines schema -----
def test_named_baselines_schema(small_components):
    """_run_named_baselines must return three baselines, each with the
    keys 'agreement' and 'fnr_critical_delta'."""
    c_detect, d_crit, s_data, d_clinical_tier, _ = small_components
    base = {"w1": 0.4, "w2": 0.25, "w3": 0.15, "w4": 0.2}
    from analysis.compute_weight_sensitivity import compute_composite_risk
    R_base = compute_composite_risk(
        c_detect, d_crit, s_data, d_clinical_tier, base
    )
    tiers_base = _assign_tier(R_base, TIER_BOUNDARIES)
    bl = _run_named_baselines(
        c_detect, d_crit, s_data, d_clinical_tier, tiers_base
    )
    assert set(bl.keys()) == {"equal_weights", "c_detect_only", "multiplicative"}
    for name, vals in bl.items():
        assert set(vals.keys()) == {"agreement", "fnr_critical_delta"}, name


# ----- Test 13: TIER_CRITICAL integer encoding -----
def test_tier_critical_constant():
    """TIER_CRITICAL == 3 per analysis/compute_rq1.py:365 (Session 11 Q-V5)."""
    assert TIER_CRITICAL == 3


# ----- Test 14: tier boundaries match YAML -----
def test_tier_boundaries_match_yaml():
    """TIER_BOUNDARIES == (0.80, 0.60, 0.40) per
    configs/composite_risk_weights.yaml (Session 8 Q-W3).
    """
    assert TIER_BOUNDARIES == (0.80, 0.60, 0.40)


# ----- Test 15: fnr_critical_delta bounded -----
def test_fnr_critical_delta_bounded(small_components):
    """fnr_critical_delta must be in [0, 1] (tier-to-tier safety-floor rate).

    Equal to baseline tiers vs themselves should give 0.0 (no rows dropped).
    Equal to maximally-perturbed tiers (all dropped to LOW) should give
    crit_base.mean() — the fraction of population that was baseline-CRITICAL.
    """
    c_detect, d_crit, s_data, d_clinical_tier, _ = small_components
    base = {"w1": 0.4, "w2": 0.25, "w3": 0.15, "w4": 0.2}
    from analysis.compute_weight_sensitivity import compute_composite_risk
    R_base = compute_composite_risk(
        c_detect, d_crit, s_data, d_clinical_tier, base
    )
    tiers_base = _assign_tier(R_base, TIER_BOUNDARIES)
    # Reflexive: no rows dropped → 0.0
    assert _fnr_critical_delta(tiers_base, tiers_base) == 0.0
    # All dropped → fraction of population that was baseline-CRITICAL
    tiers_zero = np.zeros_like(tiers_base)
    expected = float(np.mean(tiers_base == TIER_CRITICAL))
    assert _fnr_critical_delta(tiers_base, tiers_zero) == expected
    # Bounded
    delta = _fnr_critical_delta(tiers_base, tiers_zero)
    assert 0.0 <= delta <= 1.0
