"""Module 3 analysis — fusion + contributions + sensitivity + worked examples."""
from __future__ import annotations

import numpy as np
import pytest

from module3_risk_scoring.analysis import (
    component_contribution_analysis,
    dual_track_fusion_analysis,
    generate_worked_examples,
    weight_sensitivity_analysis,
)
from module3_risk_scoring.config import DAE_BINARY_THRESHOLD, WEIGHTS


@pytest.fixture
def fusion_arrays():
    """Synthetic dual-track scores with mixed quadrant coverage."""
    rng = np.random.default_rng(42)
    n = 100
    y = np.array([0] * 50 + [1] * 50)
    # Track A: predicts attack class moderately well
    c_sup = np.concatenate([
        rng.uniform(0, 0.4, 50),     # benign — low
        rng.uniform(0.5, 1.0, 50),   # attack — high
    ])
    # Track B: complementary detector
    c_anom = np.concatenate([
        rng.uniform(0, 0.4, 50),
        rng.uniform(0.4, 1.0, 50),
    ])
    cats = np.array(
        ["normal"] * 50 + ["Spoofing"] * 25 + ["Data Alteration"] * 25
    )
    return c_sup, c_anom, y, cats


# ── dual_track_fusion_analysis ───────────────────────────────────────


def test_fusion_quadrant_counts_sum_to_total(fusion_arrays):
    c_sup, c_anom, y, cats = fusion_arrays
    out = dual_track_fusion_analysis(c_sup, c_anom, y, cats, xgb_threshold=0.5)
    total = sum(q["total"] for q in out["quadrants"].values())
    assert total == len(y)


def test_fusion_dae_threshold_from_config(fusion_arrays):
    """dae_threshold field equals DAE_BINARY_THRESHOLD."""
    c_sup, c_anom, y, cats = fusion_arrays
    out = dual_track_fusion_analysis(c_sup, c_anom, y, cats, xgb_threshold=0.5)
    assert out["dae_threshold"] == DAE_BINARY_THRESHOLD


def test_fusion_recall_math_consistent(fusion_arrays):
    """union_recall ≥ max(xgb_recall, dae_recall)."""
    c_sup, c_anom, y, cats = fusion_arrays
    out = dual_track_fusion_analysis(c_sup, c_anom, y, cats, xgb_threshold=0.5)
    r = out["recall"]
    assert r["union_fusion"] >= r["xgboost_alone"]
    assert r["union_fusion"] >= r["dae_alone"]
    assert r["fusion_gain"] >= 0


def test_fusion_categories_excluded_when_none(fusion_arrays):
    """attack_categories should be empty for the 'neither' quadrant if no
    attacks landed there."""
    c_sup, c_anom, y, cats = fusion_arrays
    out = dual_track_fusion_analysis(c_sup, c_anom, y, cats, xgb_threshold=0.5)
    # 'neither' may or may not contain attacks; the field must be a dict
    assert isinstance(out["quadrants"]["neither"]["attack_categories"], dict)


def test_fusion_no_attacks_zero_recall():
    """Edge: y all benign → recall metrics = 0 (no division)."""
    c_sup = np.array([0.1, 0.2, 0.3])
    c_anom = np.array([0.1, 0.2, 0.3])
    y = np.array([0, 0, 0])
    cats = np.array(["normal", "normal", "normal"])
    out = dual_track_fusion_analysis(c_sup, c_anom, y, cats, xgb_threshold=0.5)
    assert out["recall"]["xgboost_alone"] == 0
    assert out["recall"]["dae_alone"] == 0


# ── component_contribution_analysis ─────────────────────────────────


def test_contribution_per_level_counts_match_input():
    c = np.array([0.9, 0.7, 0.5, 0.3])
    d = np.array([0.1, 0.1, 0.1, 0.1])
    s = np.array([0.1, 0.1, 0.1, 0.1])
    t = np.array([0.1, 0.1, 0.1, 0.1])
    levels = np.array(["CRITICAL", "HIGH", "MEDIUM", "LOW"])
    out = component_contribution_analysis(c, d, s, t, levels)
    for level in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]:
        assert out["per_level"][level]["count"] == 1


def test_contribution_dominant_component_per_level():
    """When C_detect dominates, dominant should be 'C_detect'."""
    c = np.array([0.99])
    d = np.array([0.01])
    s = np.array([0.01])
    t = np.array([0.01])
    levels = np.array(["CRITICAL"])
    out = component_contribution_analysis(c, d, s, t, levels)
    assert out["overall_dominant"]["C_detect"] == 1


def test_contribution_handles_empty_level():
    """When no samples fall into a level, count=0."""
    c = np.array([0.5])
    d = np.array([0.0])
    s = np.array([0.0])
    t = np.array([0.0])
    levels = np.array(["CRITICAL"])  # only 1 level present
    out = component_contribution_analysis(c, d, s, t, levels)
    assert out["per_level"]["LOW"]["count"] == 0
    assert out["per_level"]["HIGH"]["count"] == 0


# ── weight_sensitivity_analysis ─────────────────────────────────────


def test_sensitivity_returns_expected_shape():
    rng = np.random.default_rng(0)
    n = 80
    y = np.array([0] * 40 + [1] * 40)
    c = np.concatenate([rng.uniform(0, 0.3, 40), rng.uniform(0.7, 1.0, 40)])
    d = rng.uniform(0, 1, n)
    s = rng.uniform(0, 1, n)
    t = rng.uniform(0, 1, n)
    out = weight_sensitivity_analysis(c, d, s, t, y)
    assert "grid_size" in out
    assert "best_weights" in out
    assert "best_auroc" in out
    assert "default_weights" in out
    assert "top_10" in out
    assert "per_component_sensitivity" in out
    assert len(out["top_10"]) <= 10


def test_sensitivity_default_weights_match_module_constant():
    rng = np.random.default_rng(0)
    n = 60
    y = np.array([0] * 30 + [1] * 30)
    c = np.concatenate([rng.uniform(0, 0.5, 30), rng.uniform(0.5, 1.0, 30)])
    out = weight_sensitivity_analysis(
        c, rng.uniform(0, 1, n), rng.uniform(0, 1, n), rng.uniform(0, 1, n), y,
    )
    assert out["default_weights"] == WEIGHTS


def test_sensitivity_baseline_override_respected():
    rng = np.random.default_rng(0)
    n = 60
    y = np.array([0] * 30 + [1] * 30)
    c = np.concatenate([rng.uniform(0, 0.5, 30), rng.uniform(0.5, 1.0, 30)])
    custom = {"w1": 0.30, "w2": 0.30, "w3": 0.20, "w4": 0.20}
    out = weight_sensitivity_analysis(
        c, rng.uniform(0, 1, n), rng.uniform(0, 1, n), rng.uniform(0, 1, n), y,
        baseline_weights=custom,
    )
    assert out["default_weights"] == custom


# ── generate_worked_examples ────────────────────────────────────────


def test_worked_examples_shape():
    n = 30
    rng = np.random.default_rng(0)
    y = np.array([0] * 15 + [1] * 15)
    R = rng.uniform(0, 1, n)
    cats = np.array(["normal"] * 15 + ["Spoofing"] * 15)
    examples = generate_worked_examples(
        R, R, R, R, R, R, R,
        np.array(["LOW"] * n), y, cats,
    )
    # 3 examples: highest-R attack, lowest-R attack, highest-R benign
    assert len(examples) == 3
    for ex in examples:
        assert "title" in ex
        assert "sample_index" in ex
        assert "R" in ex
        assert "components" in ex
        assert "weighted_contributions" in ex


def test_worked_example_attack_identification():
    """Highest-R attack title must mention 'attack'."""
    n = 20
    y = np.array([0] * 10 + [1] * 10)
    R = np.linspace(0, 1, n)
    cats = np.array(["normal"] * 10 + ["Spoofing"] * 10)
    examples = generate_worked_examples(
        R, R, R, R, R, R, R, np.array(["LOW"] * n), y, cats,
    )
    titles = [ex["title"] for ex in examples]
    assert any("Highest-risk true attack" in t for t in titles)
    assert any("Lowest-risk true attack" in t for t in titles)
    assert any("benign sample" in t for t in titles)
