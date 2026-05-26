"""Module 3 plotting — verify all 6 plot functions accept output_dir + produce PNG."""
from __future__ import annotations

import numpy as np
import pytest

from module3_risk_scoring.plotting import (
    plot_component_breakdown,
    plot_component_scatter,
    plot_dual_track_heatmap,
    plot_risk_by_category,
    plot_risk_by_label,
    plot_risk_distribution,
    plot_weight_sensitivity_curve,
)


@pytest.fixture
def synthetic_data():
    rng = np.random.default_rng(42)
    n = 60
    R = rng.uniform(0, 1, n)
    levels = np.array(["LOW", "MEDIUM", "HIGH", "CRITICAL"] * 15)
    y = np.array([0] * 30 + [1] * 30)
    cats = np.array(["normal"] * 30 + ["Spoofing"] * 15 + ["Data Alteration"] * 15)
    c_sup = rng.uniform(0, 1, n)
    c_anom = rng.uniform(0, 1, n)
    return R, levels, y, cats, c_sup, c_anom


def test_plot_risk_distribution_writes_png(synthetic_data, tmp_path):
    R, levels, *_ = synthetic_data
    out = plot_risk_distribution(R, levels, output_dir=tmp_path)
    assert out.exists()
    assert out.suffix == ".png"


def test_plot_component_breakdown_writes_png(tmp_path):
    contributions = {
        "per_level": {
            "LOW":      {"count": 5, "mean_contributions": {"C_detect": 0.1, "D_crit": 0.1, "S_data": 0.1, "D_clinical_tier": 0.1}, "dominant_component_counts": {}},
            "MEDIUM":   {"count": 5, "mean_contributions": {"C_detect": 0.2, "D_crit": 0.1, "S_data": 0.1, "D_clinical_tier": 0.1}, "dominant_component_counts": {}},
            "HIGH":     {"count": 5, "mean_contributions": {"C_detect": 0.3, "D_crit": 0.1, "S_data": 0.1, "D_clinical_tier": 0.1}, "dominant_component_counts": {}},
            "CRITICAL": {"count": 5, "mean_contributions": {"C_detect": 0.4, "D_crit": 0.1, "S_data": 0.1, "D_clinical_tier": 0.1}, "dominant_component_counts": {}},
        },
    }
    out = plot_component_breakdown(contributions, output_dir=tmp_path)
    assert out is not None and out.exists()


def test_plot_component_breakdown_returns_none_when_empty(tmp_path):
    """No active levels → no plot (return None)."""
    contributions = {"per_level": {
        l: {"count": 0} for l in ("LOW", "MEDIUM", "HIGH", "CRITICAL")
    }}
    out = plot_component_breakdown(contributions, output_dir=tmp_path)
    assert out is None


def test_plot_dual_track_heatmap_writes_png(tmp_path):
    fusion = {"quadrants": {
        "both_flag": {"true_attacks": 10, "total": 20, "true_benign": 10, "attack_categories": {}},
        "only_dae": {"true_attacks": 5, "total": 15, "true_benign": 10, "attack_categories": {}},
        "only_xgboost": {"true_attacks": 8, "total": 12, "true_benign": 4, "attack_categories": {}},
        "neither": {"true_attacks": 2, "total": 50, "true_benign": 48, "attack_categories": {}},
    }}
    out = plot_dual_track_heatmap(fusion, output_dir=tmp_path)
    assert out.exists()


def test_plot_component_scatter_writes_png(synthetic_data, tmp_path):
    _, _, y, _, c_sup, c_anom = synthetic_data
    out = plot_component_scatter(c_sup, c_anom, y, output_dir=tmp_path)
    assert out.exists()


def test_plot_risk_by_category_writes_png(synthetic_data, tmp_path):
    R, _, y, cats, *_ = synthetic_data
    out = plot_risk_by_category(R, cats, y, output_dir=tmp_path)
    assert out.exists()


def test_plot_risk_by_label_writes_png(synthetic_data, tmp_path):
    R, _, y, *_ = synthetic_data
    out = plot_risk_by_label(R, y, output_dir=tmp_path)
    assert out.exists()


def test_plot_weight_sensitivity_curve_writes_png(tmp_path):
    per_component = {
        "C_detect": [{"weight": 0.1, "auroc": 0.85}, {"weight": 0.4, "auroc": 0.95}],
        "D_crit":   [{"weight": 0.1, "auroc": 0.80}, {"weight": 0.4, "auroc": 0.82}],
        "S_data":   [{"weight": 0.1, "auroc": 0.75}, {"weight": 0.4, "auroc": 0.76}],
        "D_clinical_tier": [{"weight": 0.1, "auroc": 0.70}, {"weight": 0.4, "auroc": 0.72}],
    }
    out = plot_weight_sensitivity_curve(per_component, 0.95, output_dir=tmp_path)
    assert out.exists()
