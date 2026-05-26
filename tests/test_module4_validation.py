"""Module 4 validation — consistency / perturbation / cross-model tests."""
from __future__ import annotations

import json

import numpy as np

from module4_explanations.validation import validate_cross_model


def test_cross_model_empty_input_returns_empty(tmp_path):
    """Edge: no models → empty result dict."""
    out = validate_cross_model({}, output_dir=tmp_path)
    assert out["pairwise_comparisons"] == {}
    assert out["consensus_top5_all_models"] == []


def test_cross_model_pairwise_comparison_keys(tmp_path):
    """Two models → one pairwise comparison."""
    imp_a = [
        {"rank": i + 1, "feature": f"f{i}", "mean_abs_shap": 1.0 - i * 0.1}
        for i in range(10)
    ]
    imp_b = [
        {"rank": i + 1, "feature": f"f{i}", "mean_abs_shap": 1.0 - i * 0.1}
        for i in range(10)
    ]
    out = validate_cross_model({"model_a": imp_a, "model_b": imp_b},
                                output_dir=tmp_path)
    assert "model_a_vs_model_b" in out["pairwise_comparisons"]


def test_cross_model_identical_rankings_give_perfect_correlation(tmp_path):
    """Two models with identical rank order → Spearman rho ≈ 1.0."""
    imp_a = [
        {"rank": i + 1, "feature": f"f{i}", "mean_abs_shap": 1.0 - i * 0.1}
        for i in range(10)
    ]
    imp_b = imp_a.copy()
    out = validate_cross_model({"a": imp_a, "b": imp_b}, output_dir=tmp_path)
    pair = out["pairwise_comparisons"]["a_vs_b"]
    assert pair["spearman_rho"] >= 0.99


def test_cross_model_consensus_top5_intersection(tmp_path):
    """Consensus = features in top-5 of ALL models."""
    imp_a = [
        {"rank": i + 1, "feature": f"f{i}", "mean_abs_shap": 1.0 - i * 0.1}
        for i in range(10)
    ]
    # Model B has top-5 = f5, f6, f7, f0, f1 (so f0 + f1 overlap with A's top-5)
    imp_b = [
        {"rank": 1, "feature": "f5", "mean_abs_shap": 1.0},
        {"rank": 2, "feature": "f6", "mean_abs_shap": 0.9},
        {"rank": 3, "feature": "f7", "mean_abs_shap": 0.8},
        {"rank": 4, "feature": "f0", "mean_abs_shap": 0.7},
        {"rank": 5, "feature": "f1", "mean_abs_shap": 0.6},
        {"rank": 6, "feature": "f2", "mean_abs_shap": 0.5},
        {"rank": 7, "feature": "f3", "mean_abs_shap": 0.4},
        {"rank": 8, "feature": "f4", "mean_abs_shap": 0.3},
        {"rank": 9, "feature": "f8", "mean_abs_shap": 0.2},
        {"rank": 10, "feature": "f9", "mean_abs_shap": 0.1},
    ]
    out = validate_cross_model({"a": imp_a, "b": imp_b}, output_dir=tmp_path)
    # A top-5: f0..f4. B top-5: f5,f6,f7,f0,f1 → consensus = f0, f1
    assert set(out["consensus_top5_all_models"]) == {"f0", "f1"}


def test_cross_model_writes_validation_json(tmp_path):
    imp = [
        {"rank": i + 1, "feature": f"f{i}", "mean_abs_shap": 1.0 - i * 0.1}
        for i in range(10)
    ]
    validate_cross_model({"a": imp, "b": imp}, output_dir=tmp_path)
    assert (tmp_path / "validation_cross_model.json").exists()


def test_cross_model_top5_overlap_count(tmp_path):
    imp_a = [
        {"rank": i + 1, "feature": f"f{i}", "mean_abs_shap": 1.0 - i * 0.1}
        for i in range(10)
    ]
    imp_b = imp_a.copy()
    out = validate_cross_model({"a": imp_a, "b": imp_b}, output_dir=tmp_path)
    pair = out["pairwise_comparisons"]["a_vs_b"]
    assert pair["top5_overlap_count"] == 5
