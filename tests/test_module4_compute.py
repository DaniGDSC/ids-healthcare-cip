"""Module 4 compute — SHAP normalisation + top-k + Y9 feat_names validation."""
from __future__ import annotations

import numpy as np
import pytest

from module4_explanations.compute import (
    _normalise_expected_value,
    _normalise_shap_output,
    _top_features_dae,
    _top_features_shap,
    compute_global_importance,
)


# ── _normalise_shap_output ──────────────────────────────────────────


def test_normalise_shap_output_list_takes_class_1():
    sv0 = np.array([[0.1, 0.2]])
    sv1 = np.array([[0.5, 0.6]])
    out = _normalise_shap_output([sv0, sv1])
    np.testing.assert_array_equal(out, sv1)


def test_normalise_shap_output_3d_slices_class_1():
    sv = np.zeros((5, 3, 2))
    sv[:, :, 1] = 1.0
    out = _normalise_shap_output(sv)
    assert out.shape == (5, 3)
    assert (out == 1.0).all()


def test_normalise_shap_output_2d_passthrough():
    sv = np.array([[1.0, 2.0], [3.0, 4.0]])
    out = _normalise_shap_output(sv)
    np.testing.assert_array_equal(out, sv)


# ── _normalise_expected_value ───────────────────────────────────────


def test_normalise_expected_scalar():
    assert _normalise_expected_value(0.5) == 0.5


def test_normalise_expected_list_two_classes():
    assert _normalise_expected_value([0.4, 0.6]) == 0.6


def test_normalise_expected_ndarray():
    assert _normalise_expected_value(np.array([0.4, 0.6])) == 0.6


# ── _top_features_shap ──────────────────────────────────────────────


def test_top_features_shap_ordering_by_abs_value():
    sv_row = np.array([0.05, -0.5, 0.3, -0.2, 0.4])
    feat_names = ["f1", "f2", "f3", "f4", "f5"]
    top = _top_features_shap(sv_row, feat_names, k=3)
    # Expected order by |SHAP|: f2 (0.5), f5 (0.4), f3 (0.3)
    assert [t["feature"] for t in top] == ["f2", "f5", "f3"]


def test_top_features_shap_direction():
    sv_row = np.array([0.5, -0.5])
    top = _top_features_shap(sv_row, ["pos", "neg"], k=2)
    pos = next(t for t in top if t["feature"] == "pos")
    neg = next(t for t in top if t["feature"] == "neg")
    assert pos["direction"] == "increases_risk"
    assert neg["direction"] == "decreases_risk"


def test_top_features_shap_returns_k_entries():
    sv_row = np.random.randn(20)
    feat_names = [f"f{i}" for i in range(20)]
    top = _top_features_shap(sv_row, feat_names, k=5)
    assert len(top) == 5


# ── _top_features_dae ───────────────────────────────────────────────


def test_top_features_dae_pct_contribution_sums_correctly():
    werr = np.array([1.0, 2.0, 3.0, 4.0])
    top = _top_features_dae(werr, ["a", "b", "c", "d"], k=4)
    total_pct = sum(t["pct_contribution"] for t in top)
    assert abs(total_pct - 100.0) < 0.5  # allow 0.5% rounding slack


def test_top_features_dae_zero_total_handled():
    werr = np.zeros(4)
    top = _top_features_dae(werr, ["a", "b", "c", "d"], k=2)
    for t in top:
        assert t["pct_contribution"] == 0.0


# ── compute_global_importance ───────────────────────────────────────


def test_global_importance_ranked_descending():
    sv = np.array([[0.1, 0.5, 0.3], [-0.2, 0.4, -0.3]])
    imp = compute_global_importance(sv, ["f1", "f2", "f3"])
    ranked_features = [e["feature"] for e in imp]
    # mean |SHAP|: f1=0.15, f2=0.45, f3=0.3 → f2, f3, f1
    assert ranked_features == ["f2", "f3", "f1"]


# ── Y9: feat_names length validation ────────────────────────────────


def test_compute_global_importance_rejects_mismatched_feat_names():
    sv = np.zeros((5, 10))
    with pytest.raises(ValueError, match="feat_names has"):
        compute_global_importance(sv, ["a", "b"])
