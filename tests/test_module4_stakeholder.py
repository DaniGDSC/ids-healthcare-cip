"""Module 4 stakeholder — _severity + 3 builders."""
from __future__ import annotations

import json

import numpy as np
import pytest

from module4_explanations.stakeholder import (
    _severity,
    build_admin_dashboard,
    build_analyst_report,
    build_clinician_summaries,
)


# ── _severity boundaries ────────────────────────────────────────────


@pytest.mark.parametrize("n_flagged,expected", [
    (4, "CRITICAL"), (3, "HIGH"), (2, "MEDIUM"), (1, "LOW"), (0, "LOW"),
])
def test_severity_mapping(n_flagged, expected):
    assert _severity(n_flagged) == expected


# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def synthetic_predictions():
    """3 samples, 3 models — sample 0 flagged by all, sample 1 by xgb only."""
    return {
        "xgboost": {"y_pred": np.array([1, 1, 0]),
                    "y_proba": np.array([0.9, 0.55, 0.2])},
        "random_forest": {"y_pred": np.array([1, 0, 0]),
                          "y_proba": np.array([0.85, 0.45, 0.15])},
        "decision_tree": {"y_pred": np.array([1, 0, 0]),
                          "y_proba": np.array([0.8, 0.4, 0.1])},
    }


@pytest.fixture
def dae_predictions():
    return {
        "y_pred": np.array([1, 0, 0]),
        "reconstruction_error": np.array([0.5, 0.1, 0.05]),
    }


@pytest.fixture
def synthetic_shap():
    return {"xgboost": np.array([
        [0.3, -0.1, 0.4, 0.05, -0.2],
        [0.1, -0.05, 0.15, 0.02, -0.08],
        [0.01, 0.0, 0.02, 0.0, -0.01],
    ])}


@pytest.fixture
def synthetic_weighted_err():
    return np.array([
        [0.5, 0.3, 0.4, 0.1, 0.05],
        [0.05, 0.02, 0.04, 0.01, 0.005],
        [0.001, 0.001, 0.002, 0.0, 0.001],
    ])


# ── build_analyst_report ────────────────────────────────────────────


def test_analyst_report_includes_flagged_samples_only(
    synthetic_shap, synthetic_predictions, dae_predictions,
    synthetic_weighted_err, tmp_path,
):
    feat_names = ["f1", "f2", "f3", "f4", "f5"]
    alerts = build_analyst_report(
        synthetic_shap, synthetic_predictions, synthetic_weighted_err,
        dae_predictions, feat_names, output_dir=tmp_path,
    )
    # Sample 0 flagged by all 4 (3 track A + dae), sample 1 by xgb only
    # Sample 2 has nothing flagged → excluded
    indices = [a["sample_index"] for a in alerts]
    assert 0 in indices
    assert 1 in indices
    assert 2 not in indices


def test_analyst_report_severity_mapping(
    synthetic_shap, synthetic_predictions, dae_predictions,
    synthetic_weighted_err, tmp_path,
):
    feat_names = ["f1", "f2", "f3", "f4", "f5"]
    alerts = build_analyst_report(
        synthetic_shap, synthetic_predictions, synthetic_weighted_err,
        dae_predictions, feat_names, output_dir=tmp_path,
    )
    a0 = next(a for a in alerts if a["sample_index"] == 0)
    assert a0["severity"] == "CRITICAL"  # 4 models flagged


def test_analyst_report_writes_json(
    synthetic_shap, synthetic_predictions, dae_predictions,
    synthetic_weighted_err, tmp_path,
):
    feat_names = ["f1", "f2", "f3", "f4", "f5"]
    build_analyst_report(
        synthetic_shap, synthetic_predictions, synthetic_weighted_err,
        dae_predictions, feat_names, output_dir=tmp_path,
    )
    assert (tmp_path / "analyst_report.json").exists()


def test_analyst_report_suffix_applied(
    synthetic_shap, synthetic_predictions, dae_predictions,
    synthetic_weighted_err, tmp_path,
):
    feat_names = ["f1", "f2", "f3", "f4", "f5"]
    build_analyst_report(
        synthetic_shap, synthetic_predictions, synthetic_weighted_err,
        dae_predictions, feat_names, suffix="_demo", output_dir=tmp_path,
    )
    assert (tmp_path / "analyst_report_demo.json").exists()


# ── build_clinician_summaries ───────────────────────────────────────


def test_clinician_summaries_only_for_xgboost_flagged(
    synthetic_shap, synthetic_predictions, dae_predictions, tmp_path,
):
    feat_names = ["f1", "f2", "f3", "f4", "f5"]
    summaries = build_clinician_summaries(
        synthetic_shap, synthetic_predictions, dae_predictions,
        feat_names, output_dir=tmp_path,
    )
    indices = [s["sample_index"] for s in summaries]
    assert indices == [0, 1]  # xgb flagged samples 0 and 1


def test_clinician_summary_includes_sample_id(
    synthetic_shap, synthetic_predictions, dae_predictions, tmp_path,
):
    feat_names = ["f1", "f2", "f3", "f4", "f5"]
    summaries = build_clinician_summaries(
        synthetic_shap, synthetic_predictions, dae_predictions,
        feat_names, output_dir=tmp_path,
    )
    for s in summaries:
        assert f"(Sample {s['sample_index']})" in s["summary"]


# ── build_admin_dashboard ───────────────────────────────────────────


def test_admin_dashboard_severity_counts(
    synthetic_shap, synthetic_predictions, dae_predictions, tmp_path,
):
    feat_names = ["f1", "f2", "f3", "f4", "f5"]
    feat_weights = np.array([0.3, 0.2, 0.2, 0.15, 0.15])
    global_imp = {"xgboost": [
        {"rank": 1, "feature": "f1", "mean_abs_shap": 0.5},
        {"rank": 2, "feature": "f3", "mean_abs_shap": 0.3},
    ]}
    dashboard = build_admin_dashboard(
        synthetic_shap, synthetic_predictions, dae_predictions, feat_names,
        feat_weights, global_imp, attack_cats=None, output_dir=tmp_path,
    )
    assert dashboard["alerts_by_severity"]["CRITICAL"] == 1  # sample 0
    assert dashboard["alerts_by_severity"]["LOW"] == 1  # sample 1


def test_admin_dashboard_attack_categories_vectorised(
    synthetic_shap, synthetic_predictions, dae_predictions, tmp_path,
):
    """N7 fix: vectorised attack-category counting via np.unique."""
    feat_names = ["f1", "f2", "f3", "f4", "f5"]
    feat_weights = np.array([0.3, 0.2, 0.2, 0.15, 0.15])
    global_imp = {"xgboost": [
        {"rank": 1, "feature": "f1", "mean_abs_shap": 0.5},
    ]}
    cats = np.array(["Spoofing", "Data Alteration", "normal"], dtype=object)
    dashboard = build_admin_dashboard(
        synthetic_shap, synthetic_predictions, dae_predictions, feat_names,
        feat_weights, global_imp, attack_cats=cats, output_dir=tmp_path,
    )
    # xgb flagged 0 (Spoofing) and 1 (Data Alteration)
    assert dashboard["alerts_by_attack_category"] == {
        "Spoofing": 1, "Data Alteration": 1,
    }
