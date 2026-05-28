"""Module 4 stakeholder — 3 builders (severity sourced from Module 3 risk_level)."""
from __future__ import annotations

import json

import numpy as np
import pytest

from module4_explanations.stakeholder import (
    build_admin_dashboard,
    build_analyst_report,
    build_clinician_summaries,
)


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


@pytest.fixture
def synthetic_risk_levels():
    """Module 3 canonical tiers for the 3 synthetic samples."""
    return np.array(["CRITICAL", "LOW", "LOW"])


# ── build_analyst_report ────────────────────────────────────────────


def test_analyst_report_includes_flagged_samples_only(
    synthetic_shap, synthetic_predictions, dae_predictions,
    synthetic_weighted_err, synthetic_risk_levels, tmp_path,
):
    feat_names = ["f1", "f2", "f3", "f4", "f5"]
    alerts = build_analyst_report(
        synthetic_shap, synthetic_predictions, synthetic_weighted_err,
        dae_predictions, feat_names, synthetic_risk_levels,
        output_dir=tmp_path,
    )
    # Sample 0 flagged by all 4 (3 track A + dae), sample 1 by xgb only
    # Sample 2 has nothing flagged AND risk_level=LOW → excluded
    indices = [a["sample_index"] for a in alerts]
    assert 0 in indices
    assert 1 in indices
    assert 2 not in indices


def test_analyst_report_severity_from_risk_level(
    synthetic_shap, synthetic_predictions, dae_predictions,
    synthetic_weighted_err, synthetic_risk_levels, tmp_path,
):
    feat_names = ["f1", "f2", "f3", "f4", "f5"]
    alerts = build_analyst_report(
        synthetic_shap, synthetic_predictions, synthetic_weighted_err,
        dae_predictions, feat_names, synthetic_risk_levels,
        output_dir=tmp_path,
    )
    a0 = next(a for a in alerts if a["sample_index"] == 0)
    assert a0["severity"] == "CRITICAL"  # risk_levels[0] == CRITICAL
    assert a0["risk_level"] == "CRITICAL"
    assert a0["consensus"] == "4/4 models flagged"  # detector signal preserved
    a1 = next(a for a in alerts if a["sample_index"] == 1)
    assert a1["severity"] == "LOW"  # risk_levels[1] == LOW (despite xgb flag)


def test_analyst_report_includes_high_risk_unflagged(
    synthetic_shap, synthetic_predictions, dae_predictions,
    synthetic_weighted_err, tmp_path,
):
    """Sample with n_flagged=0 but risk_level=HIGH must still appear.

    Module 3 can push risk_score into HIGH/CRITICAL via D_crit / S_data /
    D_clinical_tier even when no detector voted — those alerts used to
    be invisible to the analyst view.
    """
    feat_names = ["f1", "f2", "f3", "f4", "f5"]
    risk_levels = np.array(["CRITICAL", "LOW", "HIGH"])
    alerts = build_analyst_report(
        synthetic_shap, synthetic_predictions, synthetic_weighted_err,
        dae_predictions, feat_names, risk_levels, output_dir=tmp_path,
    )
    indices = [a["sample_index"] for a in alerts]
    assert 2 in indices
    a2 = next(a for a in alerts if a["sample_index"] == 2)
    assert a2["severity"] == "HIGH"
    assert a2["consensus"] == "0/4 models flagged"


def test_analyst_report_writes_json(
    synthetic_shap, synthetic_predictions, dae_predictions,
    synthetic_weighted_err, synthetic_risk_levels, tmp_path,
):
    feat_names = ["f1", "f2", "f3", "f4", "f5"]
    build_analyst_report(
        synthetic_shap, synthetic_predictions, synthetic_weighted_err,
        dae_predictions, feat_names, synthetic_risk_levels,
        output_dir=tmp_path,
    )
    assert (tmp_path / "analyst_report.json").exists()


def test_analyst_report_suffix_applied(
    synthetic_shap, synthetic_predictions, dae_predictions,
    synthetic_weighted_err, synthetic_risk_levels, tmp_path,
):
    feat_names = ["f1", "f2", "f3", "f4", "f5"]
    build_analyst_report(
        synthetic_shap, synthetic_predictions, synthetic_weighted_err,
        dae_predictions, feat_names, synthetic_risk_levels,
        suffix="_demo", output_dir=tmp_path,
    )
    assert (tmp_path / "analyst_report_demo.json").exists()


# ── build_clinician_summaries ───────────────────────────────────────


def test_clinician_summaries_only_for_xgboost_flagged(
    synthetic_shap, synthetic_predictions, dae_predictions,
    synthetic_risk_levels, tmp_path,
):
    feat_names = ["f1", "f2", "f3", "f4", "f5"]
    summaries = build_clinician_summaries(
        synthetic_shap, synthetic_predictions, dae_predictions,
        feat_names, synthetic_risk_levels, output_dir=tmp_path,
    )
    indices = [s["sample_index"] for s in summaries]
    assert indices == [0, 1]  # xgb flagged samples 0 and 1


def test_clinician_summary_includes_sample_id(
    synthetic_shap, synthetic_predictions, dae_predictions,
    synthetic_risk_levels, tmp_path,
):
    feat_names = ["f1", "f2", "f3", "f4", "f5"]
    summaries = build_clinician_summaries(
        synthetic_shap, synthetic_predictions, dae_predictions,
        feat_names, synthetic_risk_levels, output_dir=tmp_path,
    )
    for s in summaries:
        assert f"(Sample {s['sample_index']})" in s["summary"]


def test_clinician_summary_severity_from_risk_level(
    synthetic_shap, synthetic_predictions, dae_predictions, tmp_path,
):
    feat_names = ["f1", "f2", "f3", "f4", "f5"]
    risk_levels = np.array(["HIGH", "MEDIUM", "LOW"])
    summaries = build_clinician_summaries(
        synthetic_shap, synthetic_predictions, dae_predictions,
        feat_names, risk_levels, output_dir=tmp_path,
    )
    by_idx = {s["sample_index"]: s for s in summaries}
    assert by_idx[0]["severity"] == "HIGH"
    assert by_idx[1]["severity"] == "MEDIUM"


# ── build_admin_dashboard ───────────────────────────────────────────


def test_admin_dashboard_severity_counts(
    synthetic_shap, synthetic_predictions, dae_predictions, tmp_path,
):
    """alerts_by_severity reflects Module 3 risk_level, not n_flagged."""
    feat_names = ["f1", "f2", "f3", "f4", "f5"]
    feat_weights = np.array([0.3, 0.2, 0.2, 0.15, 0.15])
    global_imp = {"xgboost": [
        {"rank": 1, "feature": "f1", "mean_abs_shap": 0.5},
        {"rank": 2, "feature": "f3", "mean_abs_shap": 0.3},
    ]}
    # Sample 0 flagged 4/4 but risk_level=HIGH → bucketed HIGH
    # Sample 1 flagged 1/4 but risk_level=MEDIUM → bucketed MEDIUM
    # Sample 2 flagged 0 and risk_level=LOW → excluded from total
    risk_levels = np.array(["HIGH", "MEDIUM", "LOW"])
    dashboard = build_admin_dashboard(
        synthetic_shap, synthetic_predictions, dae_predictions, feat_names,
        feat_weights, global_imp, attack_cats=None,
        risk_levels=risk_levels, output_dir=tmp_path,
    )
    assert dashboard["alerts_by_severity"]["HIGH"] == 1
    assert dashboard["alerts_by_severity"]["MEDIUM"] == 1
    assert dashboard["alerts_by_severity"]["CRITICAL"] == 0
    assert dashboard["alerts_by_severity"]["LOW"] == 0
    assert dashboard["total_alerts"] == 2
    # model_agreement keeps the per-N-of-4 distribution as before.
    assert dashboard["model_agreement"]["4_of_4"] == 1
    assert dashboard["model_agreement"]["1_of_4"] == 1


def test_admin_dashboard_attack_categories_vectorised(
    synthetic_shap, synthetic_predictions, dae_predictions,
    synthetic_risk_levels, tmp_path,
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
        feat_weights, global_imp, attack_cats=cats,
        risk_levels=synthetic_risk_levels, output_dir=tmp_path,
    )
    # xgb flagged 0 (Spoofing) and 1 (Data Alteration)
    assert dashboard["alerts_by_attack_category"] == {
        "Spoofing": 1, "Data Alteration": 1,
    }
