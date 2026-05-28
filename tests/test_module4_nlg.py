"""Module 4 nlg — clinician_nlg + 6-step + route_explanation."""
from __future__ import annotations

import numpy as np

from module4_explanations.nlg import (
    build_shap_context,
    clinician_nlg,
    generate_clinician_alert,
    route_explanation,
)


# ── clinician_nlg ────────────────────────────────────────────────────


def test_low_severity_returns_low_template():
    out = clinician_nlg("LOW", [{"feature": "f1", "shap_value": 0.1}])
    assert "LOW ALERT" in out


def test_critical_severity_uses_critical_template():
    top = [
        {"feature": "DIntPkt", "shap_value": 0.5, "direction": "increases_risk"},
        {"feature": "Sport", "shap_value": -0.1, "direction": "decreases_risk"},
    ]
    out = clinician_nlg("CRITICAL", top)
    assert "CRITICAL ALERT" in out
    assert "unusual network packet timing" in out


def test_confidence_band_cites_secondary_indicator():
    """When top-2 SHAP is ≥80% of top-1 magnitude AND in a different
    category, secondary narrative gets mentioned."""
    top = [
        {"feature": "DIntPkt", "shap_value": 0.50, "direction": "increases_risk"},
        {"feature": "Pulse_Rate", "shap_value": 0.45, "direction": "increases_risk"},
        {"feature": "Flgs", "shap_value": 0.1, "direction": "increases_risk"},
    ]
    out = clinician_nlg("HIGH", top)
    assert "secondary indicator" in out
    assert "abnormal pulse rate" in out


def test_biometric_note_when_bio_in_topk_but_top1_network():
    """If any top-3 feature is biometric and top-1 isn't, biometric note appears."""
    top = [
        {"feature": "DIntPkt", "shap_value": 0.5, "direction": "increases_risk"},
        {"feature": "SrcLoad", "shap_value": 0.05, "direction": "increases_risk"},
        {"feature": "Pulse_Rate", "shap_value": 0.04, "direction": "increases_risk"},
    ]
    out = clinician_nlg("HIGH", top)
    assert "Biometric" in out
    assert "Pulse_Rate" in out


# ── build_shap_context ──────────────────────────────────────────────


def test_shap_context_top_category_by_abs_sum():
    """top_category should be the group with largest sum(|SHAP|)."""
    top = [
        {"feature": "DIntPkt", "shap_value": 0.4, "direction": "increases_risk"},
        {"feature": "SIntPkt", "shap_value": 0.3, "direction": "increases_risk"},
        {"feature": "Pulse_Rate", "shap_value": 0.2, "direction": "increases_risk"},
    ]
    ctx = build_shap_context(top)
    # network_timing total = 0.7, biometric = 0.2 → network_timing wins
    assert ctx["top_category"] == "network_timing"


def test_shap_context_confidence_levels():
    high_top = [{"feature": "x", "shap_value": 0.5, "direction": "increases_risk"}]
    medium_top = [{"feature": "x", "shap_value": 0.2, "direction": "increases_risk"}]
    low_top = [{"feature": "x", "shap_value": 0.05, "direction": "increases_risk"}]
    assert build_shap_context(high_top)["confidence_from_shap"] == "HIGH"
    assert build_shap_context(medium_top)["confidence_from_shap"] == "MEDIUM"
    assert build_shap_context(low_top)["confidence_from_shap"] == "LOW"


def test_shap_context_direction_elevated_vs_suppressed():
    pos = [{"feature": "x", "shap_value": 0.5, "direction": "increases_risk"}]
    neg = [{"feature": "x", "shap_value": -0.5, "direction": "decreases_risk"}]
    assert build_shap_context(pos)["shap_direction"] == "elevated"
    assert build_shap_context(neg)["shap_direction"] == "suppressed"


def test_shap_context_empty_returns_empty():
    assert build_shap_context([]) == {}


# ── generate_clinician_alert (6-step) ───────────────────────────────


def test_six_step_alert_includes_all_sections():
    sv_row = np.array([0.5, -0.2, 0.3, 0.1, -0.05])
    feat_names = ["DIntPkt", "SrcLoad", "Pulse_Rate", "TotBytes", "ST"]
    out = generate_clinician_alert(
        idx=42, sv_row=sv_row, feat_names=feat_names,
        severity="CRITICAL", confidence=0.95, consensus="2/2",
        risk_score=0.75, risk_components={
            "c_detect": 0.8, "d_crit": 0.5, "s_data": 0.6, "d_clinical_tier": 0.4,
        },
        d_clinical_tier_val=0.5,
    )
    assert "CRITICAL SECURITY ALERT" in out
    assert "intrusion detection system" in out
    assert "Composite risk score" in out
    assert "Recommended" in out


def test_six_step_alert_acuity_note_normal_when_d_clinical_zero():
    sv_row = np.zeros(5)
    feat_names = ["a", "b", "c", "d", "e"]
    out = generate_clinician_alert(
        idx=0, sv_row=sv_row, feat_names=feat_names,
        severity="HIGH", confidence=0.7, consensus="1/2",
        d_clinical_tier_val=0.0,
    )
    assert "within normal ranges" in out


# ── route_explanation ───────────────────────────────────────────────


def test_route_clinician_returns_text():
    sv_row = np.zeros(2)
    out = route_explanation(
        idx=0, stakeholder_role="clinician",
        sv_row=sv_row, feat_names=["a", "b"],
        severity="HIGH", confidence=0.8, consensus="1/2",
        risk_score=0.6, risk_components={},
        d_clinical_tier_val=0.0, dae_top_features=[],
    )
    assert out["format"] == "text"
    assert out["role"] == "clinician"


def test_route_analyst_returns_json_with_charts():
    sv_row = np.array([0.5, -0.3])
    out = route_explanation(
        idx=7, stakeholder_role="analyst",
        sv_row=sv_row, feat_names=["a", "b"],
        severity="HIGH", confidence=0.8, consensus="1/2",
        risk_score=0.6, risk_components={"c_detect": 0.5},
        d_clinical_tier_val=0.0, dae_top_features=[{"feature": "a"}],
    )
    assert out["format"] == "json"
    content = out["content"]
    assert content["sample_index"] == 7
    assert "waterfall_xgboost_sample_0007.png" in content["charts"]


def test_route_administrator_action_required_flag():
    out = route_explanation(
        idx=0, stakeholder_role="administrator",
        sv_row=np.zeros(2), feat_names=["a", "b"],
        severity="CRITICAL", confidence=0.95, consensus="2/2",
        risk_score=0.8, risk_components={},
        d_clinical_tier_val=0.0, dae_top_features=[],
    )
    assert out["content"]["action_required"] is True


def test_route_unknown_role_fallback():
    out = route_explanation(
        idx=0, stakeholder_role="unknown_role",
        sv_row=np.zeros(2), feat_names=["a", "b"],
        severity="LOW", confidence=0.5, consensus="0/2",
        risk_score=0.3, risk_components={},
        d_clinical_tier_val=0.0, dae_top_features=[],
    )
    assert out["content"] == "Unknown role"
