"""Tests for src/data_models — schema invariants + to_dict serialization."""
from __future__ import annotations

from src.data_models import (
    AlertGroundTruth,
    AlertRecord,
    MVEOutput,
    ScoredAlert,
    TestReport,
)


def _basic_mve(**kwargs) -> MVEOutput:
    return MVEOutput(
        layer_1={"baseline_behavior": "normal HTTPS to vendor",
                 "deviation_description": "outbound to unknown host",
                 "confidence_indicator": "Confidence: HIGH"},
        layer_2={"affected_system": "ECG monitor (ICU)",
                 "patient_care_impact": "vitals interruption risk",
                 "phi_exposure": "real-time patient data",
                 "severity_label": "CRITICAL",
                 "severity_rationale": "life-sustaining"},
        layer_3={"immediate_action": "isolate device",
                 "clinical_constraint": "DO NOT disconnect",
                 "escalation_path": "(1) Biomed",
                 "timeframe": "Act within 15 min"},
        **kwargs,
    )


def test_total_word_count_aggregates_three_layers():
    mve = _basic_mve()
    assert mve.total_word_count > 0
    # Sanity: should be sum across all three layers' configured fields.
    l1 = sum(len(v.split()) for v in mve.layer_1.values())
    l2 = sum(len(v.split()) for v in mve.layer_2.values())
    l3 = sum(len(v.split()) for v in mve.layer_3.values())
    assert mve.total_word_count == l1 + l2 + l3


def test_to_dict_required_fields():
    mve = _basic_mve()
    out = mve.to_dict(alert_id="ALERT-1")
    required = {
        "alert_id", "layer_1_why_anomalous", "layer_1", "layer_2", "layer_3",
        "total_word_count", "alert_involves_clinical_system", "provider",
    }
    assert required.issubset(out.keys())


def test_to_dict_includes_provider_default_rule_based():
    """Phase 5 Y10 fix: provider exposed in serialization."""
    mve = _basic_mve()
    out = mve.to_dict()
    assert out["provider"] == "rule_based"


def test_to_dict_includes_provider_openai():
    mve = _basic_mve(provider="openai")
    out = mve.to_dict()
    assert out["provider"] == "openai"


def test_to_dict_includes_provider_anthropic():
    mve = _basic_mve(provider="anthropic")
    out = mve.to_dict()
    assert out["provider"] == "anthropic"


def test_to_dict_layer_1_why_anomalous_concatenates():
    mve = _basic_mve()
    out = mve.to_dict()
    assert "normal HTTPS" in out["layer_1_why_anomalous"]
    assert "outbound to unknown" in out["layer_1_why_anomalous"]
    assert "Confidence: HIGH" in out["layer_1_why_anomalous"]


def test_mve_alert_involves_clinical_system_default():
    mve = _basic_mve()
    assert mve.alert_involves_clinical_system is True


def test_mve_alert_involves_clinical_system_override():
    mve = _basic_mve(alert_involves_clinical_system=False)
    out = mve.to_dict()
    assert out["alert_involves_clinical_system"] is False


# ── ScoredAlert ────────────────────────────────────────────────────────


def test_scored_alert_required_fields():
    s = ScoredAlert(
        adjusted_score=0.85, threshold=0.5,
        should_surface=True, risk_multiplier=1.5,
    )
    assert s.adjusted_score == 0.85
    assert s.suppression_reason is None


def test_scored_alert_with_suppression_reason():
    s = ScoredAlert(
        adjusted_score=0.30, threshold=0.5,
        should_surface=False, risk_multiplier=0.5,
        suppression_reason="maintenance window",
    )
    assert s.suppression_reason == "maintenance window"


# ── AlertGroundTruth + AlertRecord ─────────────────────────────────────


def test_alert_ground_truth_required_fields():
    gt = AlertGroundTruth(
        alert_id="A1", true_severity="CRITICAL",
        true_clinical_system="ECG", true_label="true_positive",
        device_patchable=False, device_criticality="CRITICAL",
    )
    assert gt.alert_id == "A1"
    assert gt.device_patchable is False


def test_alert_record_with_mve():
    gt = AlertGroundTruth(
        alert_id="A1", true_severity="HIGH",
        true_clinical_system="ECG", true_label="true_positive",
        device_patchable=True, device_criticality="HIGH",
    )
    rec = AlertRecord(
        alert_id="A1", raw_alert={}, device_context={},
        behavioral_baseline={}, user_context=None,
        ground_truth=gt, anomaly_score=0.8,
    )
    assert rec.scored is None
    assert rec.mve is None


# ── TestReport ─────────────────────────────────────────────────────────


def test_test_report_shape():
    r = TestReport(metrics=[], negative_tests=[], alignment=[])
    assert r.metrics == []
    assert r.negative_tests == []
    assert r.alignment == []
