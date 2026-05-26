"""Module 5 adaptive response selection + audit record builder."""
from __future__ import annotations

import pytest

from module5_responses.adaptive import build_audit_record, select_adaptive_response
from module5_responses.config import ACTION_CATALOGUE


def test_critical_diagnostic_isolates():
    r = select_adaptive_response(
        risk_level="CRITICAL", risk_score=0.92,
        attack_category="Data Alteration", device_tier="diagnostic",
    )
    assert "isolate_device" in r["actions"]
    assert "log_event" in r["actions"]


def test_life_sustaining_capped_at_restrict_traffic():
    r = select_adaptive_response(
        risk_level="CRITICAL", risk_score=0.95,
        attack_category="Data Alteration", device_tier="life_sustaining",
    )
    assert "isolate_device" not in r["actions"]
    assert "restrict_traffic" in r["actions"]
    assert r["device_constraint_applied"] is True


def test_magnitude_escalates_high_score_in_high_tier():
    r = select_adaptive_response(
        risk_level="HIGH", risk_score=0.85,
        attack_category="normal", device_tier="diagnostic",
    )
    assert "isolate_device" in r["actions"]
    assert "forensic_snapshot" in r["actions"]
    assert "Escalated" in r["rationale"]


def test_magnitude_demotes_low_score_in_high_tier():
    r = select_adaptive_response(
        risk_level="HIGH", risk_score=0.20,
        attack_category="normal", device_tier="diagnostic",
    )
    assert "isolate_device" not in r["actions"]
    assert "restrict_traffic" in r["actions"]
    assert "Demoted" in r["rationale"]


def test_unknown_attack_uses_default_routing():
    r = select_adaptive_response(
        risk_level="MEDIUM", risk_score=0.5,
        attack_category="zero_day_xyz", device_tier="diagnostic",
    )
    # Default routing adds restrict_traffic + forensic_snapshot.
    assert r["escalation_chain"]["primary"] == "IT Security"
    assert "restrict_traffic" in r["actions"] or "forensic_snapshot" in r["actions"]


def test_biometric_in_top_adds_clinical_escalation():
    r = select_adaptive_response(
        risk_level="MEDIUM", risk_score=0.5,
        attack_category="normal", device_tier="diagnostic",
        biometric_in_top_features=True,
    )
    assert "escalate_clinical" in r["actions"]


def test_log_event_always_present():
    r = select_adaptive_response(
        risk_level="NORMAL", risk_score=0.05,
        attack_category="normal", device_tier="auxiliary",
    )
    assert r["actions"][0] == "log_event"


def test_actions_sorted_by_cost():
    r = select_adaptive_response(
        risk_level="HIGH", risk_score=0.8,
        attack_category="Data Alteration", device_tier="diagnostic",
    )
    costs = [ACTION_CATALOGUE[a]["cost"] for a in r["actions"]]
    assert costs == sorted(costs)


def test_response_includes_action_descriptions():
    r = select_adaptive_response(
        risk_level="MEDIUM", risk_score=0.5,
        attack_category="Spoofing", device_tier="diagnostic",
    )
    assert len(r["action_descriptions"]) == len(r["actions"])
    assert all(isinstance(d, str) and len(d) > 0 for d in r["action_descriptions"])


# ── build_audit_record ─────────────────────────────────────────────────


@pytest.fixture
def sample_response():
    return select_adaptive_response(
        risk_level="HIGH", risk_score=0.75,
        attack_category="Spoofing", device_tier="diagnostic",
    )


def test_audit_record_required_fields(sample_response):
    rec = build_audit_record(
        idx=42, risk_score=0.75, risk_level="HIGH",
        attack_category="Spoofing", ground_truth="attack",
        response=sample_response, explanation_summary="features X,Y elevated",
    )
    required = {
        "alert_id", "timestamp", "device_tier", "attack_category",
        "risk_score", "risk_level", "recommended_actions",
        "action_rationale", "escalation_chain", "explanation_summary",
        "simulated_outcome", "integrity_hash",
    }
    assert required.issubset(rec.keys())


def test_audit_record_threat_contained_for_attack_with_mitigation(sample_response):
    rec = build_audit_record(
        idx=1, risk_score=0.9, risk_level="HIGH",
        attack_category="Data Alteration", ground_truth="attack",
        response=sample_response, explanation_summary="",
    )
    assert rec["simulated_outcome"]["outcome"] == "threat_contained"
    assert rec["simulated_outcome"]["action_effective"] is True


def test_audit_record_false_positive_for_benign_with_mitigation(sample_response):
    rec = build_audit_record(
        idx=2, risk_score=0.9, risk_level="HIGH",
        attack_category="Data Alteration", ground_truth="benign",
        response=sample_response, explanation_summary="",
    )
    assert rec["simulated_outcome"]["outcome"] == "false_positive_isolated"


def test_audit_record_integrity_hash_is_hex_16():
    response = select_adaptive_response(
        risk_level="LOW", risk_score=0.1,
        attack_category="normal", device_tier="diagnostic",
    )
    rec = build_audit_record(
        idx=0, risk_score=0.1, risk_level="LOW",
        attack_category="normal", ground_truth="benign",
        response=response, explanation_summary="",
    )
    assert len(rec["integrity_hash"]) == 16
    int(rec["integrity_hash"], 16)  # hex parse must succeed


def test_audit_record_explanation_summary_truncated_at_200(sample_response):
    long_summary = "x" * 500
    rec = build_audit_record(
        idx=0, risk_score=0.5, risk_level="MEDIUM",
        attack_category="Spoofing", ground_truth="benign",
        response=sample_response, explanation_summary=long_summary,
    )
    assert len(rec["explanation_summary"]) == 200
