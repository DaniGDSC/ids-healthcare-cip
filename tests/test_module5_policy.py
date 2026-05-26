"""Module 5 policy — PolicyEngine + clinical_safety_check decision matrix."""
from __future__ import annotations

import pytest

from module5_responses.policy import PolicyEngine, clinical_safety_check


@pytest.fixture
def engine():
    return PolicyEngine()


def test_critical_tier_includes_isolate_and_high_response(engine):
    # escalate_incident (cost 1.0) gets device-cost-capped at any tier ≤ 0.8.
    # CRITICAL must still produce isolate_device + auto_execute + heavy actions.
    rec = engine.recommend(
        "CRITICAL", device_tier="diagnostic",
        attack_category="Data Alteration", patient_acuity=0.0,
    )
    assert "isolate_device" in rec["actions"]
    assert "forensic_snapshot" in rec["actions"]
    assert "escalate_clinical" in rec["actions"]
    assert rec["auto_execute"] is True
    assert rec["max_response_min"] == 5


def test_low_tier_minimal_actions(engine):
    rec = engine.recommend("LOW", device_tier="auxiliary",
                           attack_category="normal", patient_acuity=0.0)
    assert "log_event" in rec["actions"]
    assert "isolate_device" not in rec["actions"]
    assert rec["auto_execute"] is False


def test_life_sustaining_blocks_isolation(engine):
    rec = engine.recommend(
        "CRITICAL", device_tier="life_sustaining",
        attack_category="Data Alteration", patient_acuity=0.0,
    )
    assert "isolate_device" not in rec["actions"]
    assert "restrict_traffic" in rec["actions"]


def test_elevated_acuity_adds_clinical_escalation(engine):
    rec = engine.recommend(
        "MEDIUM", device_tier="vital_monitoring",
        attack_category="Spoofing", patient_acuity=0.40,
    )
    assert rec["clinical_override"]["triggered"] is True
    assert "escalate_clinical" in rec["actions"]


def test_spoofing_routing_adds_reauth(engine):
    rec = engine.recommend(
        "HIGH", device_tier="diagnostic",
        attack_category="Spoofing", patient_acuity=0.0,
    )
    assert "re_authenticate" in rec["actions"]
    assert rec["primary_notify"] == "IT Security"
    assert rec["secondary_notify"] == "Biomedical Engineering"


def test_data_alteration_routing_adds_forensic(engine):
    rec = engine.recommend(
        "HIGH", device_tier="diagnostic",
        attack_category="Data Alteration", patient_acuity=0.0,
    )
    assert "forensic_snapshot" in rec["actions"]
    assert "escalate_clinical" in rec["actions"]


def test_actions_sorted_by_cost(engine):
    rec = engine.recommend(
        "CRITICAL", device_tier="diagnostic",
        attack_category="Data Alteration", patient_acuity=0.0,
    )
    from module5_responses.config import ACTION_CATALOGUE
    costs = [ACTION_CATALOGUE[a]["cost"] for a in rec["actions"]]
    assert costs == sorted(costs)


def test_requires_approval_set_when_isolate_present(engine):
    rec = engine.recommend(
        "CRITICAL", device_tier="diagnostic",
        attack_category="Spoofing", patient_acuity=0.0,
    )
    assert "isolate_device" in rec["actions"]
    assert rec["requires_approval"] is True


def test_clinical_safety_life_sustaining_elevated_acuity():
    actions = ["log_event", "isolate_device"]
    override = clinical_safety_check("CRITICAL", "life_sustaining", 0.50, actions)
    assert override["triggered"] is True
    assert override["clinical_confirmation_required"] is True
    assert "isolate_device" not in actions
    assert "restrict_traffic" in actions
    assert "escalate_clinical" in actions


def test_clinical_safety_diagnostic_no_override():
    actions = ["log_event", "isolate_device"]
    override = clinical_safety_check("HIGH", "diagnostic", 0.50, actions)
    assert override["triggered"] is False
    # Actions list not mutated.
    assert "isolate_device" in actions


def test_clinical_safety_low_acuity_no_override():
    actions = ["log_event", "isolate_device"]
    override = clinical_safety_check("HIGH", "vital_monitoring", 0.10, actions)
    assert override["triggered"] is False


def test_clinical_safety_vital_monitoring_no_isolate_still_escalates():
    actions = ["log_event", "restrict_traffic"]
    override = clinical_safety_check("MEDIUM", "vital_monitoring", 0.40, actions)
    assert override["triggered"] is True
    # No isolate to downgrade — escalate_clinical still added.
    assert "escalate_clinical" in actions
    assert "restrict_traffic" in actions
