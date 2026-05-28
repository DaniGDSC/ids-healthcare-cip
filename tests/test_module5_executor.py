"""Module 5 executor + notification service."""
from __future__ import annotations

from datetime import datetime

import pytest

from module5_responses.executor import ActionExecutor, NotificationService


@pytest.fixture
def ts():
    return datetime(2026, 5, 1, 12, 0, 0)


def test_action_executor_attack_with_isolation_contained(ts):
    ex = ActionExecutor()
    rec = ex.execute(
        "A-001", 0, ["log_event", "isolate_device"],
        recommendation={"auto_execute_recommended": True, "requires_approval": True,
                        "clinical_override": {"triggered": False}},
        ground_truth="attack", timestamp=ts,
    )
    assert rec["outcome"] == "threat_contained"
    assert rec["effective"] is True


def test_action_executor_attack_without_mitigation_logged(ts):
    ex = ActionExecutor()
    rec = ex.execute(
        "A-002", 1, ["log_event"],
        recommendation={"auto_execute_recommended": False, "requires_approval": False,
                        "clinical_override": {"triggered": False}},
        ground_truth="attack", timestamp=ts,
    )
    assert rec["outcome"] == "threat_logged_not_mitigated"
    assert rec["effective"] is False


def test_action_executor_benign_with_isolation_false_positive(ts):
    ex = ActionExecutor()
    rec = ex.execute(
        "A-003", 2, ["isolate_device"],
        recommendation={"auto_execute_recommended": False, "requires_approval": True,
                        "clinical_override": {"triggered": False}},
        ground_truth="benign", timestamp=ts,
    )
    assert rec["outcome"] == "false_positive_isolated"
    assert rec["effective"] is False


def test_action_executor_benign_logged(ts):
    ex = ActionExecutor()
    rec = ex.execute(
        "A-004", 3, ["log_event"],
        recommendation={"auto_execute_recommended": False, "requires_approval": False,
                        "clinical_override": {"triggered": False}},
        ground_truth="benign", timestamp=ts,
    )
    assert rec["outcome"] == "benign_logged"
    assert rec["effective"] is True


def test_action_executor_accumulates_log(ts):
    ex = ActionExecutor()
    for i in range(3):
        ex.execute(
            f"A-{i:03d}", i, ["log_event"],
            recommendation={"auto_execute_recommended": False, "requires_approval": False,
                            "clinical_override": {"triggered": False}},
            ground_truth="benign", timestamp=ts,
        )
    assert len(ex.execution_log) == 3


def test_notification_primary_only():
    n = NotificationService()
    msgs = n.notify(
        sample_index=42,
        alert_tier="HIGH",
        recommendation={
            "primary_notify": "IT Security",
            "secondary_notify": None,
            "actions": ["log_event", "isolate_device"],
        },
        clinician_summary="",
        analyst_top_features=[{"feature": "Pulse_Rate"}],
        risk_score=0.85,
    )
    assert len(msgs) == 1
    assert msgs[0]["recipient"] == "IT Security"
    assert msgs[0]["priority"] == "HIGH"


def test_notification_clinical_when_escalate_clinical_in_actions():
    n = NotificationService()
    msgs = n.notify(
        sample_index=7,
        alert_tier="CRITICAL",
        recommendation={
            "primary_notify": "IT Security",
            "secondary_notify": "Charge Nurse",
            "actions": ["log_event", "escalate_clinical"],
        },
        clinician_summary="Vitals appear erratic; review monitor reading.",
        analyst_top_features=[{"feature": "Pulse_Rate"}],
        risk_score=0.95,
    )
    recipients = [m["recipient"] for m in msgs]
    assert "Clinical Staff" in recipients
    assert "Charge Nurse" in recipients


def test_notification_skips_secondary_when_none():
    n = NotificationService()
    msgs = n.notify(
        sample_index=1, alert_tier="LOW",
        recommendation={
            "primary_notify": "IT Security",
            "secondary_notify": None,
            "actions": ["log_event"],
        },
        clinician_summary="", analyst_top_features=[], risk_score=0.1,
    )
    assert all(m["recipient"] != "Clinical Staff" for m in msgs)
    assert len(msgs) == 1


def test_notification_accumulates_across_calls():
    n = NotificationService()
    for _ in range(3):
        n.notify(
            sample_index=0, alert_tier="LOW",
            recommendation={
                "primary_notify": "IT Security",
                "secondary_notify": None,
                "actions": ["log_event"],
            },
            clinician_summary="", analyst_top_features=[], risk_score=0.1,
        )
    assert len(n.notifications) == 3
