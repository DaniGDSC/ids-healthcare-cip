"""Module 6 forms — process_alert + assign_ab_conditions + build_fda_record_for_alert."""
from __future__ import annotations

from module6_evaluation.forms import (
    assign_ab_conditions,
    build_fda_record_for_alert,
    process_alert,
)


# ── assign_ab_conditions ───────────────────────────────────────────────


def test_assign_ab_conditions_balanced():
    out = assign_ab_conditions(10, "P01")
    assert len(out) == 10
    assert sum(out) == 5  # half True
    assert sum(1 for x in out if not x) == 5


def test_assign_ab_conditions_odd_count():
    out = assign_ab_conditions(11, "P02")
    assert len(out) == 11
    # 5 True + 6 False (n_xai = n//2 = 5)
    assert sum(out) == 5


def test_assign_ab_conditions_seeded_reproducible():
    a = assign_ab_conditions(20, "P03")
    b = assign_ab_conditions(20, "P03")
    assert a == b


def test_assign_ab_conditions_differs_per_participant():
    a = assign_ab_conditions(20, "P01")
    b = assign_ab_conditions(20, "P02")
    # Different SHA seeds → different orderings (overwhelmingly likely).
    assert a != b


def test_assign_ab_conditions_empty():
    assert assign_ab_conditions(0, "P01") == []


# ── process_alert ──────────────────────────────────────────────────────


def test_process_alert_returns_required_fields():
    out = process_alert(42, {
        "alert_id": "ALERT-42",
        "risk_level": "HIGH",
        "risk_score": 0.85,
        "attack_category": "Spoofing",
        "ground_truth": "attack",
        "response": {
            "actions": ["log_event", "isolate_device"],
            "device_tier": "diagnostic",
        },
    })
    required = {
        "sample_index", "alert_id", "risk_level", "risk_score",
        "attack_category", "ground_truth", "actions", "operator_bucket",
        "device_tier",
    }
    assert required.issubset(out.keys())


def test_process_alert_isolate_bucket():
    out = process_alert(1, {
        "response": {"actions": ["isolate_device", "log_event"]},
    })
    assert out["operator_bucket"] == "isolate"


def test_process_alert_escalate_bucket():
    out = process_alert(1, {
        "response": {"actions": ["escalate_incident", "log_event"]},
    })
    assert out["operator_bucket"] == "escalate"


def test_process_alert_investigate_bucket():
    out = process_alert(1, {
        "response": {"actions": ["restrict_traffic", "log_event"]},
    })
    assert out["operator_bucket"] == "investigate"


def test_process_alert_monitor_bucket_default():
    out = process_alert(1, {"response": {"actions": ["log_event"]}})
    assert out["operator_bucket"] == "monitor"


def test_process_alert_empty_actions():
    out = process_alert(1, {"response": {"actions": []}})
    assert out["operator_bucket"] == "monitor"
    assert out["actions"] == []


def test_process_alert_isolate_beats_escalate():
    out = process_alert(1, {
        "response": {"actions": ["escalate_incident", "isolate_device"]},
    })
    # isolate > escalate per bucket_rank.
    assert out["operator_bucket"] == "isolate"


def test_process_alert_default_alert_id():
    out = process_alert(99, {})
    assert out["alert_id"] == "ALERT-00099"
    assert out["operator_bucket"] == "monitor"


# ── build_fda_record_for_alert ─────────────────────────────────────────


def test_build_fda_record_required_fields():
    rec = build_fda_record_for_alert(
        {"alert_id": "ALERT-1", "sample_index": 1, "risk_level": "CRITICAL",
         "attack_category": "Spoofing", "ground_truth": "attack",
         "response": {"actions": ["isolate_device"]}},
        participant_id="P03", role="analyst",
        chosen_action="isolate", rationale="Clear spoof",
        confidence=4, decision_time_sec=12.5,
    )
    assert rec["alert_id"] == "ALERT-1"
    assert rec["reviewer"]["participant_id"] == "P03"
    assert rec["reviewer"]["role"] == "analyst"
    assert "decided_at" in rec["reviewer"]
    assert rec["decision"]["chosen_action"] == "isolate"
    assert rec["decision"]["confidence"] == 4
    assert rec["decision"]["decision_time_sec"] == 12.5
    assert rec["policy_actions"] == ["isolate_device"]


def test_build_fda_record_truncates_long_rationale():
    long_text = "x" * 1000
    rec = build_fda_record_for_alert(
        {"alert_id": "ALERT-2", "risk_level": "LOW"},
        participant_id="P01", role="clinician",
        chosen_action="monitor", rationale=long_text,
    )
    assert len(rec["decision"]["rationale"]) == 500


def test_build_fda_record_handles_missing_optionals():
    rec = build_fda_record_for_alert(
        {"alert_id": "ALERT-3"}, participant_id="P01",
        role="administrator", chosen_action="dismiss",
    )
    assert rec["decision"]["confidence"] is None
    assert rec["decision"]["decision_time_sec"] is None
    assert rec["policy_actions"] == []
