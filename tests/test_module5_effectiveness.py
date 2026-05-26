"""Module 5 effectiveness + response stats aggregation."""
from __future__ import annotations

from module5_responses.effectiveness import (
    compute_effectiveness,
    compute_response_stats,
)


def _audit(idx, outcome, actions, gt):
    return {
        "alert_id": f"A-{idx}",
        "recommended_actions": actions,
        "simulated_outcome": {"outcome": outcome, "ground_truth": gt},
    }


def test_effectiveness_outcome_distribution():
    audits = [
        _audit(0, "threat_contained", ["isolate_device"], "attack"),
        _audit(1, "threat_contained", ["isolate_device"], "attack"),
        _audit(2, "false_positive_isolated", ["isolate_device"], "benign"),
        _audit(3, "benign_logged", ["log_event"], "benign"),
    ]
    e = compute_effectiveness(audits)
    assert e["outcome_distribution"]["threat_contained"] == 2
    assert e["outcome_distribution"]["false_positive_isolated"] == 1
    assert e["outcome_distribution"]["benign_logged"] == 1


def test_effectiveness_over_response_rate():
    audits = [
        _audit(0, "threat_contained", ["isolate_device"], "attack"),
        _audit(1, "false_positive_isolated", ["isolate_device"], "benign"),
    ]
    e = compute_effectiveness(audits)
    assert e["over_response_count"] == 1
    assert e["over_response_rate"] == 0.5


def test_effectiveness_under_response_rate():
    audits = [
        _audit(0, "threat_logged_not_mitigated", ["log_event"], "attack"),
        _audit(1, "benign_logged", ["log_event"], "benign"),
    ]
    e = compute_effectiveness(audits)
    assert e["under_response_count"] == 1
    assert e["under_response_rate"] == 0.5


def test_effectiveness_empty_records():
    e = compute_effectiveness([])
    assert e["over_response_rate"] == 0
    assert e["under_response_rate"] == 0
    # defaultdict touches for over/under counts so the dict may carry zero
    # entries — assert all values are zero rather than dict is empty.
    assert all(v == 0 for v in e["outcome_distribution"].values())


def test_effectiveness_proportionality_sorted_by_cost_desc():
    audits = [
        _audit(0, "threat_contained", ["isolate_device", "log_event"], "attack"),
        _audit(1, "false_positive_isolated", ["isolate_device", "log_event"], "benign"),
    ]
    e = compute_effectiveness(audits)
    actions = [p["action"] for p in e["proportionality_analysis"]]
    # isolate (cost 0.8) before log_event (cost 0.1)
    assert actions.index("isolate_device") < actions.index("log_event")


# ── compute_response_stats ─────────────────────────────────────────────


def _record(idx, level, gt, actions):
    return {
        "sample_index": idx,
        "risk_level": level,
        "ground_truth": gt,
        "response": {"actions": actions},
    }


def test_response_stats_precision_by_level():
    records = [
        _record(0, "HIGH", "attack", ["log_event", "isolate_device"]),
        _record(1, "HIGH", "attack", ["log_event", "isolate_device"]),
        _record(2, "HIGH", "benign", ["log_event"]),
        _record(3, "LOW", "benign", ["log_event"]),
    ]
    s = compute_response_stats(records)
    # HIGH: 2 attack / 3 total → 0.6667
    assert s["precision_by_level"]["HIGH"] == round(2 / 3, 4)
    assert s["precision_by_level"]["LOW"] == 0.0


def test_response_stats_action_counts():
    records = [
        _record(0, "HIGH", "attack", ["log_event", "isolate_device"]),
        _record(1, "HIGH", "attack", ["log_event"]),
    ]
    s = compute_response_stats(records)
    assert s["actions_triggered"]["log_event"] == 2
    assert s["actions_triggered"]["isolate_device"] == 1


def test_response_stats_empty():
    s = compute_response_stats([])
    assert s["total_alerts"] == 0
    assert s["alerts_by_level"] == {}
