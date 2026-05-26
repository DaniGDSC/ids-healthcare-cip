"""Research-integrity tests for module6_evaluation.metrics."""
from __future__ import annotations


from module6_evaluation.metrics import compute_evaluation_metrics


def _resp(pid, role, alert_id, condition, correct, time_sec=30, conf=4,
          trust=4, useful=4, comp=4, action=4):
    return {
        "participant_id": pid, "participant_role": role,
        "alert_id": alert_id, "condition": condition,
        "chosen_action": "isolate", "correct_action": "isolate",
        "decision_correct": correct,
        "decision_time_sec": time_sec, "confidence": conf,
        "likert_trust": trust, "likert_usefulness": useful,
        "likert_comprehensibility": comp, "likert_actionability": action,
        "feedback": "", "reclassification": None,
    }


def test_metrics_empty_returns_zeros():
    m = compute_evaluation_metrics([])
    assert m["n_participants"] == 0
    assert m["n_alerts"] == 0
    assert m["with_xai"]["decision_accuracy"] == 0.0
    assert m["without_xai"]["decision_accuracy"] == 0.0


def test_metrics_perfect_with_xai_accuracy():
    responses = [_resp(f"P{i}", "analyst", f"A{j}", "with_xai", True)
                 for i in range(3) for j in range(4)]
    m = compute_evaluation_metrics(responses)
    assert m["with_xai"]["decision_accuracy"] == 1.0


def test_metrics_zero_without_xai_accuracy():
    responses = [_resp(f"P{i}", "analyst", f"A{j}", "without_xai", False)
                 for i in range(3) for j in range(4)]
    m = compute_evaluation_metrics(responses)
    assert m["without_xai"]["decision_accuracy"] == 0.0


def test_metrics_balanced_50_50():
    responses = (
        [_resp(f"P{i}", "analyst", f"A{j}", "with_xai", True)
         for i in range(2) for j in range(4)] +
        [_resp(f"P{i}", "analyst", f"A{j}", "with_xai", False)
         for i in range(2) for j in range(4, 8)]
    )
    m = compute_evaluation_metrics(responses)
    assert m["with_xai"]["decision_accuracy"] == 0.5


def test_metrics_n_participants_unique():
    responses = (
        [_resp(f"P{i}", "analyst", f"A{j}", "with_xai", True)
         for i in range(5) for j in range(4)]
    )
    m = compute_evaluation_metrics(responses)
    assert m["n_participants"] == 5
    assert m["n_alerts"] == 4
    assert m["n_responses"] == 20


def test_metrics_mean_decision_time_known():
    responses = [
        _resp("P1", "analyst", "A1", "with_xai", True, time_sec=20),
        _resp("P1", "analyst", "A2", "with_xai", True, time_sec=40),
        _resp("P2", "analyst", "A1", "with_xai", True, time_sec=30),
        _resp("P2", "analyst", "A2", "with_xai", True, time_sec=30),
    ]
    m = compute_evaluation_metrics(responses)
    assert m["with_xai"]["mean_decision_time_sec"] == 30.0


def test_metrics_per_role_aggregation():
    responses = (
        [_resp(f"P{i}", "analyst", f"A{j}", "with_xai", True) for i in range(2) for j in range(3)] +
        [_resp(f"P{i}", "clinician", f"A{j}", "with_xai", False) for i in range(2) for j in range(3)]
    )
    m = compute_evaluation_metrics(responses)
    assert "analyst" in m["per_role"]
    assert "clinician" in m["per_role"]
    assert m["per_role"]["analyst"]["with_xai_accuracy"] == 1.0
    assert m["per_role"]["clinician"]["with_xai_accuracy"] == 0.0


def test_metrics_likert_means():
    responses = [
        _resp("P1", "analyst", "A1", "with_xai", True, trust=5, useful=5, comp=5, action=5),
        _resp("P2", "analyst", "A1", "with_xai", True, trust=3, useful=3, comp=3, action=3),
    ]
    m = compute_evaluation_metrics(responses)
    assert m["with_xai"]["likert_trust"] == 4.0
    assert m["with_xai"]["likert_usefulness"] == 4.0


def test_metrics_condition_isolation():
    """Without-XAI metrics must not be polluted by with-XAI rows."""
    responses = [
        _resp("P1", "analyst", "A1", "with_xai", True, time_sec=10),
        _resp("P1", "analyst", "A2", "without_xai", True, time_sec=60),
    ]
    m = compute_evaluation_metrics(responses)
    assert m["with_xai"]["mean_decision_time_sec"] == 10.0
    assert m["without_xai"]["mean_decision_time_sec"] == 60.0
