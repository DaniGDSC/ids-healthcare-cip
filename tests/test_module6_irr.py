"""Research-integrity tests for module6_evaluation.irr (Krippendorff α approx)."""
from __future__ import annotations

from module6_evaluation.irr import compute_inter_rater_reliability


def _resp(pid, alert_id, action, trust=4):
    return {
        "participant_id": pid, "participant_role": "analyst",
        "alert_id": alert_id, "condition": "with_xai",
        "chosen_action": action, "correct_action": "isolate",
        "decision_correct": True,
        "decision_time_sec": 30, "confidence": 4,
        "likert_trust": trust, "likert_usefulness": trust,
        "likert_comprehensibility": trust, "likert_actionability": trust,
        "feedback": "", "reclassification": None,
    }


def test_irr_empty_returns_empty():
    assert compute_inter_rater_reliability([]) == {}


def test_irr_perfect_agreement_kappa_one():
    # All 3 raters give the same action on 3 alerts.
    responses = []
    for pid in ("P1", "P2", "P3"):
        for alert in ("A1", "A2", "A3"):
            responses.append(_resp(pid, alert, "isolate"))
    out = compute_inter_rater_reliability(responses)
    # Perfect agreement → Do = 0 → alpha = 1.
    assert out["chosen_action"]["alpha"] == 1.0
    assert out["chosen_action"]["interpretation"] == "good"


def test_irr_systematic_disagreement_low_kappa():
    # 3 raters each picks a different action on every alert.
    actions = ["isolate", "monitor", "dismiss"]
    responses = []
    for j, alert in enumerate(("A1", "A2", "A3")):
        for k, pid in enumerate(("P1", "P2", "P3")):
            responses.append(_resp(pid, alert, actions[k]))
    out = compute_inter_rater_reliability(responses)
    # Complete disagreement → low or negative alpha.
    assert out["chosen_action"]["alpha"] < 0.3


def test_irr_returns_n_coders_and_items():
    responses = []
    for pid in ("P1", "P2", "P3", "P4"):
        for alert in ("A1", "A2"):
            responses.append(_resp(pid, alert, "isolate"))
    out = compute_inter_rater_reliability(responses)
    assert out["chosen_action"]["n_coders"] == 4
    assert out["chosen_action"]["n_items"] == 2


def test_irr_likert_perfect_agreement():
    responses = []
    for pid in ("P1", "P2", "P3"):
        for alert in ("A1", "A2", "A3"):
            responses.append(_resp(pid, alert, "isolate", trust=5))
    out = compute_inter_rater_reliability(responses)
    assert out["likert_trust"]["alpha"] == 1.0


def test_irr_interpretation_thresholds():
    # Perfect agreement → interpretation "good"
    perfect = [_resp(p, a, "isolate") for p in ("P1", "P2") for a in ("A1", "A2")]
    out = compute_inter_rater_reliability(perfect)
    assert out["chosen_action"]["interpretation"] == "good"


def test_irr_handles_uneven_coverage():
    """Krippendorff is robust to missing cells — the function shouldn't crash."""
    responses = [
        _resp("P1", "A1", "isolate"), _resp("P1", "A2", "monitor"),
        _resp("P2", "A1", "isolate"),  # P2 missing A2
        _resp("P3", "A2", "monitor"),  # P3 missing A1
    ]
    out = compute_inter_rater_reliability(responses)
    assert "chosen_action" in out
    assert isinstance(out["chosen_action"]["alpha"], float)
