"""Research-integrity tests for module6_evaluation.feedback (thematic analysis)."""
from __future__ import annotations

from module6_evaluation.feedback import THEMES, analyze_feedback


def _resp(pid, fb_text="", reclass=None):
    return {
        "participant_id": pid, "participant_role": "analyst",
        "alert_id": f"A-{pid}", "condition": "with_xai",
        "chosen_action": "isolate", "correct_action": "isolate",
        "decision_correct": True,
        "decision_time_sec": 30, "confidence": 4,
        "likert_trust": 4, "likert_usefulness": 4,
        "likert_comprehensibility": 4, "likert_actionability": 4,
        "feedback": fb_text, "reclassification": reclass,
    }


def test_feedback_empty_returns_zero_counts():
    out = analyze_feedback([])
    assert out["total_feedback_texts"] == 0
    assert out["n_reclassifications"] == 0
    assert all(v == 0 for v in out["thematic_counts"].values())


def test_feedback_extracts_more_detail_theme():
    responses = [_resp("P1", "I wanted more detail in the explanation")]
    out = analyze_feedback(responses)
    assert out["thematic_counts"]["wanted_more_detail"] == 1


def test_feedback_extracts_too_technical_theme():
    responses = [_resp("P1", "Too much jargon; very technical")]
    out = analyze_feedback(responses)
    assert out["thematic_counts"]["too_technical"] == 1


def test_feedback_extracts_helpful_theme():
    responses = [_resp("P1", "Very helpful and clear")]
    out = analyze_feedback(responses)
    assert out["thematic_counts"]["nlg_helpful"] == 1


def test_feedback_multi_theme_count():
    responses = [
        _resp("P1", "more detail please"),
        _resp("P2", "too technical for me"),
        _resp("P3", "very helpful"),
    ]
    out = analyze_feedback(responses)
    assert out["thematic_counts"]["wanted_more_detail"] == 1
    assert out["thematic_counts"]["too_technical"] == 1
    assert out["thematic_counts"]["nlg_helpful"] == 1
    assert out["total_feedback_texts"] == 3


def test_feedback_reclassification_count_matches_input():
    responses = [
        _resp("P1", reclass="HIGH"),
        _resp("P2", reclass="MEDIUM"),
        _resp("P3", reclass="HIGH"),
    ]
    out = analyze_feedback(responses)
    assert out["n_reclassifications"] == 3
    assert out["reclassification_counts"]["HIGH"] == 2
    assert out["reclassification_counts"]["MEDIUM"] == 1


def test_feedback_corrections_for_modules_3_5_well_formed():
    responses = [_resp("P1", reclass="HIGH"), _resp("P2", reclass="LOW")]
    out = analyze_feedback(responses)
    assert len(out["corrections_for_modules_3_5"]) == 2
    for c in out["corrections_for_modules_3_5"]:
        assert "alert_id" in c
        assert "suggested_tier" in c
        assert "participant_role" in c


def test_feedback_themes_keys_stable():
    expected_themes = {
        "wanted_more_detail", "nlg_helpful", "too_technical",
        "shap_useful", "trust_concern", "action_unclear",
    }
    assert set(THEMES.keys()) == expected_themes


def test_feedback_recommendations_present():
    responses = [_resp("P1", "more detail")]
    out = analyze_feedback(responses)
    assert len(out["recommendations"]) >= 3


def test_feedback_sample_capped_at_five():
    responses = [_resp(f"P{i}", f"feedback {i}") for i in range(20)]
    out = analyze_feedback(responses)
    assert len(out["sample_feedback"]) == 5
