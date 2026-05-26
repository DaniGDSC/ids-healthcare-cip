"""Research-integrity tests for module6_evaluation.statistical (Wilcoxon + Cohen's d)."""
from __future__ import annotations


from module6_evaluation.statistical import statistical_analysis


def _resp(pid, condition, accuracy, time_sec=30, trust=4):
    return {
        "participant_id": pid, "participant_role": "analyst",
        "alert_id": f"A-{condition}-{pid}",
        "condition": condition,
        "chosen_action": "isolate", "correct_action": "isolate",
        "decision_correct": bool(accuracy),
        "decision_time_sec": time_sec, "confidence": 3,
        "likert_trust": trust, "likert_usefulness": trust,
        "likert_comprehensibility": trust, "likert_actionability": trust,
        "feedback": "", "reclassification": None,
    }


def test_statistical_empty_returns_empty_dict():
    assert statistical_analysis([]) == {}


def test_statistical_no_difference_p_near_one():
    # Identical groups for n=5 participants → p ≈ 1.0 (or skipped via diff==0).
    responses = []
    for i in range(5):
        responses.append(_resp(f"P{i}", "with_xai", True, time_sec=30, trust=4))
        responses.append(_resp(f"P{i}", "without_xai", True, time_sec=30, trust=4))
    out = statistical_analysis(responses)
    # decision_correct has zero variance → caller short-circuits to p=1
    if "decision_correct" in out:
        assert out["decision_correct"]["p_value"] >= 0.99


def test_statistical_large_effect_detected():
    # Need CROSS-PARTICIPANT variance in per-participant means so pooled_std > 0.
    # Each participant has a different baseline accuracy, but the with-XAI
    # condition uniformly lifts everyone by ~0.4. Large effect.
    import random
    rng = random.Random(7)
    responses = []
    for i in range(15):
        # Each participant has a different baseline (jittered around 0.5).
        baseline = 0.3 + 0.03 * i  # 0.3 .. 0.72
        for j in range(8):
            # without_xai: baseline accuracy
            responses.append(_resp(f"P{i}", "without_xai",
                                   rng.random() < baseline))
            # with_xai: baseline + 0.4 (capped at 1.0)
            responses.append(_resp(f"P{i}", "with_xai",
                                   rng.random() < min(1.0, baseline + 0.4)))
    out = statistical_analysis(responses)
    assert "decision_correct" in out
    assert out["decision_correct"]["p_value"] < 0.05
    assert out["decision_correct"]["significant"] is True
    assert abs(out["decision_correct"]["cohens_d"]) > 0.8


def test_statistical_skips_when_fewer_than_three_participants():
    responses = []
    for i in range(2):  # only 2 participants — must be skipped (< 3 paired)
        responses.append(_resp(f"P{i}", "with_xai", True))
        responses.append(_resp(f"P{i}", "without_xai", False))
    out = statistical_analysis(responses)
    assert out == {} or "decision_correct" not in out


def test_statistical_cohens_d_sign():
    """Cohen's d should be positive when with_xai > without_xai.

    Use varied baselines so cross-participant variance > 0 in the
    per-participant mean arrays Cohen's d operates on.
    """
    import random
    rng = random.Random(11)
    responses = []
    for i in range(15):
        # Each participant baselines around different accuracies + trust levels.
        base_acc = 0.3 + 0.04 * i
        base_trust = 1 + (i % 3)  # 1, 2, or 3 baseline
        for j in range(4):
            responses.append(_resp(f"P{i}", "with_xai",
                                   rng.random() < min(1.0, base_acc + 0.4),
                                   trust=min(5, base_trust + 3)))
            responses.append(_resp(f"P{i}", "without_xai",
                                   rng.random() < base_acc,
                                   trust=base_trust))
    out = statistical_analysis(responses)
    assert out["decision_correct"]["cohens_d"] > 0
    assert out["likert_trust"]["cohens_d"] > 0


def test_statistical_effect_size_classification():
    """Strong mean separation with cross-participant variance > 0 →
    effect_size should classify as ``large``.
    """
    import random
    rng = random.Random(13)
    responses = []
    for i in range(15):
        base_acc = 0.2 + 0.04 * i
        for j in range(4):
            responses.append(_resp(f"P{i}", "with_xai",
                                   rng.random() < min(1.0, base_acc + 0.5),
                                   trust=4 + (i % 2)))
            responses.append(_resp(f"P{i}", "without_xai",
                                   rng.random() < base_acc,
                                   trust=1 + (i % 2)))
    out = statistical_analysis(responses)
    sizes = {out[m]["effect_size"] for m in out if "effect_size" in out[m]}
    assert "large" in sizes


def test_statistical_returns_all_measures_when_data_sufficient():
    responses = []
    import random
    rng = random.Random(42)
    for i in range(15):
        responses.append(_resp(f"P{i}", "with_xai", rng.random() < 0.9,
                               time_sec=int(rng.gauss(28, 5)), trust=rng.randint(3, 5)))
        responses.append(_resp(f"P{i}", "without_xai", rng.random() < 0.7,
                               time_sec=int(rng.gauss(45, 8)), trust=rng.randint(2, 4)))
    out = statistical_analysis(responses)
    expected = {"decision_correct", "decision_time_sec", "confidence",
                "likert_trust", "likert_usefulness",
                "likert_comprehensibility", "likert_actionability"}
    assert expected.issubset(set(out.keys()))
