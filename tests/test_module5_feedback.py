"""Module 5 FeedbackLoop — TP/FP/FN classification + threshold adjustments."""
from __future__ import annotations

from module5_responses.feedback import FeedbackLoop


def test_record_tp_attack_high():
    fl = FeedbackLoop()
    fl.record("A-0", "attack", "HIGH", 0.7, ["isolate_device"])
    assert fl.records[0]["is_tp"] is True
    assert fl.records[0]["is_fp"] is False
    assert fl.records[0]["is_fn"] is False


def test_record_fp_benign_medium():
    fl = FeedbackLoop()
    fl.record("A-1", "benign", "MEDIUM", 0.45, ["log_event"])
    assert fl.records[0]["is_fp"] is True
    assert fl.records[0]["is_tp"] is False


def test_record_fn_attack_low():
    fl = FeedbackLoop()
    fl.record("A-2", "attack", "LOW", 0.1, ["log_event"])
    assert fl.records[0]["is_fn"] is True
    assert fl.records[0]["is_tp"] is False


def test_compute_adjustments_high_fpr_raises_thresholds():
    fl = FeedbackLoop()
    # 1 TP + 4 FP → fpr = 0.8 (>> 10%)
    fl.record("A-0", "attack", "HIGH", 0.7, [])
    for i in range(4):
        fl.record(f"A-{i+1}", "benign", "HIGH", 0.65, [])
    out = fl.compute_adjustments()
    base = {"CRITICAL": 0.80, "HIGH": 0.60, "MEDIUM": 0.40}
    assert out["fpr"] >= 0.79
    assert out["suggested_threshold_change"]["MEDIUM"] > base["MEDIUM"]
    assert out["suggested_threshold_change"]["HIGH"] > base["HIGH"]


def test_compute_adjustments_high_fnr_lowers_thresholds():
    fl = FeedbackLoop()
    # 4 FN + 1 benign → fnr = 0.8
    for i in range(4):
        fl.record(f"A-{i}", "attack", "LOW", 0.2, [])
    fl.record("A-4", "benign", "LOW", 0.05, [])
    out = fl.compute_adjustments()
    base = {"CRITICAL": 0.80, "HIGH": 0.60, "MEDIUM": 0.40}
    assert out["fnr"] >= 0.79
    assert out["suggested_threshold_change"]["MEDIUM"] < base["MEDIUM"]
    assert out["suggested_threshold_change"]["HIGH"] < base["HIGH"]


def test_compute_adjustments_calibrated_when_rates_within_bounds():
    fl = FeedbackLoop()
    # 19 TP + 1 FP + 0 FN → fpr=0.05, fnr=0.0 (calibrated)
    for i in range(19):
        fl.record(f"A-{i}", "attack", "HIGH", 0.8, [])
    fl.record("A-19", "benign", "MEDIUM", 0.5, [])
    out = fl.compute_adjustments()
    metrics = [a["metric"] for a in out["adjustments"]]
    assert "calibrated" in metrics


def test_compute_adjustments_empty_returns_empty():
    fl = FeedbackLoop()
    assert fl.compute_adjustments() == {}


def test_save_load_state_round_trip(tmp_path):
    # Y9 fix: closed-loop labels survive a restart.
    fl = FeedbackLoop()
    fl.record("A-0", "attack", "HIGH", 0.7, ["isolate_device"])
    fl.record("A-1", "benign", "MEDIUM", 0.45, ["log_event"])

    state_path = tmp_path / "feedback_state.json"
    fl.save_state(state_path)
    assert state_path.exists()

    fl2 = FeedbackLoop()
    fl2.load_state(state_path)
    assert len(fl2.records) == 2
    assert fl2.records[0]["alert_id"] == "A-0"
    assert fl2.compute_adjustments()["total_evaluated"] == 2


def test_load_state_missing_file_no_error(tmp_path):
    fl = FeedbackLoop()
    fl.load_state(tmp_path / "does_not_exist.json")
    assert fl.records == []
