"""Tests for module6_evaluation.alerts — curation + ground-truth + device class."""
from __future__ import annotations

import pandas as pd
import pytest

from module6_evaluation.alerts import (
    _build_eval_alert,
    _curate_split_paths,
    _derive_device_class,
    _ground_truth_action,
)


def test_curate_split_paths_test():
    p = _curate_split_paths("test")
    assert p["suffix"] == ""
    assert "risk_npz" in p
    assert "parquet" in p
    assert "analyst" in p
    assert "clinician" in p
    assert "examples" in p


def test_curate_split_paths_demo():
    p = _curate_split_paths("demo")
    assert p["suffix"] == "_demo"


def test_curate_split_paths_strict_validation():
    # split_paths.suffix rejects unknown splits with ValueError.
    with pytest.raises(ValueError):
        _curate_split_paths("staging")


def test_ground_truth_action_benign_dismiss():
    assert _ground_truth_action("LOW", False) == "dismiss"
    assert _ground_truth_action("CRITICAL", False) == "dismiss"


def test_ground_truth_action_critical_isolate():
    assert _ground_truth_action("CRITICAL", True) == "isolate"
    assert _ground_truth_action("HIGH", True) == "isolate"


def test_ground_truth_action_medium_investigate():
    assert _ground_truth_action("MEDIUM", True) == "investigate"


def test_ground_truth_action_low_monitor():
    assert _ground_truth_action("LOW", True) == "monitor"


def test_derive_device_class_returns_string():
    # Build a minimal DataFrame the common.device_class helper can consume.
    # The function signature requires a per-row Series; any "other" fallback
    # is acceptable here — we only care the shim doesn't crash.
    df = pd.DataFrame([{
        "Pulse_Rate": 80.0, "SrcLoad": 0.1, "TotBytes": 1000,
        "ST": 0.01, "DIntPkt": 0.01,
    }])
    out = _derive_device_class(0, df)
    assert isinstance(out, str)


def test_build_eval_alert_required_fields():
    import numpy as np
    R = np.array([0.95])
    levels = np.array(["CRITICAL"])
    y_true = np.array([1])
    attack_cats = np.array(["Spoofing"])
    alert = _build_eval_alert(
        idx=0, R=R, levels=levels, y_true=y_true, attack_cats=attack_cats,
        analyst_by_idx={}, clinician_by_idx={}, examples_by_idx={},
        test_df=None,
    )
    required = {
        "alert_id", "sample_index", "ground_truth", "attack_category",
        "risk_score", "risk_level", "device_class", "device_criticality",
        "affected_system", "patient_care_impact", "active_device",
        "xai_explanation", "correct_action",
    }
    assert required.issubset(alert.keys())
    assert alert["alert_id"] == "EVAL-0000"
    assert alert["ground_truth"] == "attack"
    assert alert["correct_action"] == "isolate"


def test_build_eval_alert_xai_explanation_shape():
    import numpy as np
    alert = _build_eval_alert(
        idx=0, R=np.array([0.5]), levels=np.array(["MEDIUM"]),
        y_true=np.array([0]), attack_cats=np.array(["normal"]),
        analyst_by_idx={}, clinician_by_idx={}, examples_by_idx={},
    )
    xai = alert["xai_explanation"]
    assert "xgboost_top_features" in xai
    assert "dae_top_features" in xai
    assert "consensus" in xai
    assert "clinician_summary" in xai
    assert alert["correct_action"] == "dismiss"


def test_generate_simulated_responses_seeded_reproducible():
    from module6_evaluation.simulated_responses import generate_simulated_responses
    alerts = [
        {"alert_id": f"EVAL-{i:04d}", "correct_action": "isolate",
         "ground_truth": "attack"}
        for i in range(20)
    ]
    r1 = generate_simulated_responses(alerts)
    r2 = generate_simulated_responses(alerts)
    # Seeded ndarray RandomState(42) → identical sequences.
    assert len(r1) == len(r2)
    assert r1[0]["decision_time_sec"] == r2[0]["decision_time_sec"]
    assert r1[100]["likert_trust"] == r2[100]["likert_trust"]


def test_generate_simulated_responses_total_count():
    from module6_evaluation.simulated_responses import generate_simulated_responses
    alerts = [
        {"alert_id": f"EVAL-{i:04d}", "correct_action": "isolate",
         "ground_truth": "attack"}
        for i in range(20)
    ]
    responses = generate_simulated_responses(alerts, n_participants=15)
    assert len(responses) == 15 * 20  # 300 total
    roles = {r["participant_role"] for r in responses}
    assert roles == {"analyst", "clinician", "administrator"}
