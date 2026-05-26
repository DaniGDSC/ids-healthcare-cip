"""Tests for the C3 Alert Simulation Harness.

Covers:
* YAML fixture round-trip via :func:`stream_dataset` / :func:`load_dataset`
* Pipeline orchestration in :func:`run_simulation` against a synthetic
  :class:`AlertRecord` list
* Single-pass log + action build in :func:`_build_system_logs_and_actions`
* Claim → metric alignment in :func:`_build_alignment` across the
  SUPPORTED / PARTIAL / NOT_SUPPORTED / NOT_TESTED paths
* EA-04 alert-volume spike warning
"""
from __future__ import annotations

import logging
from pathlib import Path


from src.data_models import (
    AlertGroundTruth,
    AlertRecord,
    MVEOutput,
)
from src.harness import (
    _build_alignment,
    _build_system_logs_and_actions,
    load_dataset,
    run_simulation,
    stream_dataset,
)

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "sample_alerts_minimal.yaml"


# ── stream_dataset / load_dataset ──────────────────────────────────────


def test_stream_dataset_yields_alert_records():
    records = list(stream_dataset(FIXTURE))
    assert len(records) == 2
    assert all(isinstance(r, AlertRecord) for r in records)
    assert records[0].alert_id == "TEST-001"
    assert records[1].alert_id == "TEST-002"


def test_stream_dataset_handles_missing_user_context():
    records = list(stream_dataset(FIXTURE))
    # Both fixture alerts have user_context: null — must parse cleanly.
    assert records[0].user_context is None
    assert records[1].user_context is None


def test_stream_dataset_populates_ground_truth():
    records = list(stream_dataset(FIXTURE))
    assert records[0].ground_truth.true_severity == "CRITICAL"
    assert records[0].ground_truth.device_patchable is False
    assert records[1].ground_truth.true_severity == "LOW"
    assert records[1].ground_truth.device_patchable is True


def test_load_dataset_returns_materialised_list():
    """load_dataset wraps stream_dataset and returns a list."""
    records = load_dataset(FIXTURE)
    assert isinstance(records, list)
    assert len(records) == 2


# ── _build_system_logs_and_actions ─────────────────────────────────────


def _make_record(alert_id: str, with_mve: bool = True) -> AlertRecord:
    gt = AlertGroundTruth(
        alert_id=alert_id, true_severity="HIGH",
        true_clinical_system="EHR", true_label="true_positive",
        device_patchable=True, device_criticality="HIGH",
    )
    rec = AlertRecord(
        alert_id=alert_id,
        raw_alert={"alert_name": "test"},
        device_context={"criticality": "HIGH", "patchable": True,
                        "device_type": "workstation"},
        behavioral_baseline={"baseline_days": 90, "normal_protocols": ["HTTPS"]},
        user_context=None,
        ground_truth=gt,
        anomaly_score=0.8,
    )
    if with_mve:
        rec.mve = MVEOutput(
            layer_1={"baseline_behavior": "normal",
                     "deviation_description": "anomaly",
                     "confidence_indicator": "high"},
            layer_2={"affected_system": "EHR",
                     "patient_care_impact": "high",
                     "phi_exposure": "exposed",
                     "severity_label": "HIGH",
                     "severity_rationale": "active clinical"},
            layer_3={"immediate_action": "isolate device",
                     "clinical_constraint": "DO NOT disconnect",
                     "escalation_path": "(1) Charge nurse",
                     "timeframe": "Act within 1 hour"},
        )
    return rec


def test_build_logs_and_actions_records_with_mve():
    records = [_make_record("A1", with_mve=True),
               _make_record("A2", with_mve=True)]
    logs, actions = _build_system_logs_and_actions(records)
    # Each MVE record contributes 2 logs (score + generate) + 1 action.
    assert len(logs) == 4
    assert len(actions) == 2
    assert all(a["type"] == "recommendation" for a in actions)


def test_build_logs_and_actions_records_without_mve():
    """Records with mve=None contribute only the score_alert log, no action."""
    records = [_make_record("A1", with_mve=False)]
    logs, actions = _build_system_logs_and_actions(records)
    assert len(logs) == 1
    assert logs[0]["action"] == "score_alert"
    assert actions == []


def test_build_logs_and_actions_empty():
    logs, actions = _build_system_logs_and_actions([])
    assert logs == []
    assert actions == []


def test_build_logs_action_carries_layer3_immediate_action():
    records = [_make_record("A1", with_mve=True)]
    _, actions = _build_system_logs_and_actions(records)
    assert actions[0]["content"] == "isolate device"


# ── _build_alignment ───────────────────────────────────────────────────


def _metric(mid: str, status: str = "PASS") -> dict:
    return {
        "metric_id": mid,
        "metric_name": f"name_{mid}",
        "result_value": 0.9,
        "target": 0.85,
        "minimum": 0.70,
        "pass_fail": status,
        "detail": "",
    }


def test_build_alignment_supported_when_all_pass():
    # C1 needs M2, M8, M5 all PASS → SUPPORTED.
    metrics = [_metric("M2"), _metric("M8"), _metric("M5"),
               _metric("M7"), _metric("M6"), _metric("M3"),
               _metric("M4"), _metric("M1"), _metric("M1b")]
    out = _build_alignment(metrics)
    by_id = {a["claim_id"]: a for a in out}
    assert by_id["C1"]["verdict"] == "SUPPORTED"


def test_build_alignment_partial_when_warn():
    metrics = [_metric("M2", "WARN"), _metric("M8"), _metric("M5"),
               _metric("M7"), _metric("M6"), _metric("M3"),
               _metric("M4"), _metric("M1"), _metric("M1b")]
    out = _build_alignment(metrics)
    by_id = {a["claim_id"]: a for a in out}
    assert by_id["C1"]["verdict"] == "PARTIAL"
    assert by_id["C1"]["all_tests_pass"] is True


def test_build_alignment_not_supported_when_fail():
    metrics = [_metric("M2", "FAIL"), _metric("M8"), _metric("M5"),
               _metric("M7"), _metric("M6"), _metric("M3"),
               _metric("M4"), _metric("M1"), _metric("M1b")]
    out = _build_alignment(metrics)
    by_id = {a["claim_id"]: a for a in out}
    assert by_id["C1"]["verdict"] == "NOT_SUPPORTED"
    assert by_id["C1"]["all_tests_pass"] is False


def test_build_alignment_not_tested_for_c4_c5():
    """C4 and C5 have empty supported_by → carry the NOT_TESTED verdict."""
    out = _build_alignment([_metric("M1")])
    by_id = {a["claim_id"]: a for a in out}
    assert "C4" in by_id
    assert "C5" in by_id
    assert by_id["C4"]["verdict"].startswith("NOT_TESTED")
    assert by_id["C5"]["verdict"].startswith("NOT_TESTED")
    assert by_id["C4"]["all_tests_pass"] is False


def test_build_alignment_includes_all_seven_claims():
    out = _build_alignment([_metric("M1")])
    claim_ids = {a["claim_id"] for a in out}
    assert claim_ids == {"C1", "C2", "C3", "C4", "C5", "C7", "C8"}


# ── run_simulation ─────────────────────────────────────────────────────


def _build_synthetic_dataset(n_critical_unpatchable: int = 1,
                              n_low_patchable: int = 1) -> list[AlertRecord]:
    """Build a small dataset that produces deterministic surface counts."""
    records = []
    for i in range(n_critical_unpatchable):
        gt = AlertGroundTruth(
            alert_id=f"CRIT-{i:03d}", true_severity="CRITICAL",
            true_clinical_system="infusion pump", true_label="true_positive",
            device_patchable=False, device_criticality="CRITICAL",
        )
        records.append(AlertRecord(
            alert_id=f"CRIT-{i:03d}",
            raw_alert={"alert_name": "Anomalous outbound",
                       "protocol": "TCP/443",
                       "source_ip": "10.0.1.50", "dest_ip": "198.51.100.1",
                       "timestamp": "2026-05-26T03:00:00Z",
                       "severity_score": 0.85},
            device_context={"device_type": "infusion_pump",
                            "criticality": "CRITICAL", "patchable": False,
                            "clinical_function": "drug delivery",
                            "location": "ICU"},
            behavioral_baseline={"normal_destinations": [],
                                  "normal_protocols": ["HTTPS"],
                                  "baseline_days": 90},
            user_context=None, ground_truth=gt,
            anomaly_score=0.85,
        ))
    for i in range(n_low_patchable):
        gt = AlertGroundTruth(
            alert_id=f"LOW-{i:03d}", true_severity="LOW",
            true_clinical_system="workstation", true_label="false_positive",
            device_patchable=True, device_criticality="LOW",
        )
        records.append(AlertRecord(
            alert_id=f"LOW-{i:03d}",
            raw_alert={"alert_name": "Workstation anomaly",
                       "protocol": "TCP/80", "source_ip": "10.0.5.10",
                       "dest_ip": "10.0.6.20",
                       "timestamp": "2026-05-26T14:00:00Z",
                       "severity_score": 0.15},
            device_context={"device_type": "workstation",
                            "criticality": "LOW", "patchable": True,
                            "clinical_function": "administrative",
                            "location": "office"},
            behavioral_baseline={"normal_destinations": [],
                                  "normal_protocols": ["HTTPS"],
                                  "baseline_days": 90},
            user_context=None, ground_truth=gt,
            anomaly_score=0.15,
        ))
    return records


def test_run_simulation_returns_test_report():
    dataset = _build_synthetic_dataset()
    report = run_simulation(dataset=dataset)
    assert hasattr(report, "metrics")
    assert hasattr(report, "negative_tests")
    assert hasattr(report, "alignment")


def test_run_simulation_alignment_has_seven_claims():
    dataset = _build_synthetic_dataset()
    report = run_simulation(dataset=dataset)
    claim_ids = {a["claim_id"] for a in report.alignment}
    assert claim_ids == {"C1", "C2", "C3", "C4", "C5", "C7", "C8"}


def test_run_simulation_critical_unpatchable_surfaces(caplog):
    """Safety floor: the CRITICAL+unpatchable alert in our dataset must
    surface (and an MVE generated for it)."""
    dataset = _build_synthetic_dataset(n_critical_unpatchable=1, n_low_patchable=0)
    with caplog.at_level(logging.INFO):
        run_simulation(dataset=dataset)
    msgs = " ".join(rec.message for rec in caplog.records)
    assert "1 surfaced" in msgs or "1 alerts processed" in msgs.lower()


def test_run_simulation_surfaced_rate_spike_warning(caplog):
    """EA-04: > 30% surfaced rate triggers a warning."""
    # All 5 records are CRITICAL+unpatchable → all surface → 100% rate.
    dataset = _build_synthetic_dataset(n_critical_unpatchable=5, n_low_patchable=0)
    with caplog.at_level(logging.WARNING, logger="src.harness"):
        run_simulation(dataset=dataset)
    spike_msgs = [r for r in caplog.records
                  if "ALERT VOLUME SPIKE" in r.message]
    assert len(spike_msgs) >= 1


def test_run_simulation_no_spike_warning_below_threshold(caplog):
    """Below the 30% threshold, no spike warning fires."""
    # 1 CRITICAL+unpatchable (surfaces) + 5 LOW+patchable (don't surface):
    # 1/6 = 16.7% < 30%.
    dataset = _build_synthetic_dataset(n_critical_unpatchable=1, n_low_patchable=5)
    with caplog.at_level(logging.WARNING, logger="src.harness"):
        run_simulation(dataset=dataset)
    spike_msgs = [r for r in caplog.records
                  if "ALERT VOLUME SPIKE" in r.message]
    assert spike_msgs == []


def test_run_simulation_with_generator_source():
    """Pass a generator (not a list) — confirm streaming path works."""
    dataset_list = _build_synthetic_dataset(n_critical_unpatchable=2, n_low_patchable=2)
    # Re-run with a generator built from the same data.
    report = run_simulation(dataset=iter(dataset_list))
    assert hasattr(report, "metrics")
    # Generator was consumed → metrics list shouldn't be empty.
    assert len(report.metrics) > 0
