"""Tests for ``module4_explanations.phase0_metrics``.

The metrics drive the Phase-0 baseline / CI gate, so a regression in
their computation would mask a real regression in the upgrade. Tests
exercise both the per-metric functions and the ``collect_baseline``
driver against synthetic inputs that mirror the real artifact shapes.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from module4_explanations.phase0_metrics import (
    collect_baseline,
    compute_action_specificity,
    compute_counterfactual_coverage,
    compute_narrative_faithfulness,
    is_specific,
    narrative_category_from_summary,
)


# ── Reverse-mapping ─────────────────────────────────────────────────


def test_narrative_reverse_map_biometric():
    summary = "CRITICAL ALERT (Sample 5): primary indicator was abnormal blood pressure. Recommend review."
    assert narrative_category_from_summary(summary) == "biometric"


def test_narrative_reverse_map_network_timing():
    summary = "HIGH ALERT: Key factor: unusual network packet timing. Verify device."
    assert narrative_category_from_summary(summary) == "network_timing"


def test_narrative_reverse_map_unknown():
    summary = "LOW ALERT: Borderline detection by one model. Likely benign; logged for audit."
    assert narrative_category_from_summary(summary) is None


def test_narrative_reverse_map_empty():
    assert narrative_category_from_summary("") is None


def test_narrative_reverse_map_prefers_longest_phrase():
    summary = "abnormal blood pressure detected — and abnormal heart rate also noted"
    assert narrative_category_from_summary(summary) == "biometric"


# ── narrative_faithfulness ──────────────────────────────────────────


def test_narrative_faithfulness_perfect_match():
    analyst = [
        {"sample_index": 1, "models": {"xgboost": {"top_features": [{"feature": "SYS"}]}}},
        {"sample_index": 2, "models": {"xgboost": {"top_features": [{"feature": "DIntPkt"}]}}},
    ]
    clinician = [
        {"sample_index": 1, "summary": "HIGH ALERT: Key factor: abnormal blood pressure."},
        {"sample_index": 2, "summary": "MODERATE: unusual network packet timing detected."},
    ]
    out = compute_narrative_faithfulness(analyst, clinician)
    assert out["n"] == 2
    assert out["n_matched"] == 2
    assert out["rate"] == 1.0


def test_narrative_faithfulness_partial_miss():
    analyst = [
        {"sample_index": 1, "models": {"xgboost": {"top_features": [{"feature": "SYS"}]}}},
        {"sample_index": 2, "models": {"xgboost": {"top_features": [{"feature": "DIntPkt"}]}}},
    ]
    clinician = [
        {"sample_index": 1, "summary": "HIGH ALERT: abnormal blood pressure."},     # matches
        {"sample_index": 2, "summary": "HIGH ALERT: abnormal blood pressure."},     # mismatch
    ]
    out = compute_narrative_faithfulness(analyst, clinician)
    assert out["n"] == 2
    assert out["n_matched"] == 1
    assert out["rate"] == 0.5
    assert len(out["sample_mismatches"]) == 1
    assert out["sample_mismatches"][0]["sample_index"] == 2


def test_narrative_faithfulness_unknown_narrative_excluded():
    analyst = [
        {"sample_index": 1, "models": {"xgboost": {"top_features": [{"feature": "SYS"}]}}},
    ]
    clinician = [
        {"sample_index": 1, "summary": "LOW ALERT: Borderline detection."},
    ]
    out = compute_narrative_faithfulness(analyst, clinician)
    assert out["n"] == 0
    assert out["n_unknown_narrative"] == 1


def test_narrative_faithfulness_handles_missing_analyst_entry():
    analyst: list = []
    clinician = [{"sample_index": 1, "summary": "abnormal blood pressure"}]
    out = compute_narrative_faithfulness(analyst, clinician)
    assert out["n"] == 0


# ── action_specificity ─────────────────────────────────────────────


def test_is_specific_generic_text():
    ok, hits = is_specific("Isolate device immediately and notify SOC.")
    assert ok is False
    assert hits == []


def test_is_specific_with_ip():
    ok, hits = is_specific("Block destination 198.51.100.42")
    assert ok is True
    assert "ipv4" in hits


def test_is_specific_with_port():
    ok, hits = is_specific("Suspicious traffic on tcp/44312 from device")
    assert ok is True
    assert "port_number" in hits


def test_is_specific_with_bed_and_extension():
    ok, hits = is_specific("Check SpO2 sensor on Bed-12 and page ext 4422")
    assert ok is True
    assert "bed_or_room" in hits
    assert "extension" in hits


def test_is_specific_with_mitre():
    ok, hits = is_specific("Consistent with MITRE T1565.001 (Stored Data Manipulation)")
    assert ok is True
    assert "mitre_id" in hits


def test_is_specific_with_alert_id():
    ok, hits = is_specific("[ALERT-0042 · patient_monitor] Isolate device")
    assert ok is True
    assert "alert_id" in hits


def test_compute_action_specificity_all_generic():
    responses = [
        {
            "sample_index": 0,
            "response": {"action_descriptions": ["Isolate device", "Notify SOC"]},
            "explanation": {
                "mve": {
                    "layer_3": {
                        "immediate_action": "Restrict traffic and enable logging.",
                        "escalation_path": "(1) Security lead, (2) Clinical Engineering.",
                    }
                }
            },
        }
    ]
    clinician = [{"sample_index": 0, "summary": "HIGH: abnormal blood pressure."}]
    out = compute_action_specificity(responses, clinician)
    assert out["overall_rate"] == 0.0
    assert all(rate == 0.0 for rate in out["per_source_rate"].values())


def test_compute_action_specificity_mixed():
    responses = [
        {
            "sample_index": 0,
            "response": {"action_descriptions": ["Block tcp/44312"]},  # specific
            "explanation": {
                "mve": {
                    "layer_3": {
                        "immediate_action": "Isolate device",          # generic
                        "escalation_path": "Page ext 4422",            # specific
                    }
                }
            },
        }
    ]
    clinician = [{"sample_index": 0, "summary": "Check Bed-12 SpO2"}]   # specific
    out = compute_action_specificity(responses, clinician)
    assert out["overall_rate"] == pytest.approx(0.75)
    assert out["per_source_rate"]["layer3_action"] == 0.0
    assert out["per_source_rate"]["clinician_summary"] == 1.0


# ── counterfactual_coverage ────────────────────────────────────────


def test_counterfactual_coverage_baseline_zero():
    responses = [{"sample_index": i} for i in range(5)]
    out = compute_counterfactual_coverage(responses)
    assert out["n"] == 5
    assert out["n_with_counterfactual"] == 0
    assert out["rate"] == 0.0


def test_counterfactual_coverage_detects_payload_at_any_location():
    responses = [
        {"sample_index": 0, "counterfactual": {"changes": []}},
        {"sample_index": 1, "explanation": {"counterfactual": {"changes": []}}},
        {"sample_index": 2, "explanation": {"mve": {"counterfactual": {"x": 1}}}},
        {"sample_index": 3},  # none
    ]
    out = compute_counterfactual_coverage(responses)
    assert out["n_with_counterfactual"] == 3
    assert out["rate"] == 0.75


def test_counterfactual_coverage_ignores_empty_payload():
    responses = [
        {"sample_index": 0, "counterfactual": None},
        {"sample_index": 1, "counterfactual": {}},
        {"sample_index": 2, "explanation": {"counterfactual": []}},
    ]
    out = compute_counterfactual_coverage(responses)
    assert out["n_with_counterfactual"] == 0


# ── collect_baseline integration ───────────────────────────────────


def test_collect_baseline_end_to_end(tmp_path: Path):
    reports = tmp_path / "reports"
    reports.mkdir()

    analyst = [
        {"sample_index": 1, "models": {"xgboost": {"top_features": [{"feature": "SYS"}]}}},
    ]
    clinician = [
        {"sample_index": 1, "summary": "HIGH ALERT: abnormal blood pressure on Bed-7"},
    ]
    responses = [
        {
            "sample_index": 1,
            "response": {"action_descriptions": ["Isolate device"]},
            "explanation": {
                "mve": {
                    "layer_3": {
                        "immediate_action": "Block tcp/443",
                        "escalation_path": "Notify SOC",
                    }
                }
            },
        }
    ]
    (reports / "analyst_report.json").write_text(json.dumps(analyst))
    (reports / "clinician_summaries.json").write_text(json.dumps(clinician))
    (reports / "alert_responses.json").write_text(json.dumps(responses))

    out = collect_baseline(reports)

    assert out["_meta"]["n_analyst_alerts"] == 1
    assert out["narrative_faithfulness"]["rate"] == 1.0
    assert out["action_specificity"]["overall_rate"] > 0.0
    assert out["counterfactual_coverage"]["rate"] == 0.0


def test_collect_baseline_accepts_envelope_format(tmp_path: Path):
    reports = tmp_path / "reports"
    reports.mkdir()

    envelope = {
        "_provenance": {"split": "test"},
        "records": [
            {
                "sample_index": 1,
                "response": {"action_descriptions": []},
                "explanation": {"mve": {"layer_3": {}}},
            }
        ],
    }
    (reports / "analyst_report.json").write_text(json.dumps([]))
    (reports / "clinician_summaries.json").write_text(json.dumps([]))
    (reports / "alert_responses.json").write_text(json.dumps(envelope))

    out = collect_baseline(reports)
    assert out["_meta"]["n_alert_responses"] == 1
