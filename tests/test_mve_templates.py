"""Per-template tests for the decomposed rule-based MVE generator.

Each ``_template_tN`` helper is tested in isolation against a synthetic
``_TemplateContext``. The dispatcher round-trip is verified by feeding raw
inputs through ``_generate_rule_based`` and asserting the right helper
fires for each alert_type.
"""
from __future__ import annotations

import pytest

from src.data_models import MVEOutput
from src.mve_generator import (
    _build_template_context,
    _generate_rule_based,
    _template_t1,
    _template_t2,
    _template_t3,
    _template_t4,
    _template_t5,
)


# ── _build_template_context ────────────────────────────────────────────


def _basic_inputs(criticality="CRITICAL", device_type="infusion_pump"):
    return {
        "raw_alert": {
            "alert_name": "Anomalous outbound from infusion pump",
            "source_ip": "10.0.1.50", "dest_ip": "198.51.100.42",
            "protocol": "TCP/443", "timestamp": "2026-05-26T03:15:00Z",
            "severity_score": 0.85,
        },
        "device_context": {
            "device_type": device_type,
            "criticality": criticality,
            "patchable": False,
            "clinical_function": "drug delivery",
            "location": "ICU",
        },
        "baseline": {
            "normal_destinations": ["10.0.2.0/24"],
            "normal_protocols": ["HTTPS"],
            "baseline_days": 90,
        },
    }


def test_build_template_context_basic_fields():
    inp = _basic_inputs()
    ctx = _build_template_context(inp["raw_alert"], inp["device_context"],
                                   inp["baseline"], "T1")
    assert ctx.criticality == "CRITICAL"
    assert ctx.device_type == "infusion_pump"
    assert ctx.location == "ICU"
    assert ctx.clinical_fn == "drug delivery"
    assert ctx.source_ip == "10.0.1.50"
    assert ctx.dest_ip == "198.51.100.42"
    assert ctx.time_str == "03:15"


def test_build_template_context_life_sustaining_floor_low_to_high():
    """FIX-E: infusion_pump at LOW criticality gets elevated to HIGH."""
    inp = _basic_inputs(criticality="LOW", device_type="infusion_pump")
    ctx = _build_template_context(inp["raw_alert"], inp["device_context"],
                                   inp["baseline"], "T1")
    assert ctx.criticality == "HIGH"
    assert "Severity elevated" in ctx.severity_rationale


def test_build_template_context_life_sustaining_floor_medium_to_high():
    inp = _basic_inputs(criticality="MEDIUM", device_type="ventilator")
    ctx = _build_template_context(inp["raw_alert"], inp["device_context"],
                                   inp["baseline"], "T1")
    assert ctx.criticality == "HIGH"


def test_build_template_context_high_critical_not_floored():
    """Life-sustaining devices already at HIGH/CRITICAL stay where they are."""
    for crit in ("HIGH", "CRITICAL"):
        inp = _basic_inputs(criticality=crit, device_type="ventilator")
        ctx = _build_template_context(inp["raw_alert"], inp["device_context"],
                                       inp["baseline"], "T1")
        assert ctx.criticality == crit


def test_build_template_context_non_life_sustaining_no_floor():
    """Patient_monitor is NOT in _LIFE_SUSTAINING — LOW stays LOW."""
    inp = _basic_inputs(criticality="LOW", device_type="patient_monitor")
    ctx = _build_template_context(inp["raw_alert"], inp["device_context"],
                                   inp["baseline"], "T1")
    assert ctx.criticality == "LOW"


def test_build_template_context_normal_dests_formatted():
    inp = _basic_inputs()
    ctx = _build_template_context(inp["raw_alert"], inp["device_context"],
                                   inp["baseline"], "T1")
    # _fmt_dests is applied — exact format depends on _fmt_dests, just
    # confirm it's a non-empty string.
    assert isinstance(ctx.normal_dests, str)
    assert ctx.normal_dests


def test_build_template_context_escalation_critical_includes_icu():
    inp = _basic_inputs(criticality="CRITICAL")
    ctx = _build_template_context(inp["raw_alert"], inp["device_context"],
                                   inp["baseline"], "T1")
    assert "ICU" in ctx.escalation or "Clinical" in ctx.escalation


# ── _template_t1 ───────────────────────────────────────────────────────


def test_template_t1_returns_mve_output():
    inp = _basic_inputs()
    ctx = _build_template_context(inp["raw_alert"], inp["device_context"],
                                   inp["baseline"], "T1")
    mve = _template_t1(ctx)
    assert isinstance(mve, MVEOutput)


def test_template_t1_critical_includes_nac():
    inp = _basic_inputs(criticality="CRITICAL", device_type="infusion_pump")
    ctx = _build_template_context(inp["raw_alert"], inp["device_context"],
                                   inp["baseline"], "T1")
    mve = _template_t1(ctx)
    assert "NAC" in mve.layer_3["immediate_action"] or \
           "quarantine" in mve.layer_3["immediate_action"].lower()


def test_template_t1_low_does_not_block():
    inp = _basic_inputs(criticality="LOW", device_type="workstation")
    ctx = _build_template_context(inp["raw_alert"], inp["device_context"],
                                   inp["baseline"], "T1")
    mve = _template_t1(ctx)
    assert "Log" in mve.layer_3["immediate_action"]


def test_template_t1_clinical_constraint_starts_with_do_not():
    inp = _basic_inputs(criticality="HIGH")
    ctx = _build_template_context(inp["raw_alert"], inp["device_context"],
                                   inp["baseline"], "T1")
    mve = _template_t1(ctx)
    assert mve.layer_3["clinical_constraint"].startswith("DO NOT")


# ── _template_t2 ───────────────────────────────────────────────────────


def test_template_t2_includes_user_id():
    inp = _basic_inputs()
    ctx = _build_template_context(inp["raw_alert"], inp["device_context"],
                                   inp["baseline"], "T2")
    user_context = {
        "user_id": "U123", "role": "nurse",
        "department": "cardiology", "shift": "night",
        "normal_access_scope": "cardiology unit",
        "normal_access_volume": 15,
    }
    mve = _template_t2(ctx, user_context)
    assert "U123" in mve.layer_1["baseline_behavior"]
    assert "U123" in mve.layer_3["immediate_action"]


def test_template_t2_always_high_severity():
    inp = _basic_inputs(criticality="LOW")
    ctx = _build_template_context(inp["raw_alert"], inp["device_context"],
                                   inp["baseline"], "T2")
    user_context = {"user_id": "U1", "role": "tech",
                    "department": "x", "shift": "day",
                    "normal_access_scope": "y", "normal_access_volume": 5}
    mve = _template_t2(ctx, user_context)
    assert mve.layer_2["severity_label"] == "HIGH"


def test_template_t2_alert_involves_clinical_system_true():
    """EHR access is always clinical, regardless of device tier."""
    inp = _basic_inputs(criticality="MEDIUM")
    ctx = _build_template_context(inp["raw_alert"], inp["device_context"],
                                   inp["baseline"], "T2")
    user_context = {"user_id": "U1", "role": "x", "department": "y",
                    "shift": "z", "normal_access_scope": "a",
                    "normal_access_volume": 1}
    mve = _template_t2(ctx, user_context)
    assert mve.alert_involves_clinical_system is True


# ── _template_t3 ───────────────────────────────────────────────────────


def test_template_t3_lateral_movement_layer1_mentions_vlan():
    inp = _basic_inputs(criticality="HIGH")
    ctx = _build_template_context(inp["raw_alert"], inp["device_context"],
                                   inp["baseline"], "T3")
    mve = _template_t3(ctx)
    assert "VLAN" in mve.layer_1["deviation_description"]


def test_template_t3_layer3_blocks_at_firewall():
    inp = _basic_inputs(criticality="HIGH")
    ctx = _build_template_context(inp["raw_alert"], inp["device_context"],
                                   inp["baseline"], "T3")
    mve = _template_t3(ctx)
    assert "firewall" in mve.layer_3["immediate_action"].lower()
    assert mve.layer_3["clinical_constraint"].startswith("DO NOT")


# ── _template_t4 ───────────────────────────────────────────────────────


def test_template_t4_layer3_blocks_outbound():
    inp = _basic_inputs(criticality="HIGH")
    ctx = _build_template_context(inp["raw_alert"], inp["device_context"],
                                   inp["baseline"], "T4")
    mve = _template_t4(ctx)
    assert "Block outbound" in mve.layer_3["immediate_action"]
    assert mve.layer_3["clinical_constraint"].startswith("DO NOT")


def test_template_t4_phi_exfiltration_mentioned():
    inp = _basic_inputs(criticality="HIGH")
    ctx = _build_template_context(inp["raw_alert"], inp["device_context"],
                                   inp["baseline"], "T4")
    mve = _template_t4(ctx)
    assert "exfiltration" in mve.layer_2["patient_care_impact"].lower()


# ── _template_t5 ───────────────────────────────────────────────────────


@pytest.mark.parametrize("device_type", [
    "ventilator", "infusion_pump", "insulin_pump", "patient_monitor",
])
def test_template_t5_device_specific_constraints(device_type):
    inp = _basic_inputs(criticality="HIGH", device_type=device_type)
    ctx = _build_template_context(inp["raw_alert"], inp["device_context"],
                                   inp["baseline"], "T5")
    mve = _template_t5(ctx)
    constraint = mve.layer_3["clinical_constraint"]
    assert constraint.startswith("DO NOT")
    assert "SAFE:" in constraint


def test_template_t5_unknown_device_uses_default_template():
    inp = _basic_inputs(criticality="MEDIUM", device_type="workstation")
    ctx = _build_template_context(inp["raw_alert"], inp["device_context"],
                                   inp["baseline"], "T5")
    mve = _template_t5(ctx)
    # Default template path — still produces valid layer_3.
    assert "rate-limit" in mve.layer_3["immediate_action"].lower()


# ── Dispatcher (round-trip via _generate_rule_based) ───────────────────


@pytest.mark.parametrize("alert_type", ["T1", "T2", "T3", "T4", "T5"])
def test_dispatcher_routes_to_template(alert_type):
    inp = _basic_inputs(criticality="HIGH")
    user_context = None
    if alert_type == "T2":
        user_context = {"user_id": "U1", "role": "x", "department": "y",
                        "shift": "z", "normal_access_scope": "a",
                        "normal_access_volume": 1}
    mve = _generate_rule_based(
        inp["raw_alert"], inp["device_context"], inp["baseline"],
        user_context, alert_type,
    )
    assert isinstance(mve, MVEOutput)
    # Word budget invariant (M1: <= 150 total).
    assert mve.total_word_count <= 150


def test_dispatcher_t2_falls_back_to_t1_without_user_context():
    """T2 alert_type but missing user_context → falls through to T1."""
    inp = _basic_inputs(criticality="HIGH")
    mve = _generate_rule_based(
        inp["raw_alert"], inp["device_context"], inp["baseline"],
        None, "T2",
    )
    # No "user" mentioned in layer 1 → T1 fired, not T2.
    assert "User" not in mve.layer_1["baseline_behavior"]


def test_dispatcher_unknown_alert_type_falls_back_to_t1():
    inp = _basic_inputs(criticality="HIGH")
    mve = _generate_rule_based(
        inp["raw_alert"], inp["device_context"], inp["baseline"],
        None, "T_UNKNOWN",
    )
    assert isinstance(mve, MVEOutput)


# ── Word budget regression — all templates must stay under M1 limits ───


@pytest.mark.parametrize("alert_type,device_type,criticality", [
    ("T1", "infusion_pump", "CRITICAL"),
    ("T1", "patient_monitor", "HIGH"),
    ("T1", "workstation", "LOW"),
    ("T3", "infusion_pump", "HIGH"),
    ("T4", "ehr_workstation", "HIGH"),
    ("T5", "ventilator", "HIGH"),
    ("T5", "infusion_pump", "CRITICAL"),
    ("T5", "patient_monitor", "MEDIUM"),
])
def test_template_word_budget_under_150(alert_type, device_type, criticality):
    inp = _basic_inputs(criticality=criticality, device_type=device_type)
    mve = _generate_rule_based(
        inp["raw_alert"], inp["device_context"], inp["baseline"],
        None, alert_type,
    )
    assert mve.total_word_count <= 150, (
        f"{alert_type}/{device_type}/{criticality} exceeded 150-word budget: "
        f"{mve.total_word_count}"
    )
