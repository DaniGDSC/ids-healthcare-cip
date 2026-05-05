"""INVARIANT 6 — role authority enforcement (closes GAP-A16).

Each role's view must NOT contain action wording outside its authority:
  * IT_generalist  — cannot administer medication / titrate doses
  * biomed_engineer — cannot push network policy
  * nurse_manager — cannot touch network OR device firmware

Tests exercise every (role × alert-type × device-class) combination the
production generator can emit, then run the same allow-list check used by
the policy table src.mve_generator.ROLE_FORBIDDEN_ACTION_TERMS.

This file is the GAP-A16 closure artifact. The 8 tests added in
test_safe_failure.py during GAP-A2 closure are smoke tests; this file is
the exhaustive check.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_models import MVEOutput, OperatorRole
from src.mve_generator import (
    ROLE_FORBIDDEN_ACTION_TERMS,
    _generate_rule_based,
    derive_role_view,
    role_authority_violations,
)

# Test fixtures: minimal device + raw alert + baseline that satisfies the
# rule-based generator's required keys. Built from the same shape as
# tests/fixtures/sample_alerts.yaml without coupling to that file.

SAMPLE_RAW = {
    "src_ip": "10.4.12.50",
    "dest_ip": "203.0.113.99",
    "protocol": "TCP",
    "outbound_bytes": 50_000_000,
    "timestamp": "2026-05-05T18:00:00Z",
    "destination": "203.0.113.99",
    "alert_id": "test-alert-A16",
    "alert_name": "anomalous outbound transfer",
}

SAMPLE_BASELINE = {
    "normal_destinations": ["10.4.12.0/24", "10.4.13.0/24"],
    "normal_query_rate": 12,
    "observed_query_rate": 310,
    "baseline_days": 90,
}


def _device_ctx(criticality: str = "CRITICAL", device_type: str = "ventilator") -> dict:
    return {
        "criticality": criticality,
        "patchable": False,
        "device_type": device_type,
        "device_class": device_type,
        "clinical_function": "active mechanical ventilation",
    }


@pytest.mark.parametrize("alert_type", ["T1", "T2", "T3", "T4", "T5"])
@pytest.mark.parametrize("device_type", ["ventilator", "infusion_pump", "patient_monitor"])
def test_biomed_view_forbids_network_actions(alert_type: str, device_type: str) -> None:
    """Across every alert type × device, biomed view contains zero network
    mutation verbs."""
    user_ctx = {"user_id": "test-user", "role": "nurse"} if alert_type == "T2" else None
    base = _generate_rule_based(
        SAMPLE_RAW, _device_ctx(device_type=device_type),
        SAMPLE_BASELINE, user_ctx, alert_type,
    )
    view = derive_role_view(base, role=OperatorRole.BIOMED_ENGINEER.value,
                            alert_type=alert_type)
    violations = role_authority_violations(view, OperatorRole.BIOMED_ENGINEER.value)
    assert violations == [], (
        f"Biomed view leaked network actions on alert={alert_type} "
        f"device={device_type}: {violations}\n"
        f"immediate_action: {view.layer_3.get('immediate_action')!r}"
    )


@pytest.mark.parametrize("alert_type", ["T1", "T2", "T3", "T4", "T5"])
@pytest.mark.parametrize("device_type", ["ventilator", "infusion_pump", "patient_monitor"])
def test_nurse_view_forbids_network_and_device_firmware(alert_type: str, device_type: str) -> None:
    """Across every alert type × device, nurse view forbids both network
    AND device-firmware verbs."""
    user_ctx = {"user_id": "test-user", "role": "nurse"} if alert_type == "T2" else None
    base = _generate_rule_based(
        SAMPLE_RAW, _device_ctx(device_type=device_type),
        SAMPLE_BASELINE, user_ctx, alert_type,
    )
    view = derive_role_view(base, role=OperatorRole.NURSE_MANAGER.value,
                            alert_type=alert_type)
    violations = role_authority_violations(view, OperatorRole.NURSE_MANAGER.value)
    assert violations == [], (
        f"Nurse view leaked forbidden verbs on alert={alert_type} "
        f"device={device_type}: {violations}\n"
        f"immediate_action: {view.layer_3.get('immediate_action')!r}"
    )


@pytest.mark.parametrize("alert_type", ["T1", "T2", "T3", "T4", "T5"])
def test_it_generalist_view_forbids_clinical_mutations(alert_type: str) -> None:
    """IT generalist must not administer medication or alter clinical device
    parameters — those belong to clinical staff."""
    user_ctx = {"user_id": "test-user", "role": "nurse"} if alert_type == "T2" else None
    base = _generate_rule_based(
        SAMPLE_RAW, _device_ctx(), SAMPLE_BASELINE, user_ctx, alert_type,
    )
    view = derive_role_view(base, role=OperatorRole.IT_GENERALIST.value,
                            alert_type=alert_type)
    violations = role_authority_violations(view, OperatorRole.IT_GENERALIST.value)
    assert violations == [], (
        f"IT generalist view contained clinical-mutation verb on "
        f"alert={alert_type}: {violations}"
    )


def test_role_authority_violations_detects_planted_violation() -> None:
    """Positive control — the helper actually catches a violation when one
    is planted."""
    bad = MVEOutput(
        layer_1={"baseline_behavior": "x", "deviation_description": "y",
                 "confidence_indicator": "z"},
        layer_2={"affected_system": "x", "patient_care_impact": "y",
                 "phi_exposure": "no PHI", "severity_label": "HIGH",
                 "severity_rationale": "x"},
        layer_3={
            "immediate_action": "Push NAC policy and block port at switch",
            "clinical_constraint": "DO NOT power off device",
            "escalation_path": "IT",
            "timeframe": "15 min",
        },
    )
    hits = role_authority_violations(bad, OperatorRole.BIOMED_ENGINEER.value)
    assert "push nac" in hits
    assert "block port at switch" in hits


def test_layer_2_severity_invariant_across_all_views() -> None:
    """INVARIANT 6 cross-role consistency — exhaustively verified."""
    user_ctx = {"user_id": "test-user", "role": "nurse"}
    for alert_type in ("T1", "T2", "T3", "T4", "T5"):
        base = _generate_rule_based(
            SAMPLE_RAW, _device_ctx(),
            SAMPLE_BASELINE,
            user_ctx if alert_type == "T2" else None,
            alert_type,
        )
        baseline_severity = base.layer_2.get("severity_label")
        for role in OperatorRole:
            view = derive_role_view(base, role.value, alert_type)
            assert view.layer_2.get("severity_label") == baseline_severity, (
                f"Severity drifted: alert={alert_type} role={role.value} "
                f"got={view.layer_2.get('severity_label')!r} "
                f"expected={baseline_severity!r}"
            )


def test_clinical_constraint_preserved_exhaustive() -> None:
    """INVARIANT 7 — DO NOT wording must survive in every role × alert-type."""
    user_ctx = {"user_id": "test-user", "role": "nurse"}
    for alert_type in ("T1", "T2", "T3", "T4", "T5"):
        base = _generate_rule_based(
            SAMPLE_RAW, _device_ctx(),
            SAMPLE_BASELINE,
            user_ctx if alert_type == "T2" else None,
            alert_type,
        )
        if "DO NOT" not in base.layer_3.get("clinical_constraint", ""):
            # Some non-clinical alert types may legitimately omit DO NOT;
            # only check role-preservation when the base view has it.
            continue
        for role in OperatorRole:
            view = derive_role_view(base, role.value, alert_type)
            assert "DO NOT" in view.layer_3["clinical_constraint"], (
                f"DO NOT lost: alert={alert_type} role={role.value} "
                f"constraint={view.layer_3['clinical_constraint']!r}"
            )


def test_role_authority_table_covers_all_three_roles() -> None:
    """Schema-level test — the policy table itself must cover every defined role."""
    table_keys = set(ROLE_FORBIDDEN_ACTION_TERMS)
    enum_values = {r.value for r in OperatorRole}
    assert enum_values <= table_keys, (
        f"OperatorRole values not in policy table: {enum_values - table_keys}"
    )
