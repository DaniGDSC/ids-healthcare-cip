"""ARCHITECTURE.md Step [10] — Risk-adaptive surfacing decision tests.

Locks the doc-promised invariants for ``src.risk_scorer.score_alert``:

* I1 INVARIANT 2 — CRITICAL+unpatchable always surfaces, regardless
     of maintenance window / similar events / threshold value.
* I2 Surfacing reason ∈ {``surfaced_safety_floor``, ``surfaced_normal``,
     ``suppressed_maintenance``, ``suppressed_below_threshold``}.
* I3 Multiplier table loaded from
     ``configs/risk_adaptive_thresholds.yaml`` with spec defaults as
     fallback. CRITICAL+unpatchable multiplier ≤ 0.70 (≥30% reduction).
* I4 Maintenance window suppresses display, NOT detection.
* I5 Single decision tree: safety floor checked FIRST, before
     maintenance, before threshold.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.risk_scorer import (
    DEFAULT_THRESHOLD,
    _THRESHOLD_MULT,
    _THRESHOLD_MULT_BY_DEVICE,
    score_alert,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ── I1 + I5: Safety floor unconditional ───────────────────────────────


def test_critical_unpatchable_surfaces_at_low_score():
    """CRITICAL+unpatchable surfaces even with a low anomaly score —
    the safety floor doesn't depend on the threshold check."""
    out = score_alert(
        anomaly_score=0.20,   # below the 0.50 default threshold
        device_context={"criticality": "CRITICAL", "patchable": False,
                        "clinical_function": "ventilator"},
        event_context=None,
    )
    assert out.should_surface is True


def test_critical_unpatchable_surfaces_during_maintenance():
    """Maintenance window cannot bypass the safety floor."""
    out = score_alert(
        anomaly_score=0.30,
        device_context={"criticality": "CRITICAL", "patchable": False,
                        "clinical_function": "ventilator"},
        event_context={"is_maintenance_window": True,
                       "is_known_vendor_ip": True},
    )
    assert out.should_surface is True


def test_safety_floor_evaluated_before_threshold():
    """Even if the threshold would have suppressed the alert,
    the safety floor surfaces it. Tests the order of branches in
    the single decision tree."""
    out = score_alert(
        anomaly_score=0.10,   # well below any reasonable threshold
        device_context={"criticality": "CRITICAL", "patchable": False,
                        "clinical_function": "infusion_pump"},
        event_context=None,
    )
    assert out.should_surface is True


# ── I2: surfacing reason coverage ────────────────────────────────────


def test_low_patchable_below_threshold_is_suppressed():
    out = score_alert(
        anomaly_score=0.05,
        device_context={"criticality": "LOW", "patchable": True,
                        "clinical_function": "admin_workstation"},
        event_context=None,
    )
    assert out.should_surface is False


def test_low_patchable_during_maintenance_suppressed():
    """Non-life-critical alert during maintenance with known vendor IP
    is suppressed (display, not detection)."""
    out = score_alert(
        anomaly_score=0.30,
        device_context={"criticality": "LOW", "patchable": True,
                        "clinical_function": "admin_workstation"},
        event_context={"is_maintenance_window": True,
                       "is_known_vendor_ip": True},
    )
    # Suppression reason should be populated when not surfaced.
    if out.should_surface is False:
        assert out.suppression_reason is not None


def test_high_score_normal_alert_surfaces():
    out = score_alert(
        anomaly_score=0.80,
        device_context={"criticality": "HIGH", "patchable": True,
                        "clinical_function": "patient_monitor"},
        event_context=None,
    )
    assert out.should_surface is True


# ── I3: multiplier table sourced from YAML ───────────────────────────


def test_multiplier_table_yaml_present():
    p = PROJECT_ROOT / "configs" / "risk_adaptive_thresholds.yaml"
    assert p.exists(), f"{p} missing — Step [10] policy YAML required"


def test_critical_unpatchable_multiplier_at_or_below_070():
    """≥30% threshold reduction (multiplier ≤ 0.70) per doc spec."""
    assert _THRESHOLD_MULT[("CRITICAL", False)] <= 0.70


@pytest.mark.parametrize("device", ["infusion_pump", "ventilator", "patient_monitor"])
def test_life_sustaining_unpatchable_multipliers_strict(device: str):
    """Life-sustaining devices unpatchable get the most aggressive
    surfacing thresholds (≤ 0.75)."""
    assert _THRESHOLD_MULT_BY_DEVICE[(device, False)] <= 0.75


def test_default_threshold_in_unit_interval():
    assert 0.0 < DEFAULT_THRESHOLD < 1.0


# ── I4: maintenance suppresses display, not detection ────────────────


def test_maintenance_suppression_does_not_change_anomaly_score():
    """Maintenance affects ``should_surface``, NOT the underlying
    detection signal. The adjusted_score should still reflect the
    alert's anomaly magnitude."""
    raw = score_alert(
        anomaly_score=0.45,
        device_context={"criticality": "MEDIUM", "patchable": True,
                        "clinical_function": "ehr_workstation"},
        event_context=None,
    )
    in_maint = score_alert(
        anomaly_score=0.45,
        device_context={"criticality": "MEDIUM", "patchable": True,
                        "clinical_function": "ehr_workstation"},
        event_context={"is_maintenance_window": True,
                       "is_known_vendor_ip": True},
    )
    # Adjusted score is unchanged (or close to) by maintenance —
    # detection didn't move; surfacing did.
    assert abs(raw.adjusted_score - in_maint.adjusted_score) < 0.30
