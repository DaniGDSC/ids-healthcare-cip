"""ARCHITECTURE.md Step [13] — cross-role consistency contract tests.

Locks the role-view invariants:

* INVARIANT 6 — Layer 2 severity invariant across role views (existing
  coverage in ``test_role_authority.py``; this module adds a
  deterministic regression test).
* INVARIANT 7 — Layer 3 ``clinical_constraint`` (DO NOT) preserved
  across roles for CRITICAL/HIGH/MEDIUM clinical alerts.
* **INVARIANT 9 (NEW)** — shared anchor identical across all role views
  produced from the same source alert (``alert_id``, ``risk_tier``,
  ``device_id``, ``one_line_summary``, ``timestamp``).
"""
from __future__ import annotations

import pytest

from src.data_models import MVEOutput, OperatorRole, SharedAnchor
from src.mve_generator import role_authority_violations
from module5_responses.module5_pipeline import render_views_for_alert


def _make_mve(severity: str = "HIGH", do_not: str = "DO NOT power-cycle.") -> MVEOutput:
    return MVEOutput(
        layer_1={
            "baseline_behavior": "Idle device",
            "deviation_description": "Outbound traffic spike",
            "confidence_indicator": "Confidence: HIGH — calibrated XGB.",
        },
        layer_2={
            "affected_system": "infusion pump bed-4",
            "patient_care_impact": "active drug delivery at risk",
            "phi_exposure": "none",
            "severity_label": severity,
            "severity_rationale": "life-sustaining device on shift.",
        },
        layer_3={
            "immediate_action": "Isolate device from network.",
            "clinical_constraint": do_not,
            "escalation_path": "(1) call IT, (2) page biomed.",
            "timeframe": "Act within 15 minutes.",
        },
    )


# ── INVARIANT 9: shared anchor identical across role views ────────────


def test_shared_anchor_identical_across_three_role_views():
    anchor = SharedAnchor(
        alert_id="EVAL-0301",
        risk_tier="HIGH",
        device_id="patient_monitor-A12",
        one_line_summary="Suspected C2 traffic from monitor",
        timestamp="2026-05-07T18:00:00Z",
    )
    views = render_views_for_alert(_make_mve(), shared_anchor=anchor)

    role_keys = {r.value for r in OperatorRole}
    assert set(views.keys()) == role_keys

    anchors = [views[r]["shared_anchor"] for r in role_keys]
    # Byte-identical dicts — phone-handoff requires every operator sees
    # the same header.
    assert all(a == anchors[0] for a in anchors), (
        "Shared anchor diverged across role views — INVARIANT 9 violated"
    )
    assert anchors[0]["alert_id"] == "EVAL-0301"
    assert anchors[0]["device_id"] == "patient_monitor-A12"


def test_shared_anchor_carries_all_five_required_fields():
    anchor = SharedAnchor(
        alert_id="EVAL-0976",
        risk_tier="CRITICAL",
        device_id="ventilator-B-3",
        one_line_summary="Suspected data alteration",
        timestamp="2026-05-07T19:30:00Z",
    )
    d = anchor.to_dict()
    for k in ("alert_id", "risk_tier", "device_id", "one_line_summary", "timestamp"):
        assert k in d, f"Shared anchor missing required field {k!r}"


def test_render_views_without_anchor_is_backward_compatible():
    """Legacy callers that don't pass a shared_anchor must still get
    the original ``{role: MVEOutput}`` shape."""
    views = render_views_for_alert(_make_mve())
    role_keys = {r.value for r in OperatorRole}
    assert set(views.keys()) == role_keys
    for v in views.values():
        assert isinstance(v, MVEOutput)


# ── INVARIANT 6: cross-role severity invariance ───────────────────────


def test_layer_2_severity_invariant_across_roles():
    views = render_views_for_alert(_make_mve(severity="HIGH"))
    severities = {
        role: view.layer_2.get("severity_label")
        for role, view in views.items()
    }
    assert len(set(severities.values())) == 1, (
        f"Layer 2 severity diverged across roles: {severities}"
    )


# ── INVARIANT 7: DO NOT preserved on clinical alerts ──────────────────


@pytest.mark.parametrize("severity", ["CRITICAL", "HIGH", "MEDIUM"])
def test_do_not_clause_present_in_layer_3_for_clinical(severity: str):
    views = render_views_for_alert(_make_mve(severity=severity))
    for role, view in views.items():
        assert "DO NOT" in view.layer_3.get("clinical_constraint", "").upper(), (
            f"Role {role} ({severity} alert) lost the DO NOT clause"
        )


# ── Authority bounds (sanity) ─────────────────────────────────────────


def test_role_authority_bounds_hold_after_render():
    """Rendered views must not contain forbidden action terms for
    their role (substring match, case-insensitive)."""
    views = render_views_for_alert(_make_mve())
    for role, view in views.items():
        violations = role_authority_violations(view, role)
        assert not violations, (
            f"Role {role} layer_3 contains forbidden terms: {violations}"
        )


# ── RQ2_Compliance.md §6 — positive role differentiation ────────────


def test_layer_3_immediate_action_differs_across_roles():
    """Positive proof of role adaptation: Layer 3 ``immediate_action``
    must differ between at least two role views derived from the same
    source MVE.  If all three roles produce identical action text the
    role-scoping has silently failed — Invariants 6 + 9 alone don't
    catch this regression (they assert *what should match*, not *what
    should differ*).
    """
    views = render_views_for_alert(_make_mve())
    actions = {
        role: view.layer_3.get("immediate_action", "")
        for role, view in views.items()
    }
    distinct = set(actions.values())
    assert len(distinct) >= 2, (
        "Role adaptation failed: all three roles produced identical "
        f"Layer 3 immediate_action text.  Samples: "
        f"{[(r, (a or '')[:80]) for r, a in actions.items()]}"
    )
