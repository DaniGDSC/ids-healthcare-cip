"""Tests for Phase 3 upgrades — decision-tree playbook + role-mismatch
warning.

Covers:
  - playbook selection per (top_category, severity)
  - playbook structure invariants (≥2 steps, terminal action last)
  - Markdown render shape
  - routing-mismatch detection across the canonical category set
  - role normalisation / aliases
  - Response schema accepts ``playbook`` + ``routing_warning``
"""
from __future__ import annotations

import pytest


# ── Phase 3.1 — Playbooks ─────────────────────────────────────────


def test_select_playbook_returns_biometric_for_biometric_category():
    from module5_responses.playbooks import select_playbook
    pb = select_playbook("biometric", "HIGH")
    assert pb.name == "biometric_anomaly"


def test_select_playbook_returns_default_for_unknown_category():
    from module5_responses.playbooks import select_playbook
    pb = select_playbook("totally_made_up", "HIGH")
    assert pb.name == "default"


def test_select_playbook_returns_network_for_each_network_category():
    from module5_responses.playbooks import select_playbook
    for cat in ("network_volume", "network_protocol", "network_timing",
                "network_packet", "network_loss"):
        pb = select_playbook(cat, "HIGH")
        assert pb.name.startswith("network_"), (
            f"category {cat} should map to a network playbook, got {pb.name}"
        )


def test_playbook_has_at_least_two_steps():
    """Acceptance: every playbook must have ≥2 steps (one check + one
    terminal action), otherwise it's not really a decision tree."""
    from module5_responses.playbooks import _CATEGORY_PLAYBOOKS, _PLAYBOOK_DEFAULT
    for pb in list(_CATEGORY_PLAYBOOKS.values()) + [_PLAYBOOK_DEFAULT]:
        assert len(pb.steps) >= 2, f"playbook {pb.name} has only {len(pb.steps)} step(s)"


def test_playbook_last_step_is_terminal():
    """The final step must always be terminal (empty check) so the
    operator never reaches the end without an action."""
    from module5_responses.playbooks import _CATEGORY_PLAYBOOKS, _PLAYBOOK_DEFAULT
    for pb in list(_CATEGORY_PLAYBOOKS.values()) + [_PLAYBOOK_DEFAULT]:
        last = pb.steps[-1]
        assert last.is_terminal(), (
            f"playbook {pb.name} last step has check {last.check!r} — should be terminal"
        )
        assert last.action_yes, "terminal action_yes must not be empty"


def test_render_markdown_produces_numbered_checklist():
    from module5_responses.playbooks import render_markdown, select_playbook
    pb = select_playbook("biometric", "HIGH")
    md = render_markdown(pb)
    assert md.startswith("**Playbook:")
    assert "1." in md
    assert "2." in md
    assert "**Check:**" in md or "**Action:**" in md


def test_playbook_to_dict_is_json_safe():
    import json
    from module5_responses.playbooks import select_playbook
    pb = select_playbook("network_volume", "HIGH")
    json.dumps(pb.to_dict())  # must not raise


def test_playbook_step_is_terminal_method():
    from module5_responses.playbooks import PlaybookStep
    assert PlaybookStep(check="", action_yes="do X").is_terminal()
    assert not PlaybookStep(check="Q?", action_yes="do X").is_terminal()


# ── Phase 3.2 — Routing mismatch ──────────────────────────────────


def test_biometric_routed_to_it_security_is_mismatch():
    """The canonical mismatch: biometric SHAP signal routed to network
    security. Phase 3.2's reason d'être."""
    from module5_responses.role_routing import detect_routing_mismatch
    out = detect_routing_mismatch("biometric", "IT Security")
    assert out.mismatch
    assert out.suggested_role == "Clinical Engineering"
    assert "biometric" in out.reason.lower()


def test_network_volume_routed_to_clinical_engineering_is_mismatch():
    """Reverse case: network anomaly routed to clinical staff."""
    from module5_responses.role_routing import detect_routing_mismatch
    out = detect_routing_mismatch("network_volume", "Charge Nurse")
    assert out.mismatch
    assert out.suggested_role == "IT Security"


def test_aligned_routing_no_warning():
    """When the routing primary matches the expected audience for the
    SHAP category, no warning is emitted."""
    from module5_responses.role_routing import detect_routing_mismatch
    out = detect_routing_mismatch("network_protocol", "IT Security")
    assert not out.mismatch
    out = detect_routing_mismatch("biometric", "Clinical Engineering")
    assert not out.mismatch


def test_role_alias_treated_as_aligned():
    """Routing primary 'Security lead' or 'SOC' is the same audience as
    'IT Security' and should NOT trigger a warning."""
    from module5_responses.role_routing import detect_routing_mismatch
    for alias in ("Security lead", "SOC", "CISO", "Incident Commander"):
        out = detect_routing_mismatch("network_volume", alias)
        assert not out.mismatch, (
            f"alias {alias!r} should align with IT Security audience"
        )


def test_clinical_role_aliases_treated_as_aligned():
    from module5_responses.role_routing import detect_routing_mismatch
    for alias in ("Biomedical Engineering", "Charge Nurse",
                   "ICU charge nurse", "On-call Physician"):
        out = detect_routing_mismatch("biometric", alias)
        assert not out.mismatch, (
            f"alias {alias!r} should align with Clinical Engineering audience"
        )


def test_unknown_category_emits_no_mismatch():
    """When the SHAP category is unknown (or absent), we can't decide
    the expected audience — the warning stays inert rather than
    falsely flagging."""
    from module5_responses.role_routing import detect_routing_mismatch
    out = detect_routing_mismatch("unknown", "IT Security")
    assert not out.mismatch


def test_no_routing_primary_emits_no_mismatch():
    """NORMAL alerts have no routing primary (None) — no warning."""
    from module5_responses.role_routing import detect_routing_mismatch
    out = detect_routing_mismatch("biometric", None)
    assert not out.mismatch
    out = detect_routing_mismatch("biometric", "")
    assert not out.mismatch


def test_routing_warning_dict_is_json_safe():
    import json
    from module5_responses.role_routing import detect_routing_mismatch
    out = detect_routing_mismatch("biometric", "IT Security")
    json.dumps(out.to_dict())


# ── Schema integration ───────────────────────────────────────────


def test_response_schema_accepts_playbook():
    from common.alert_response_schema import Response, EscalationChain
    payload = {
        "actions": ["log_event"], "action_descriptions": ["x"],
        "escalation_chain": EscalationChain(primary=None, secondary=None, tertiary=None),
        "escalation_rationale": "", "max_response_min": 0, "priority": 4,
        "rationale": "", "device_tier": "vital_monitoring",
        "device_constraint_applied": False,
        "playbook": {
            "name": "biometric_anomaly",
            "description": "Verify vital first",
            "steps": [{"check": "Q?", "action_yes": "A", "action_no": ""}],
        },
    }
    r = Response(**payload)
    assert r.playbook["name"] == "biometric_anomaly"


def test_response_schema_accepts_routing_warning():
    from common.alert_response_schema import Response, RoutingWarning, EscalationChain
    payload = {
        "actions": ["log_event"], "action_descriptions": ["x"],
        "escalation_chain": EscalationChain(primary=None, secondary=None, tertiary=None),
        "escalation_rationale": "", "max_response_min": 0, "priority": 4,
        "rationale": "", "device_tier": "vital_monitoring",
        "device_constraint_applied": False,
        "routing_warning": {
            "mismatch": True,
            "current_primary": "IT Security",
            "suggested_role": "Clinical Engineering",
            "reason": "Biometric category should go to clinical staff.",
        },
    }
    r = Response(**payload)
    assert r.routing_warning.mismatch is True


def test_response_schema_legacy_records_still_validate():
    """Pre-Phase-3 records have neither playbook nor routing_warning —
    must still validate."""
    from common.alert_response_schema import Response, EscalationChain
    r = Response(
        actions=["log_event"], action_descriptions=["x"],
        escalation_chain=EscalationChain(primary=None, secondary=None, tertiary=None),
        escalation_rationale="", max_response_min=0, priority=4,
        rationale="", device_tier="vital_monitoring",
        device_constraint_applied=False,
    )
    assert r.playbook is None
    assert r.routing_warning is None
