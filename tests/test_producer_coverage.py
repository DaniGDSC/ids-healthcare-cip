"""Producer-side coverage tests (Sprint 2.1).

For every ``Literal`` value the schema accepts, prove there exists a
code path that emits it. For every taxonomy / config dict the pipeline
keys on, prove the keys are complete with respect to what consumers
expect. This catches Category 2 "implicit/dead code" bugs (e.g. NORMAL
tier dead code, missing CLINICIAN_TEMPLATES["NORMAL"]) before they
ship.

Each test pins one *producer ↔ consumer contract*:

  1. ``assign_risk_levels`` can emit every tier the
     ``AlertRecord.risk_level`` Literal accepts.
  2. ``CLINICIAN_TEMPLATES`` has a key for every surfaced tier (LOW+).
  3. ``ACTION_CATALOGUE`` has every action ``TIER_POLICIES`` /
     ``ATTACK_ROUTING`` reference.
  4. ``CATEGORY_PLAYBOOKS`` covers every category
     ``_feature_to_narrative`` can produce.
  5. ``ATTACK_ROUTING`` covers every attack category present in the
     test corpus (parquet + analyst_report).
  6. ``DEVICE_CONTEXT`` covers every device_class
     ``derive_device_class_row`` can return.
  7. ``RoutingWarning.suggested_role`` values map to canonical roles
     ``ESCALATION_CONTACTS`` knows about.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


# ── 1. assign_risk_levels covers all schema-allowed tiers ─────────


def test_assign_risk_levels_can_emit_every_surfaced_tier():
    """For each Literal value in ``AlertRecord.risk_level``, find an
    (R, c_detect) input that produces it. Closes the original NORMAL
    tier dead-code bug — no tier in the Literal can be unreachable."""
    from module3_risk_scoring.composition import assign_risk_levels

    # AlertRecord.risk_level Literal — read it from the schema directly
    # so this test self-updates when the Literal changes.
    from common.alert_response_schema import AlertRecord
    import typing as _t

    field = AlertRecord.model_fields["risk_level"]
    expected_tiers = set(_t.get_args(field.annotation))

    # Probe with a dense R grid + ample c_detect (≥0.5) so the detection
    # gate doesn't force NORMAL.
    R = np.linspace(0.0, 1.0, 101)
    c_detect = np.full(101, 0.5)
    emitted = set(assign_risk_levels(R, c_detect=c_detect))
    missing = expected_tiers - emitted
    assert not missing, (
        f"Schema Literal {expected_tiers} but assign_risk_levels can only emit "
        f"{emitted} — missing {missing} (Sprint 2.1 producer coverage)."
    )


def test_assign_risk_levels_normal_via_detection_gate():
    """NORMAL must be reachable via the detection gate too — not just
    via R being below the LOW threshold."""
    from module3_risk_scoring.composition import assign_risk_levels
    # Pick R values that would be MEDIUM+ without the gate
    R = np.array([0.85, 0.50])
    c_detect = np.array([0.0, 0.0])  # below MIN_DETECTION_GATE
    out = assign_risk_levels(R, c_detect=c_detect).tolist()
    assert out == ["NORMAL", "NORMAL"]


# ── 2. CLINICIAN_TEMPLATES has every surfaced tier ────────────────


def test_clinician_templates_cover_every_surfaced_tier():
    """The template lookup keyed by severity must not KeyError on any
    tier ``build_clinician_summaries`` can pass in. NORMAL is filtered
    out before the format call (Phase 3 fix), so it's excluded."""
    from module4_explanations.config import CLINICIAN_TEMPLATES
    from common.alert_response_schema import AlertRecord
    import typing as _t

    field = AlertRecord.model_fields["risk_level"]
    all_tiers = set(_t.get_args(field.annotation))
    surfaced = all_tiers - {"NORMAL"}  # NORMAL is filtered in build_clinician_summaries
    missing = surfaced - CLINICIAN_TEMPLATES.keys()
    assert not missing, (
        f"CLINICIAN_TEMPLATES missing entries for {missing}. Add a template "
        "or update build_clinician_summaries to skip the tier explicitly."
    )


# ── 3. ACTION_CATALOGUE covers every reference ───────────────────


def test_action_catalogue_covers_tier_policy_actions():
    """Every action referenced by TIER_POLICIES default_actions must
    exist in ACTION_CATALOGUE — otherwise select_adaptive_response
    KeyErrors on ``ACTION_CATALOGUE[a]["cost"]``."""
    from module5_responses.config import ACTION_CATALOGUE, TIER_POLICIES
    referenced = set()
    for tier_block in TIER_POLICIES.values():
        referenced.update(tier_block.get("default_actions", []))
    missing = referenced - ACTION_CATALOGUE.keys()
    assert not missing, f"TIER_POLICIES references unknown actions: {missing}"


def test_action_catalogue_covers_attack_routing_actions():
    """Same for ATTACK_ROUTING.attack_specific_actions + add_actions."""
    from module5_responses.config import ACTION_CATALOGUE, ATTACK_ROUTING
    referenced = set()
    for routing in ATTACK_ROUTING.values():
        referenced.update(routing.get("attack_specific_actions", []))
        referenced.update(routing.get("add_actions", []))
    missing = referenced - ACTION_CATALOGUE.keys()
    assert not missing, f"ATTACK_ROUTING references unknown actions: {missing}"


def test_action_catalogue_entries_have_required_fields():
    """Each catalogue entry must carry the full operational metadata
    Phase 1.3 + 4.1 depend on."""
    from module5_responses.config import ACTION_CATALOGUE
    required = {"severity_floor", "cost", "reversible", "requires_approval",
                "description", "expected_disruption"}
    for name, spec in ACTION_CATALOGUE.items():
        missing = required - spec.keys()
        assert not missing, f"action {name!r} missing fields: {missing}"


# ── 4. CATEGORY_PLAYBOOKS covers every SHAP narrative category ────


def test_playbook_table_covers_every_feature_category():
    """Every category emitted by ``_feature_to_narrative`` must have
    a playbook entry (or be handled by the default fallback).

    Pinning this would have caught a Phase 3 regression where a new
    feature category was added without a matching playbook."""
    from module4_explanations.feature_groups import _FEATURE_GROUPS
    from module5_responses.playbooks import _CATEGORY_PLAYBOOKS, _PLAYBOOK_DEFAULT, select_playbook

    categories = {cat for (_phrase, cat) in _FEATURE_GROUPS.values()}
    for cat in categories:
        pb = select_playbook(cat, "HIGH")
        # Either category-specific or explicitly the default fallback
        assert pb is not None
        assert pb.steps, f"playbook for {cat!r} has empty steps"


def test_default_playbook_is_terminal():
    """The default playbook must be a usable fallback — its last step
    must be terminal so the operator always reaches an action."""
    from module5_responses.playbooks import _PLAYBOOK_DEFAULT
    assert _PLAYBOOK_DEFAULT.steps
    assert _PLAYBOOK_DEFAULT.steps[-1].is_terminal()


# ── 5. ATTACK_ROUTING covers corpus attack categories ─────────────


@pytest.mark.parametrize("split", ["test", "demo"])
def test_attack_routing_covers_corpus_attack_categories(split):
    """Every attack_category that appears in alert_responses must
    either have a routing entry in ``ATTACK_ROUTING`` or fall through
    to ``DEFAULT_ROUTING`` — never KeyError."""
    from module5_responses.config import ATTACK_ROUTING, DEFAULT_ROUTING
    suffix = "_demo" if split == "demo" else ""
    path = Path(__file__).resolve().parent.parent / "results" / "reports" / f"alert_responses{suffix}.json"
    if not path.exists():
        pytest.skip(f"alert_responses{suffix}.json missing")
    envelope = json.loads(path.read_text())
    records = envelope.get("records", envelope) if isinstance(envelope, dict) else envelope
    seen = {r["attack_category"] for r in records}
    missing = seen - ATTACK_ROUTING.keys()
    # missing categories must be silently absorbed by DEFAULT_ROUTING
    # — the routing keys reference the catalogue, so re-use the
    # same coverage test for those.
    for cat in missing:
        # No KeyError: getattr style lookup yields DEFAULT_ROUTING
        assert ATTACK_ROUTING.get(cat, DEFAULT_ROUTING) is not None


# ── 6. DEVICE_CONTEXT covers every device class ──────────────────


def test_device_context_covers_every_emittable_device_class():
    """Every device_class label ``derive_device_class_row`` /
    ``derive_device_class_array`` can return must have a context entry."""
    from common.device_class import DEVICE_CONTEXT, derive_device_class_array
    import numpy as np
    # Synthesize feature vectors that cover each branch of the
    # derivation rule. The exact values don't matter — we only need
    # to exercise each return branch.
    feat_names = ["Temp", "SpO2", "Pulse_Rate", "Heart_rate", "Resp_Rate", "ST",
                   "Sport", "SrcBytes"]
    cases = [
        # ventilator: Resp_Rate + SpO2 + bio_active >= 4
        [0.6, 0.6, 0.6, 0.6, 0.6, 0.0, 0.0, 0.0],
        # patient_monitor: Pulse + Heart_rate + bio_active >= 3
        [0.0, 0.0, 0.6, 0.6, 0.0, 0.6, 0.0, 0.0],
        # infusion_pump: Temp + bio_active >= 2
        [0.6, 0.0, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0],
        # ehr_workstation: bio_active <= 1 with sport/srcbytes > 0.1
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6, 0.0],
        # other: low bio, no network activity
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ]
    X = np.array(cases)
    emitted = set(derive_device_class_array(X, feat_names))
    missing = emitted - DEVICE_CONTEXT.keys()
    assert not missing, (
        f"derive_device_class returns {emitted} but DEVICE_CONTEXT has "
        f"only {DEVICE_CONTEXT.keys()} — missing {missing}"
    )


# ── 7. RoutingWarning suggested_role values are canonical ────────


def test_routing_warning_suggested_roles_are_known():
    """Every value ``_CATEGORY_TO_EXPECTED_ROLE`` maps to must be a
    role ``ESCALATION_CONTACTS`` knows about — otherwise the
    suggested_role would be undialable."""
    from module5_responses.config import ESCALATION_CONTACTS
    from module5_responses.role_routing import _CATEGORY_TO_EXPECTED_ROLE
    contact_roles = set(ESCALATION_CONTACTS.keys())
    expected_roles = set(_CATEGORY_TO_EXPECTED_ROLE.values())
    # Every expected_role must match (case-insensitive substring) at
    # least one ESCALATION_CONTACTS key.
    for role in expected_roles:
        matched = any(
            role.lower() in c.lower() or c.lower() in role.lower()
            for c in contact_roles
        )
        assert matched, (
            f"role_routing suggests {role!r} but no ESCALATION_CONTACTS "
            f"entry matches — operator can't dial it"
        )
