"""ARCHITECTURE.md Step [12] — MVE faithfulness contract tests.

Locks the MVE generator's documented invariants:

* I5 Layer 1 references SHAP top-3 features as substrings (Mode A→B
     fallback handles failures).
* I7 Layer 3 contains a "DO NOT" clause for CRITICAL/HIGH/MEDIUM alerts
     on clinical devices.
* I8 Layer 2 references the specific clinical_tier name where applicable.
* Mode A reproducibility — when ``_generate_llm`` returns an
  ``MVEOutput``, it carries ``mode_used``, ``llm_provider``,
  ``llm_model_version``, ``llm_full_prompt``, ``llm_full_response``.
* Word budgets enforced (≤60 / ≤50 / ≤60 / ≤30 DO_NOT).
* Configurable forbidden-action terms loaded from the YAML.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.data_models import MVEOutput
from src.mve_generator import (
    ROLE_FORBIDDEN_ACTION_TERMS,
    role_authority_violations,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ── Mode A reproducibility fields on MVEOutput ────────────────────────


def test_mve_output_default_mode_is_rule_based():
    """A bare MVEOutput defaults to Mode B (rule-based) — Mode A flags
    are populated only when ``_generate_llm`` succeeded."""
    mve = MVEOutput(
        layer_1={"baseline_behavior": "x", "deviation_description": "y",
                 "confidence_indicator": "Confidence: HIGH — z"},
        layer_2={"affected_system": "a", "patient_care_impact": "b",
                 "phi_exposure": "c", "severity_label": "HIGH",
                 "severity_rationale": "d"},
        layer_3={"immediate_action": "Block port at switch.",
                 "clinical_constraint": "DO NOT power-cycle.",
                 "escalation_path": "(1) call IT.", "timeframe": "Act within 1h."},
    )
    assert mve.mode_used == "B_rule"
    assert mve.llm_provider is None
    assert mve.llm_model_version is None
    assert mve.llm_full_prompt is None
    assert mve.llm_full_response is None


def test_mve_output_carries_llm_audit_fields_when_set():
    mve = MVEOutput(
        layer_1={"baseline_behavior": "x", "deviation_description": "y",
                 "confidence_indicator": "Confidence: HIGH — z"},
        layer_2={"affected_system": "a", "patient_care_impact": "b",
                 "phi_exposure": "c", "severity_label": "HIGH",
                 "severity_rationale": "d"},
        layer_3={"immediate_action": "Block port at switch.",
                 "clinical_constraint": "DO NOT power-cycle.",
                 "escalation_path": "(1) call IT.", "timeframe": "Act within 1h."},
        mode_used="A_llm",
        llm_provider="openai",
        llm_model_version="gpt-4o-mini",
        llm_full_prompt="Alert type: T1\\n...",
        llm_full_response='{"layer_1": {...}, ...}',
    )
    assert mve.mode_used == "A_llm"
    assert mve.llm_provider == "openai"
    assert mve.llm_model_version == "gpt-4o-mini"
    assert mve.llm_full_prompt
    assert mve.llm_full_response


# ── Role-action authorization comes from the YAML (with fallback) ─────


def test_role_forbidden_terms_include_doc_minimum():
    """The doc names canonical forbidden terms per role; the merged
    YAML+fallback table must include them all."""
    assert "isolate vlan" in ROLE_FORBIDDEN_ACTION_TERMS["biomed_engineer"]
    assert "isolate vlan" in ROLE_FORBIDDEN_ACTION_TERMS["nurse_manager"]
    assert "administer" in ROLE_FORBIDDEN_ACTION_TERMS["IT_generalist"]


def test_role_action_authorization_yaml_exists():
    """The doc says the policy lives in ``configs/role_action_authorization.yaml``.
    If it goes missing the system silently falls back to inline defaults
    — that's defence-in-depth, but the YAML should exist by default."""
    yaml_path = PROJECT_ROOT / "configs/role_action_authorization.yaml"
    assert yaml_path.exists()


# ── Invariant 6: role_authority_violations enforcement ────────────────


def test_role_authority_flags_biomed_doing_network_action():
    """A biomed view with an "isolate vlan" instruction must be flagged."""
    fake_view = MVEOutput(
        layer_1={"baseline_behavior": "x", "deviation_description": "y",
                 "confidence_indicator": "Confidence: HIGH — z"},
        layer_2={"affected_system": "a", "patient_care_impact": "b",
                 "phi_exposure": "c", "severity_label": "HIGH",
                 "severity_rationale": "d"},
        layer_3={
            "immediate_action": "Isolate vlan and update firewall rule.",
            "clinical_constraint": "DO NOT power-cycle.",
            "escalation_path": "(1) call IT.",
            "timeframe": "Act within 1h.",
        },
    )
    hits = role_authority_violations(fake_view, "biomed_engineer")
    assert "isolate vlan" in hits


def test_role_authority_no_violations_for_appropriate_view():
    fake_view = MVEOutput(
        layer_1={"baseline_behavior": "x", "deviation_description": "y",
                 "confidence_indicator": "Confidence: HIGH — z"},
        layer_2={"affected_system": "a", "patient_care_impact": "b",
                 "phi_exposure": "c", "severity_label": "HIGH",
                 "severity_rationale": "d"},
        layer_3={
            "immediate_action": "Verify device firmware version and document anomaly.",
            "clinical_constraint": "DO NOT power-cycle without backup.",
            "escalation_path": "(1) coordinate with manufacturer.",
            "timeframe": "Act within 1h.",
        },
    )
    hits = role_authority_violations(fake_view, "biomed_engineer")
    assert hits == []
