"""ARCHITECTURE.md Step [12], Mode A — PHI must not cross the LLM API boundary.

Locks the contract that ``_filter_for_llm`` whittles every dict bound
for the Anthropic API down to the explicit allow-list in
``configs/llm_data_flow.yaml``. The forbidden list (patient_id, MRN,
DOB, SSN, EHR fields, ...) raises a hard ``AssertionError`` if it
ever appears in an alert payload — the system refuses to silently
honor a request that would leak PHI.

Why this matters
----------------
HIPAA compliance for an LLM-backed clinical reasoning layer requires:

1. Affirmative allow-list (default-deny), not deny-list.
2. Forbidden-field guard with hard failure (no warning-only).
3. Full prompt + response logging for audit reproducibility.

This module covers (1) and (2). Reproducibility logging is verified
by ``tests/test_step12_mve_faithfulness.py``.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.mve_generator import (
    _filter_for_llm,
    _load_llm_data_flow,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LLM_DATA_FLOW_YAML = PROJECT_ROOT / "configs" / "llm_data_flow.yaml"


# ── YAML shape ────────────────────────────────────────────────────────


def test_llm_data_flow_yaml_exists():
    assert LLM_DATA_FLOW_YAML.exists(), (
        f"{LLM_DATA_FLOW_YAML} missing — Mode A LLM cannot enforce its "
        "PHI allow-list without the config."
    )


def test_llm_data_flow_yaml_has_required_sections():
    body = yaml.safe_load(LLM_DATA_FLOW_YAML.read_text(encoding="utf-8"))
    assert isinstance(body, dict)
    inputs = body.get("mode_a_llm_inputs") or {}
    assert "allowed" in inputs and isinstance(inputs["allowed"], list)
    assert "forbidden" in inputs and isinstance(inputs["forbidden"], list)
    assert len(inputs["allowed"]) > 0, "allow-list must not be empty"
    assert len(inputs["forbidden"]) > 0, "forbidden list must not be empty"


def test_forbidden_list_includes_canonical_phi_fields():
    """The doc enumerates a minimum set of PHI fields that must always
    appear on the forbidden list. If a maintainer accidentally trims
    them, this test catches it."""
    cfg = _load_llm_data_flow()
    forbidden = set(cfg["forbidden"])
    canonical = {
        "patient_id",
        "patient_name",
        "mrn",
        "medical_record_number",
        "ssn",
        "dob",
        "date_of_birth",
        "ehr_record",
    }
    missing = canonical - forbidden
    assert not missing, (
        f"PHI fields {sorted(missing)} are missing from the forbidden "
        f"list in {LLM_DATA_FLOW_YAML.name}. They MUST always be denied."
    )


# ── Filtering: drop non-allowlisted ───────────────────────────────────


def test_filter_for_llm_drops_unknown_fields_silently():
    """Fields that aren't allowed AND aren't forbidden are silently
    dropped — default-deny semantics. New schema fields are invisible
    to the LLM until explicitly approved by a YAML edit."""
    payload = {
        "alert_id": "EVAL-0001",          # allowed
        "device_class": "ventilator",     # allowed
        "made_up_new_field": "secret",    # not allowed → dropped
    }
    out = _filter_for_llm(payload)
    assert "alert_id" in out
    assert "device_class" in out
    assert "made_up_new_field" not in out


# ── Filtering: hard-fail on forbidden ─────────────────────────────────


@pytest.mark.parametrize(
    "phi_key",
    ["patient_id", "mrn", "ssn", "dob", "ehr_record"],
)
def test_filter_for_llm_raises_on_forbidden_field(phi_key: str):
    """Presence of any explicitly-forbidden field triggers a hard
    ``AssertionError``. The system refuses to silently honor a request
    that would leak PHI."""
    payload = {
        "alert_id": "EVAL-9999",
        phi_key: "PII-PAYLOAD",
    }
    with pytest.raises(AssertionError, match="PHI red flag"):
        _filter_for_llm(payload)


def test_filter_for_llm_passes_typical_alert_intact():
    """The realistic alert dict produced by Module 6's curation has
    only allow-listed fields — filtering is a no-op."""
    realistic = {
        "alert_id": "EVAL-0301",
        "alert_type": "KNOWN_ATTACK",
        "fusion_class": "KNOWN_ATTACK",
        "risk_level": "HIGH",
        "risk_score": 0.78,
        "device_class": "patient_monitor",
        "device_criticality": "HIGH",
        "patchable": False,
        "attack_category": "Spoofing",
        "data_quality": "OK",
    }
    out = _filter_for_llm(realistic)
    assert out == realistic, (
        "All fields in this realistic alert are on the allow-list — "
        "filter must be a no-op."
    )


# ── Allow + forbidden lists must be disjoint ──────────────────────────


def test_allowed_and_forbidden_are_disjoint():
    cfg = _load_llm_data_flow()
    overlap = set(cfg["allowed"]) & set(cfg["forbidden"])
    assert not overlap, (
        f"Fields {sorted(overlap)} appear in BOTH allowed and forbidden "
        "lists. The YAML must be unambiguous."
    )
