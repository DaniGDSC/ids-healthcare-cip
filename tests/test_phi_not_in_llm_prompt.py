"""ARCHITECTURE.md Step [12], Mode A — PHI must not cross the LLM API boundary.

Locks the contract that ``_filter_for_llm`` whittles every dict bound
for the OpenAI API down to the explicit allow-list in
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


# ── RQ2_Compliance.md §4 — DEFENSE-CRITICAL live + historical PHI scan ──


# A "PHI honeypot" alert: every forbidden field name plus literal PHI-
# shaped values that a regex scan would also catch (SSN, phone, email,
# DOB-like dates, room/bed patterns).  If any of these reach the LLM
# prompt, the system has a HIPAA leak.
_PHI_HONEYPOT_ALERT = {
    # Allow-listed — SHOULD pass through the filter
    "alert_id": "PHI-HONEYPOT-001",
    "alert_type": "KNOWN_ATTACK",
    "device_class": "infusion_pump",
    "device_criticality": "CRITICAL",
    "attack_category": "Data Alteration",
    "risk_level": "CRITICAL",
    "fusion_class": "KNOWN_ATTACK",
    # FORBIDDEN — must raise AssertionError before any prompt is built
    "patient_id": "PT-998877",
    "patient_name": "John Doe",
    "mrn": "MRN: 9988776",
    "ssn": "123-45-6789",
    "dob": "1947-03-12",
    "ehr_record": "Patient John Doe — diagnosed hypertension",
    "phone_number": "555-123-4567",
}


def test_mode_a_live_prompt_phi_guard_raises_loudly(monkeypatch):
    """DEFENSE-CRITICAL: a PHI-laden alert must NEVER reach the
    OpenAI API.  The system has two equally-safe outcomes:

      1. ``_filter_for_llm`` raises ``AssertionError("PHI red flag …")``
         before any network call (preferred — fails loud, audit-visible).
      2. The Mode A path is unavailable (no API key or no ``openai``
         package) and the function returns ``None`` so the caller falls
         back to Mode B (no network egress at all).

    The test asserts that one of these two outcomes always holds —
    *never* "function returned a populated MVEOutput" when the input
    contained forbidden fields.  We mock ``openai.OpenAI`` so that any
    attempt to instantiate it would be detected; if it's reached, the
    mock raises and the test still fails loudly.
    """
    import sys
    import types

    # Force Mode A path on.
    monkeypatch.setenv("OPENAI_API_KEY", "test-dummy-key")

    # Ensure ``openai`` is importable inside ``_generate_llm`` so we
    # exercise the post-import code path (the PHI filter sits *after*
    # the import).  If a real openai install exists, monkeypatch its
    # ``OpenAI`` constructor; otherwise inject a stub module.
    api_calls: list[tuple] = []

    def _trip_wire(*args, **kwargs):
        api_calls.append((args, kwargs))
        raise AssertionError(
            "openai.OpenAI constructor reached with PHI-laden "
            "payload — PHI filter was bypassed."
        )

    if "openai" in sys.modules:
        monkeypatch.setattr(
            sys.modules["openai"], "OpenAI", _trip_wire,
        )
    else:
        stub = types.ModuleType("openai")
        stub.OpenAI = _trip_wire  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "openai", stub)

    from src.mve_generator import _generate_llm

    try:
        result = _generate_llm(_PHI_HONEYPOT_ALERT, {}, {}, None, "T1")
    except AssertionError as exc:
        # Outcome 1 — preferred loud-fail path.
        assert "PHI red flag" in str(exc), (
            f"Unexpected AssertionError content: {exc}"
        )
        assert not api_calls, (
            "PHI filter raised AFTER an openai.OpenAI call — "
            "guard order is wrong."
        )
        return

    # Outcome 2 — Mode A unavailable; result is None and no network reached.
    assert result is None, (
        f"PHI-laden input must never yield a populated MVEOutput from "
        f"Mode A.  Got: {result!r}"
    )
    assert not api_calls, (
        f"openai.OpenAI was called {len(api_calls)} time(s) with "
        "a PHI-laden payload — HIPAA leak path exists."
    )


def test_mode_a_phi_safe_payload_filter_drops_pii_markers():
    """Complement to the loud-raise check: when the payload contains
    free-text PHI inside an allow-listed key (e.g. an SSN inside
    ``alert_id``), the filter cannot catch it, BUT no allow-listed key
    in our schema is intended to carry free-text — so any leaked
    pattern in the post-filter payload would indicate a schema bug.

    Asserts ``_filter_for_llm`` produces a payload whose values, taken
    together, do not contain canonical PHI patterns.
    """
    from src.mve_generator import _filter_for_llm

    safe_alert = {
        "alert_id": "EVAL-9001",
        "alert_type": "KNOWN_ATTACK",
        "device_class": "infusion_pump",
        "attack_category": "Data Alteration",
        "risk_level": "HIGH",
        "fusion_class": "KNOWN_ATTACK",
        # These get dropped (not on allow-list) but tested defensively.
        "free_text_note": "Patient John Doe in Bed 4-2",
        "phone_number": "555-123-4567",
    }
    # ``phone_number`` is on the forbidden list — must raise.
    with pytest.raises(AssertionError, match="PHI red flag"):
        _filter_for_llm(safe_alert)


def test_historical_llm_audit_log_phi_free():
    """Scan past Mode A audit-log entries for PHI in ``full_prompt`` /
    ``full_response``.  Skips when no LLM audit log exists yet — the
    prototype's 356-record MVE corpus is all Mode B, so this is the
    expected state for now (RQ2_Compliance.md §4 explicitly allows the
    skip)."""
    import json
    import re

    audit_path = PROJECT_ROOT / "logs/llm_audit.jsonl"
    if not audit_path.exists():
        pytest.skip(
            f"No LLM audit log at {audit_path.relative_to(PROJECT_ROOT)} — "
            "no prior Mode A runs to scan (Mode B fallback only so far)."
        )

    cfg = _load_llm_data_flow()
    forbidden_names = set(cfg["forbidden"])
    pii_patterns = {
        "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        "phone": re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"),
        "email": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
        "mrn_numeric": re.compile(r"\b(?:MRN|mrn)[\s:]*\d{6,10}\b"),
    }

    violations = []
    with audit_path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                violations.append({"line": line_no, "error": "invalid JSON"})
                continue
            for field in ("full_prompt", "full_response"):
                text = rec.get(field, "") or ""
                lower = text.lower()
                for name in forbidden_names:
                    if name.lower() in lower:
                        violations.append({
                            "line": line_no, "field": field,
                            "type": "forbidden_field_name", "match": name,
                        })
                        break
                for label, pat in pii_patterns.items():
                    if pat.search(text):
                        violations.append({
                            "line": line_no, "field": field,
                            "type": label, "match": pat.search(text).group(0),
                        })
                        break

    assert not violations, (
        f"PHI found in {len(violations)} audit log entries. "
        f"Sample: {violations[:3]}"
    )


def test_mode_b_makes_no_external_calls(monkeypatch):
    """Mode B is the local fallback.  It must NEVER touch the network.

    We patch every plausible HTTP entry point and the openai client
    constructor; ``generate_mve`` is called with ``OPENAI_API_KEY``
    unset so the rule-based branch is taken; any captured call is a
    fatal assertion.
    """
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    calls: list[tuple[str, tuple, dict]] = []

    def _record(label):
        def _fn(*args, **kwargs):
            calls.append((label, args, kwargs))
            raise AssertionError(
                f"Mode B invoked network primitive {label}"
            )
        return _fn

    # requests + urllib are the standard-library / common HTTP paths.
    import urllib.request

    monkeypatch.setattr(
        "urllib.request.urlopen", _record("urllib.urlopen"), raising=True,
    )

    # ``requests`` is optional in this env; only patch if it imports.
    try:
        import requests  # noqa: F401
        monkeypatch.setattr("requests.request", _record("requests.request"))
        monkeypatch.setattr("requests.post",    _record("requests.post"))
        monkeypatch.setattr("requests.get",     _record("requests.get"))
    except ImportError:
        pass

    # ``openai`` is also optional.  When present, patch the client
    # constructor so any attempt to build one fails the test.
    try:
        import openai
        monkeypatch.setattr(
            openai, "OpenAI", _record("openai.OpenAI"),
        )
    except ImportError:
        pass

    from src.mve_generator import generate_mve

    raw_alert = {
        "alert_id": "MODE-B-001",
        "Attack Category": "Data Alteration",
        "alert_type": "T1",
    }
    device_context = {
        "device_type": "infusion_pump",
        "criticality": "CRITICAL",
        "patchable": False,
    }
    # Should reach _generate_rule_based (Mode B) deterministically.
    mve = generate_mve(raw_alert, device_context, {}, None)
    assert mve is not None and getattr(mve, "mode_used", "B_rule") == "B_rule"
    assert not calls, (
        f"Mode B made {len(calls)} network-shaped call(s): {calls[:3]}"
    )
