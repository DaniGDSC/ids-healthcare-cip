"""PHI redaction at the LLM-prompt boundary — black-box invariant.

Verifies that ``_build_user_prompt`` never lets SSN / MRN / patient-name /
DOB patterns reach the assembled prompt body, regardless of which
defence layer is upstream.

There are now two cooperating defences:

1. **Allow-list (Path A, primary):** ``_filter_for_llm`` (unit-tested in
   ``tests/test_phi_not_in_llm_prompt.py``) drops any field not on the
   allow-list in ``configs/llm_data_flow.yaml`` and raises on any field
   on the forbidden list. PHI keys are either silently dropped (unknown
   identifiers) or hard-fail (canonical PHI keys).
2. **Sentinel scrub (Path B, defence-in-depth):**
   :func:`src.sanitize_for_log` pattern-replaces SSN / MRN / DOB /
   patient-name substrings inside any string value that does reach the
   serialiser, in case a value squeaks through with a PHI-looking
   substring in an allow-listed field.

This file is the contract test for the combined behaviour — the prompt
never carries a raw PHI substring, period.
"""
from __future__ import annotations

from src.mve_generator import _build_user_prompt


def _phi_payloads() -> tuple[dict, dict, dict, dict]:
    """A worst-case bundle of PHI patterns spread across all four payloads."""
    raw_alert = {
        "device_id": "monitor-1",
        "patient_ssn": "123-45-6789",
        "note": "Patient John Smith presented with elevated HR",
    }
    device_context = {
        "criticality": "CRITICAL",
        "patchable": False,
        "MRN: 8842331": "linked record",
    }
    baseline = {
        "window_days": 90,
        "dob_note": "DOB 01/15/1980 — geriatric baseline",
    }
    user_context = {
        "operator_id": "rn-204",
        "note": "Patient Jane Doe and SSN 999-12-3456 in shift notes",
    }
    return raw_alert, device_context, baseline, user_context


def _phi_strings() -> tuple[str, ...]:
    """Concrete substrings that MUST NOT appear in the assembled prompt."""
    return (
        "123-45-6789",          # SSN
        "999-12-3456",          # SSN
        "MRN: 8842331",         # MRN
        "01/15/1980",           # DOB
        "John Smith",           # patient name
        "Jane Doe",             # patient name
    )


def test_build_user_prompt_strips_phi():
    """Every known PHI pattern in the payloads is replaced before send."""
    raw_alert, device_context, baseline, user_context = _phi_payloads()
    prompt = _build_user_prompt(
        raw_alert=raw_alert,
        device_context=device_context,
        baseline=baseline,
        user_context=user_context,
        alert_type="data_alteration",
        risk_level="HIGH",
    )
    leaked = [s for s in _phi_strings() if s in prompt]
    assert not leaked, (
        f"PHI leaked through the LLM prompt boundary: {leaked!r}. "
        f"Check sanitize_for_log() in src/__init__.py and "
        f"_build_user_prompt in src/mve_generator.py."
    )


def test_build_user_prompt_keeps_allowlisted_payload():
    """Allow-listed non-PHI fields round-trip into the prompt.

    Path A drops anything off the allow-list, so the test fixture uses
    only allow-listed keys to confirm those survive verbatim.
    """
    allow_listed = {
        "alert_id":           "EVAL-0001",
        "alert_type":         "KNOWN_ATTACK",
        "device_class":       "patient_monitor",
        "device_criticality": "CRITICAL",
        "attack_category":    "Spoofing",
    }
    prompt = _build_user_prompt(
        raw_alert=allow_listed,
        device_context=allow_listed,
        baseline={},
        user_context=None,
        alert_type="data_alteration",
        risk_level="HIGH",
    )
    for token in ("EVAL-0001", "patient_monitor", "Spoofing", "data_alteration"):
        assert token in prompt, (
            f"allow-listed token {token!r} missing from prompt — "
            "Path A filter may be over-aggressive."
        )


def test_build_user_prompt_path_a_drops_unknown_fields():
    """Path A is default-deny — fields not on the allow-list never reach
    the prompt, even if they are non-PHI operational context.

    This codifies the audit trade-off: the allow-list is the operational
    contract; new fields must be approved by a YAML edit
    (``configs/llm_data_flow.yaml``) before they cross the boundary.
    """
    payload_with_unknown_keys = {
        "alert_id":      "EVAL-0042",         # allowed
        "device_id":     "monitor-7",         # NOT allowed → dropped
        "operator_note": "shift change 7AM",  # NOT allowed → dropped
    }
    prompt = _build_user_prompt(
        raw_alert=payload_with_unknown_keys,
        device_context={},
        baseline={},
        user_context=None,
        alert_type="data_alteration",
        risk_level="HIGH",
    )
    assert "EVAL-0042" in prompt
    for dropped in ("monitor-7", "operator_note", "shift change"):
        assert dropped not in prompt, (
            f"non-allow-listed token {dropped!r} reached the prompt — "
            "Path A filter is not running."
        )
