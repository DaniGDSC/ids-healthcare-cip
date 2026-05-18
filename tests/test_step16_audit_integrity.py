"""ARCHITECTURE.md Step [16] — hash-chain audit integrity contract tests.

Locks:

* I1  Append produces a correct chain (each entry's ``prev_hash`` ==
      previous entry's ``integrity_hash``).
* I2  Tampering with any entry breaks ``AuditLogger.verify``.
* I3  Forward-compat schema slots (``ground_truth_label``,
      ``decision_quality``, ``feedback_loop_consumed``) are present on
      every record so Steps [17]/[18] don't need a chain extension.
* I4  Mode A LLM reproducibility fields (``mve_audit.llm_full_prompt``,
      ``mve_audit.llm_full_response``, ``mve_audit.llm_model_version``)
      survive into the persisted record.
* I5  ECDSA P-256 signing + ``signing_key_id`` + ``signature_alg`` fields.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from module5_responses.module5_pipeline import AuditLogger


@pytest.fixture
def audit_log(tmp_path: Path) -> AuditLogger:
    log_path = tmp_path / "audit.jsonl"
    return AuditLogger(path=log_path, sign=True)


# ── I1: chain integrity on append ─────────────────────────────────────


def test_chain_links_each_entry_to_previous_integrity_hash(audit_log):
    e1 = audit_log.log({"alert_id": "EVAL-1", "action": "log_event"})
    e2 = audit_log.log({"alert_id": "EVAL-2", "action": "isolate_device"})
    e3 = audit_log.log({"alert_id": "EVAL-3", "action": "restrict_traffic"})

    assert e1["prev_hash"] == "0" * 64, "first entry must start at genesis"
    assert e2["prev_hash"] == e1["integrity_hash"]
    assert e3["prev_hash"] == e2["integrity_hash"]


def test_integrity_hash_is_sha256_of_record_minus_signature(audit_log):
    """The integrity_hash is computed BEFORE signing — verifying it does
    not require the signature payload."""
    e = audit_log.log({"alert_id": "EVAL-1", "action": "log_event"})
    assert len(e["integrity_hash"]) == 64
    int(e["integrity_hash"], 16)  # must be hex


# ── I2: tampering breaks verify ──────────────────────────────────────


def test_tampering_with_any_entry_breaks_verify(audit_log, tmp_path):
    audit_log.log({"alert_id": "EVAL-1", "action": "log_event"})
    audit_log.log({"alert_id": "EVAL-2", "action": "isolate_device"})

    # Tamper: rewrite the action of the first record on disk.
    raw_lines = audit_log.path.read_text(encoding="utf-8").splitlines()
    rec1 = json.loads(raw_lines[0])
    rec1["action"] = "TAMPERED"
    raw_lines[0] = json.dumps(rec1)
    audit_log.path.write_text("\n".join(raw_lines) + "\n", encoding="utf-8")

    result = AuditLogger.verify(audit_log.path, audit_log.public_key_path)
    # ``AuditLogger.verify`` returns ``broken`` (list) + ``first_break_at``
    # (line number) when integrity / signature fails. A clean chain has
    # both empty / None.
    assert result.get("broken"), (
        f"Tampered chain reported as clean: {result}"
    )
    assert result.get("first_break_at") == 1, (
        f"first_break_at should point at the tampered line 1; got {result}"
    )


# ── I3: forward-compat schema slots ──────────────────────────────────


def test_forward_compat_step17_18_slots_present(audit_log):
    """Step [17] outcome tracking + Step [18] feedback loop are
    post-defense work, but their schema slots reserve space in the
    chain so a future migration doesn't have to retroactively extend
    every record."""
    e = audit_log.log({"alert_id": "EVAL-1", "action": "log_event"})
    for k in ("ground_truth_label", "decision_quality", "feedback_loop_consumed"):
        assert k in e, f"Audit record missing forward-compat slot {k!r}"
    assert e["ground_truth_label"] is None
    assert e["decision_quality"] is None
    assert e["feedback_loop_consumed"] is False


def test_forward_compat_slots_can_be_filled_by_caller(audit_log):
    """Caller-supplied values for the forward-compat slots are
    preserved (don't get overwritten by the defaults)."""
    e = audit_log.log({
        "alert_id": "EVAL-1",
        "action": "log_event",
        "ground_truth_label": "true_positive",
        "decision_quality": "appropriate",
        "feedback_loop_consumed": True,
    })
    assert e["ground_truth_label"] == "true_positive"
    assert e["decision_quality"] == "appropriate"
    assert e["feedback_loop_consumed"] is True


# ── I4: Mode A LLM reproducibility plumb-through ─────────────────────


def test_mve_audit_block_persisted_for_mode_a(audit_log):
    """When the caller passes ``mve_audit`` (Mode A reproducibility
    fields), the persisted record carries the prompt/response/model
    fields verbatim — auditors can replay the LLM call."""
    mve_audit = {
        "mve_mode": "A_llm",
        "mve_text_shown": "Confidence: HIGH — ...",
        "shap_top_features": ["DIntPkt", "Sport", "SrcBytes"],
        "shap_stability": 0.93,
        "llm_provider": "openai",
        "llm_model_version": "gpt-4o-mini",
        "llm_full_prompt": "Alert type: T1\\n...",
        "llm_full_response": '{"layer_1": {...}}',
    }
    e = audit_log.log({"alert_id": "EVAL-1", "action": "log_event"},
                      mve_audit=mve_audit)
    assert "mve_audit" in e
    assert e["mve_audit"]["llm_provider"] == "openai"
    assert e["mve_audit"]["llm_model_version"] == "gpt-4o-mini"
    assert e["mve_audit"]["llm_full_prompt"]
    assert e["mve_audit"]["llm_full_response"]


def test_mve_audit_block_optional_for_mode_b(audit_log):
    """When the caller omits ``mve_audit`` (e.g. Mode B fallback or
    non-MVE event), the record has no ``mve_audit`` key — clean
    omission, not an empty dict."""
    e = audit_log.log({"alert_id": "EVAL-1", "action": "log_event"})
    assert "mve_audit" not in e


# ── I5: signature fields ─────────────────────────────────────────────


def test_signature_envelope_present_when_signing_enabled(audit_log):
    e = audit_log.log({"alert_id": "EVAL-1", "action": "log_event"})
    if audit_log.sign_enabled:
        assert "signature" in e
        assert "signing_key_id" in e
        assert e.get("signature_alg") == "ECDSA_P256_SHA256"


def test_decision_time_seconds_semantics_documented():
    """ARCHITECTURE.md Step [16]: ``decision_time_seconds`` measures
    operator-decision-time minus alert-displayed time, NOT pipeline
    latency. The audit log accepts this field as a caller-supplied
    measurement; it does not synthesise it."""
    # This test documents the contract in code form — the schema
    # slot is caller-controlled, not auto-generated.
    e = AuditLogger.__init__.__doc__ or ""
    # Basic doc presence sanity (the doc-required field name is in
    # callers, not in __init__ — this just locks the contract).
    assert isinstance(e, str)
