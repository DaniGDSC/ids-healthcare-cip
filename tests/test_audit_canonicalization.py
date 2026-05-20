"""Unit tests for common/audit_canonicalization.py.

These verify the read-side hash helpers produce byte-for-byte the same
canonical JSON / integrity_hash that the writer
(module5_responses.module5_pipeline.AuditLogger) produces.
"""
from __future__ import annotations

import hashlib
import json

import pytest

from common.audit_canonicalization import (
    GENESIS_PREV_HASH,
    canonical_json,
    compute_integrity_hash,
    verify_entry_hash,
)


def test_canonical_json_is_sorted_no_whitespace():
    out = canonical_json({"b": 1, "a": 2})
    assert out == b'{"a":2,"b":1}'


def test_canonical_json_key_order_invariant():
    a = canonical_json({"a": 1, "b": 2, "c": 3})
    b = canonical_json({"c": 3, "b": 2, "a": 1})
    assert a == b


def test_canonical_json_matches_writer_internal():
    """Same impl as module5_pipeline._canonical_json — assert parity."""
    from module5_responses.module5_pipeline import _canonical_json
    record = {"alert_id": "X", "value": 42, "nested": {"z": 1, "a": 2}}
    assert canonical_json(record) == _canonical_json(record)


def test_genesis_prev_hash_is_64_zeros():
    assert GENESIS_PREV_HASH == "0" * 64
    assert len(GENESIS_PREV_HASH) == 64


def test_compute_integrity_hash_requires_prev_hash():
    with pytest.raises(ValueError, match="prev_hash"):
        compute_integrity_hash({"alert_id": "X"})


def test_compute_integrity_hash_rejects_envelope_fields():
    with pytest.raises(ValueError, match="signature envelope"):
        compute_integrity_hash({
            "prev_hash": GENESIS_PREV_HASH, "alert_id": "X",
            "integrity_hash": "a" * 64,
        })


def test_compute_integrity_hash_is_deterministic():
    body = {"prev_hash": GENESIS_PREV_HASH, "alert_id": "X", "action": "log"}
    h1 = compute_integrity_hash(body)
    h2 = compute_integrity_hash(dict(body))
    assert h1 == h2
    assert len(h1) == 64
    int(h1, 16)  # hex check


def test_chain_link_changes_hash():
    body_a = {"prev_hash": GENESIS_PREV_HASH, "alert_id": "X"}
    body_b = {"prev_hash": "1" * 64, "alert_id": "X"}
    assert compute_integrity_hash(body_a) != compute_integrity_hash(body_b)


def test_verify_entry_hash_round_trip():
    body = {"prev_hash": GENESIS_PREV_HASH, "alert_id": "X", "action": "log"}
    h = compute_integrity_hash(body)
    entry = dict(body, integrity_hash=h)
    result = verify_entry_hash(entry, GENESIS_PREV_HASH)
    assert result["is_valid"]
    assert result["prev_hash_match"]
    assert result["integrity_hash_match"]


def test_verify_detects_body_tamper():
    body = {"prev_hash": GENESIS_PREV_HASH, "alert_id": "X", "action": "log"}
    h = compute_integrity_hash(body)
    tampered = dict(body, action="TAMPERED", integrity_hash=h)
    result = verify_entry_hash(tampered, GENESIS_PREV_HASH)
    assert not result["is_valid"]
    assert result["prev_hash_match"]
    assert not result["integrity_hash_match"]


def test_verify_detects_chain_break():
    body = {"prev_hash": GENESIS_PREV_HASH, "alert_id": "X"}
    h = compute_integrity_hash(body)
    entry = dict(body, integrity_hash=h)
    result = verify_entry_hash(entry, "1" * 64)
    assert not result["is_valid"]
    assert not result["prev_hash_match"]


def test_verify_ignores_signature_envelope():
    """Signature fields are stripped before recomputing integrity_hash."""
    body = {"prev_hash": GENESIS_PREV_HASH, "alert_id": "X"}
    h = compute_integrity_hash(body)
    entry = dict(body, integrity_hash=h,
                 signature="bogus-base64==",
                 signing_key_id="ecdsa-test",
                 signature_alg="ECDSA_P256_SHA256")
    result = verify_entry_hash(entry, GENESIS_PREV_HASH)
    assert result["is_valid"], result


def test_compute_integrity_matches_writer_output(tmp_path):
    """End-to-end: write via AuditLogger, recompute via wrapper."""
    from module5_responses.module5_pipeline import AuditLogger

    log_path = tmp_path / "audit.jsonl"
    logger = AuditLogger(path=log_path, sign=False)  # no ECDSA for portability
    written = logger.log({"alert_id": "EVAL-1", "action": "log_event"})

    # Reproduce via wrapper
    body = {k: v for k, v in written.items() if k != "integrity_hash"}
    h = compute_integrity_hash(body)
    assert h == written["integrity_hash"], (
        f"Wrapper hash {h[:12]}... != writer hash "
        f"{written['integrity_hash'][:12]}..."
    )
