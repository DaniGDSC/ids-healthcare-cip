"""Module 5 audit — signing primitives + AuditLogger chain + verify + rotate."""
from __future__ import annotations

import json

import pytest

from module5_responses.audit.logger import AuditLogger
from module5_responses.audit.signing import (
    _canonical_json,
    _HAVE_CRYPTOGRAPHY,
    _load_signing_key,
)


# ── _canonical_json determinism (Y10 fix) ──────────────────────────────


def test_canonical_json_sorted_keys_compact_separators():
    data = {"b": 2, "a": 1, "c": [3, 4]}
    out = _canonical_json(data)
    assert out == b'{"a":1,"b":2,"c":[3,4]}'


def test_canonical_json_deterministic_across_dict_orderings():
    d1 = {"x": 1, "y": 2, "z": 3}
    d2 = {"z": 3, "y": 2, "x": 1}
    assert _canonical_json(d1) == _canonical_json(d2)


def test_canonical_json_utf8_bytes():
    out = _canonical_json({"name": "ECG"})
    assert isinstance(out, bytes)
    assert out.decode("utf-8") == '{"name":"ECG"}'


# ── _load_signing_key bootstrap ────────────────────────────────────────


@pytest.mark.skipif(not _HAVE_CRYPTOGRAPHY, reason="cryptography not installed")
def test_load_signing_key_bootstraps_when_missing(tmp_path):
    priv = tmp_path / "k.pem"
    pub = tmp_path / "k.pub.pem"
    assert not priv.exists()
    key, ret_pub, key_id = _load_signing_key(priv, pub)
    assert priv.exists()
    assert pub.exists()
    assert ret_pub == pub
    assert key_id.startswith("ecdsa-p256-")


@pytest.mark.skipif(not _HAVE_CRYPTOGRAPHY, reason="cryptography not installed")
def test_load_signing_key_reuses_existing(tmp_path):
    priv = tmp_path / "k.pem"
    pub = tmp_path / "k.pub.pem"
    _, _, kid1 = _load_signing_key(priv, pub)
    _, _, kid2 = _load_signing_key(priv, pub)
    assert kid1 == kid2


# ── AuditLogger end-to-end ─────────────────────────────────────────────


def _make_logger(tmp_path):
    return AuditLogger(
        tmp_path / "audit.jsonl",
        signing_key_path=tmp_path / "priv.pem",
        public_key_path=tmp_path / "pub.pem",
        retention_days=365,
    )


def test_audit_logger_appends_and_chains(tmp_path):
    al = _make_logger(tmp_path)
    r1 = al.log({"event": "first"})
    r2 = al.log({"event": "second"})
    assert r1["prev_hash"] == "0" * 64
    assert r2["prev_hash"] == r1["integrity_hash"]
    # File contains two lines.
    lines = (tmp_path / "audit.jsonl").read_text().strip().split("\n")
    assert len(lines) == 2


def test_audit_logger_restart_continues_chain(tmp_path):
    al1 = _make_logger(tmp_path)
    r1 = al1.log({"event": "before-restart"})
    # New logger pointing at same file.
    al2 = _make_logger(tmp_path)
    assert al2.prev_hash == r1["integrity_hash"]
    r2 = al2.log({"event": "after-restart"})
    assert r2["prev_hash"] == r1["integrity_hash"]


def test_audit_logger_verify_clean_chain(tmp_path):
    al = _make_logger(tmp_path)
    for i in range(3):
        al.log({"event": f"e{i}"})
    report = AuditLogger.verify(al.path, al.public_key_path)
    assert report["total"] == 3
    assert report["first_break_at"] is None


def test_audit_logger_verify_detects_tampered_body(tmp_path):
    al = _make_logger(tmp_path)
    al.log({"event": "orig-1"})
    al.log({"event": "orig-2"})

    # Tamper with the body of line 2.
    lines = al.path.read_text().strip().split("\n")
    rec = json.loads(lines[1])
    rec["event"] = "tampered"
    lines[1] = json.dumps(rec)
    al.path.write_text("\n".join(lines) + "\n")

    report = AuditLogger.verify(al.path, al.public_key_path)
    assert report["first_break_at"] == 2


def test_audit_logger_verify_missing_file(tmp_path):
    report = AuditLogger.verify(tmp_path / "nope.jsonl")
    assert report["total"] == 0
    assert any("does not exist" in b["reason"] for b in report["broken"])


def test_audit_logger_reviewer_block_present(tmp_path):
    al = _make_logger(tmp_path)
    rec = al.log(
        {"event": "review"},
        reviewer_id="P03",
        reviewer_role="Security Analyst",
        review_action="confirm",
    )
    assert rec["reviewer"]["reviewer_id"] == "P03"
    assert rec["reviewer"]["reviewer_role"] == "Security Analyst"
    assert rec["reviewer"]["review_action"] == "confirm"
    # Signature still valid after reviewer block.
    report = AuditLogger.verify(al.path, al.public_key_path)
    assert report["first_break_at"] is None


def test_audit_logger_rotate_empty_log_no_op(tmp_path):
    al = _make_logger(tmp_path)
    report = al.rotate_and_purge(retention_days=365)
    assert report["rotated"] is False
    assert "empty or missing" in report["reason"]


def test_audit_logger_rotate_recent_log_no_op(tmp_path):
    al = _make_logger(tmp_path)
    al.log({"event": "fresh"})
    report = al.rotate_and_purge(retention_days=365)
    assert report["rotated"] is False
    assert "retention window" in report["reason"]


def test_audit_logger_signing_disabled_still_chains(tmp_path):
    al = AuditLogger(
        tmp_path / "no_sign.jsonl",
        signing_key_path=tmp_path / "priv.pem",
        public_key_path=tmp_path / "pub.pem",
        sign=False,
    )
    r1 = al.log({"event": "a"})
    r2 = al.log({"event": "b"})
    assert "signature" not in r1
    assert "signature" not in r2
    assert r2["prev_hash"] == r1["integrity_hash"]
    # Verify with legacy_ok=True passes.
    report = AuditLogger.verify(al.path, legacy_ok=True)
    assert report["first_break_at"] is None
    assert report["valid_legacy"] == 2
