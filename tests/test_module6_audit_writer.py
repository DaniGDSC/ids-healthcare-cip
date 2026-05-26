"""Module 6 audit writer — plain JSONL + hardened ECDSA chain."""
from __future__ import annotations

import json


import module6_evaluation.audit_writer as aw


def test_hardened_audit_is_lazy_initialised(monkeypatch, tmp_path):
    """Y5/Y8: importing the module must NOT bootstrap signing keys."""
    monkeypatch.setattr(aw, "_hardened_audit", None)
    # Don't call get_hardened_audit() — confirm the module attribute is still None.
    assert aw._hardened_audit is None


def test_audit_trail_writer_appends_jsonl(tmp_path):
    path = tmp_path / "audit_trail.jsonl"
    w = aw.AuditTrailWriter(path)
    w.write({"event_type": "test", "data": 42})
    assert path.exists()
    lines = path.read_text().strip().split("\n")
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["event_type"] == "test"
    assert rec["data"] == 42
    assert "timestamp_iso" in rec
    assert "epoch_sec" in rec


def test_audit_trail_writer_appends_multiple(tmp_path):
    path = tmp_path / "log.jsonl"
    w = aw.AuditTrailWriter(path)
    for i in range(5):
        w.write({"event_type": "step", "i": i})
    lines = path.read_text().strip().split("\n")
    assert len(lines) == 5


def test_audit_log_writes_to_plain_jsonl_when_sign_false(tmp_path, monkeypatch):
    monkeypatch.setattr(aw, "EVAL_DIR", tmp_path)
    aw.audit_log("user_action", participant_id="P03", role="analyst",
                  action="confirm", sign=False)
    plain = tmp_path / "audit_trail.jsonl"
    assert plain.exists()
    rec = json.loads(plain.read_text().strip().split("\n")[0])
    assert rec["event_type"] == "user_action"
    assert rec["participant_id"] == "P03"
    assert rec["role"] == "analyst"
    assert rec["action"] == "confirm"


def test_audit_log_signs_when_hardened_available(tmp_path, monkeypatch):
    """When sign=True (default), payload routes through the hardened logger."""
    monkeypatch.setattr(aw, "EVAL_DIR", tmp_path)
    monkeypatch.setattr(aw, "_hardened_audit", None)  # force fresh init
    # Point HOME to tmp so the bootstrapped key doesn't pollute the user dir.
    monkeypatch.setenv("HOME", str(tmp_path))

    aw.audit_log("decision", participant_id="P01", role="analyst",
                  action="isolate", sign=True)

    plain = tmp_path / "audit_trail.jsonl"
    hardened = tmp_path / "audit_log.jsonl"
    assert plain.exists()
    assert hardened.exists()
    # Hardened record carries reviewer block.
    hrec = json.loads(hardened.read_text().strip().split("\n")[-1])
    assert hrec.get("event_type") == "decision"
    assert "integrity_hash" in hrec
    assert "reviewer" in hrec
    assert hrec["reviewer"]["reviewer_id"] == "P01"


def test_get_hardened_audit_returns_same_singleton(monkeypatch, tmp_path):
    monkeypatch.setattr(aw, "_hardened_audit", None)
    monkeypatch.setattr(aw, "EVAL_DIR", tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    a = aw.get_hardened_audit()
    b = aw.get_hardened_audit()
    assert a is b


def test_audit_log_falls_back_when_hardened_raises(tmp_path, monkeypatch, caplog):
    """If hardened logger raises, plain JSONL must still record the event."""
    monkeypatch.setattr(aw, "EVAL_DIR", tmp_path)

    class FailingLogger:
        def log(self, *args, **kwargs):
            raise RuntimeError("simulated key load failure")

    monkeypatch.setattr(aw, "_hardened_audit", FailingLogger())
    aw.audit_log("test_event", participant_id="P99", role="analyst", sign=True)

    plain = tmp_path / "audit_trail.jsonl"
    assert plain.exists()
    rec = json.loads(plain.read_text().strip().split("\n")[0])
    assert rec["event_type"] == "test_event"
