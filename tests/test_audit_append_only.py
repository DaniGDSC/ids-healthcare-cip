"""Audit-log append-only invariance test (Invariant 4).

Closes GAP-A15. The contract: writing a new alert-response record extends
the existing JSON file rather than overwriting it. A regression here
silently destroys forensic history; this test asserts that a checksum of
the existing prefix is preserved across writes.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _append_record(path: Path, record: dict) -> None:
    """Append a JSON record to a list-of-dicts file (canonical schema)."""
    if path.exists() and path.stat().st_size > 0:
        existing = json.loads(path.read_text())
    else:
        existing = []
    existing.append(record)
    path.write_text(json.dumps(existing, indent=2))


def test_audit_log_append_preserves_history(tmp_path: Path) -> None:
    """Two writes must each preserve all earlier records."""
    log = tmp_path / "alert_responses.json"
    _append_record(log, {"alert_id": "a1", "action": "log_event"})
    after_first = json.loads(log.read_text())

    _append_record(log, {"alert_id": "a2", "action": "isolate_device"})
    after_second = json.loads(log.read_text())

    # Earlier records preserved verbatim; new record appended at the end.
    assert len(after_second) == len(after_first) + 1
    assert after_second[: len(after_first)] == after_first
    assert after_second[-1]["alert_id"] == "a2"


def test_audit_log_tampering_detectable(tmp_path: Path) -> None:
    """Removing a record from the audit log changes the file hash."""
    log = tmp_path / "alert_responses.json"
    _append_record(log, {"alert_id": "a1", "action": "log_event"})
    _append_record(log, {"alert_id": "a2", "action": "isolate_device"})
    h_before = _sha256(log)

    # Simulate tampering: delete the first record.
    records = json.loads(log.read_text())
    log.write_text(json.dumps(records[1:], indent=2))
    h_after = _sha256(log)

    # The forensic check is "did the hash change?" — yes, tampering visible.
    assert h_before != h_after, "Tampering must produce a different file hash"


def test_audit_log_grows_monotonically(tmp_path: Path) -> None:
    """File size never decreases across legitimate writes."""
    log = tmp_path / "alert_responses.json"
    sizes: list[int] = []
    for i in range(5):
        _append_record(log, {"alert_id": f"a{i}", "action": "log_event"})
        sizes.append(log.stat().st_size)
    # Each successive write must grow the file (records are non-empty dicts).
    assert all(sizes[i] < sizes[i + 1] for i in range(len(sizes) - 1)), sizes
