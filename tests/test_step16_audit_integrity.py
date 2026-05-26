"""RQ3 Invariant tests — Step 16 audit-log hash-chain integrity.

Mirrors the verification logic in `tools/rq3_verify_audit.py` but as
pytest fixtures so the suite can be invoked by CI. Verifies:

  1. Each entry's `integrity_hash` equals SHA256(canonical_json(record \
     minus integrity_hash + signature fields)).
  2. Each entry's `prev_hash` matches the previous entry's
     `integrity_hash`, with `prev_hash == "0"*64` indicating a legitimate
     archive restart.
  3. No entry is missing required fields (prev_hash, integrity_hash,
     timestamp, alert_id).
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

LOG_PATH = PROJECT_ROOT / "results/reports/audit_log.jsonl"


def _canonical_json(record: dict) -> bytes:
    """Same as module5_pipeline._canonical_json — compact separators."""
    return json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _canonical_json_legacy(record: dict) -> bytes:
    """Default-separator form used by unsigned pre-migration records."""
    return json.dumps(record, sort_keys=True).encode("utf-8")


def _compute_integrity(record: dict) -> tuple[str, str]:
    """Return (canonical_hash, legacy_hash) for the record."""
    clean = {k: v for k, v in record.items()
             if k not in ("integrity_hash", "signature",
                          "signing_key_id", "signature_alg")}
    return (
        hashlib.sha256(_canonical_json(clean)).hexdigest(),
        hashlib.sha256(_canonical_json_legacy(clean)).hexdigest(),
    )


@pytest.fixture(scope="module")
def audit_records():
    if not LOG_PATH.exists():
        pytest.skip(f"{LOG_PATH} missing — run module5 pipeline first")
    records = []
    with open(LOG_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    if not records:
        pytest.skip("audit log is empty")
    return records


def test_step16_required_fields_present(audit_records):
    """Every record must have the chain-verification structural fields
    (prev_hash, integrity_hash). Non-alert events (reviewer interactions,
    phase0 security events) use a different schema where `timestamp` is
    named `logged_at` and `alert_id` may be absent — those are filtered
    out of the alert-id/timestamp check.
    """
    chain_fields = ("prev_hash", "integrity_hash")
    missing_chain = [(i, f) for i, r in enumerate(audit_records)
                     for f in chain_fields if f not in r]
    assert not missing_chain, (
        f"{len(missing_chain)} records lack chain fields. "
        f"First: {missing_chain[:5]}"
    )

    # Alert-audit records (event_type is unset) must additionally carry
    # alert_id + timestamp.
    alert_records = [r for r in audit_records if not r.get("event_type")]
    missing_alert = []
    for i, r in enumerate(alert_records):
        for f in ("alert_id", "timestamp"):
            if f not in r:
                missing_alert.append((i, f))
    assert not missing_alert, (
        f"{len(missing_alert)} alert-audit records lack alert_id/timestamp. "
        f"First: {missing_alert[:5]}"
    )


def test_step16_integrity_hash_matches_canonical(audit_records):
    """SHA256 of canonical-JSON-minus-hash equals the stored integrity_hash."""
    mismatches = []
    for i, r in enumerate(audit_records):
        canonical, legacy = _compute_integrity(r)
        stored = r.get("integrity_hash")
        if stored != canonical and stored != legacy:
            mismatches.append({
                "index": i,
                "alert_id": r.get("alert_id"),
                "canonical": canonical[:16],
                "legacy": legacy[:16],
                "stored": str(stored)[:16],
            })
    assert not mismatches, (
        f"{len(mismatches)} integrity-hash mismatches — chain tampered. "
        f"First: {mismatches[:3]}"
    )


def test_step16_prev_hash_chains_correctly(audit_records):
    """prev_hash must equal previous record's integrity_hash, with
    explicit archive-restart tolerance when prev_hash == '0'*64 mid-stream.
    """
    expected_prev = "0" * 64
    breaks = []
    restarts = []
    for i, r in enumerate(audit_records):
        rec_prev = r.get("prev_hash")
        if rec_prev == expected_prev:
            pass  # ok
        elif rec_prev == "0" * 64 and i > 0:
            # Legitimate archive restart
            restarts.append(i)
        else:
            breaks.append({
                "index": i,
                "alert_id": r.get("alert_id"),
                "expected": expected_prev[:16],
                "got": str(rec_prev)[:16],
            })
        expected_prev = r.get("integrity_hash") or expected_prev
    assert not breaks, (
        f"{len(breaks)} chain breaks (and {len(restarts)} archive restarts). "
        f"First break: {breaks[:3]}"
    )


def test_step16_chain_genesis_is_zero_hash(audit_records):
    """The first record's prev_hash must be all zeros (genesis marker)."""
    assert audit_records[0]["prev_hash"] == "0" * 64, (
        f"first record prev_hash is {audit_records[0]['prev_hash']!r}, "
        f"expected genesis hash (all zeros)"
    )


def test_step16_timestamps_monotonic(audit_records):
    """Within a single chain segment timestamps should be non-decreasing.

    Resets at archive boundaries are allowed. We test a soft monotonic
    property — accept up to 1% out-of-order entries (operator timezone
    skew, late arrival).
    """
    out_of_order = 0
    prev_ts = None
    for r in audit_records:
        ts = r.get("timestamp")
        if prev_ts is not None and ts is not None and ts < prev_ts:
            # Reset at archive boundary is fine — heuristic: clock jumped
            # backward by more than a day = treat as new segment
            try:
                from datetime import datetime
                a = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                b = datetime.fromisoformat(prev_ts.replace("Z", "+00:00"))
                if (b - a).total_seconds() < 86400:
                    out_of_order += 1
            except (ValueError, AttributeError):
                out_of_order += 1
        prev_ts = ts
    fraction = out_of_order / max(1, len(audit_records))
    assert fraction < 0.01, (
        f"{out_of_order} / {len(audit_records)} timestamps out of order "
        f"({fraction*100:.2f}% > 1% tolerance)"
    )
