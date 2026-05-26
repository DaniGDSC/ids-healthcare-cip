#!/usr/bin/env python3
"""Verify RQ3 audit log integrity + required-field completeness.

Two checks:
  1. Hash-chain integrity: each `integrity_hash` must equal SHA256 of the
     record contents (excluding the hash itself); `prev_hash` must match
     the previous entry's `integrity_hash`.
  2. Required-field completeness per RQ3 spec §3: every entry should have
     alert_id, fusion_class, risk_tier, operator_role, decision_time_seconds,
     operator_confidence, mve_text_shown, shap_features_shown, previous_hash,
     entry_hash, timestamp. Audit_log.jsonl uses a different field naming
     convention from the spec — this verifier maps both vocabularies and
     reports the gap explicitly.

Writes:
  * results/rq3_audit_integrity.json
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = PROJECT_ROOT / "results" / "reports" / "audit_log.jsonl"
OUT = PROJECT_ROOT / "results" / "rq3_audit_integrity.json"

# Spec-mandated fields per RQ3 §3, mapped to actual field name in audit log
# where the canonical schema diverged from the spec wording.
SPEC_FIELDS = {
    "alert_id":              "alert_id",
    "fusion_class":          "ground_truth",     # closest match (label, not detector ensemble)
    "risk_tier":             None,                # NOT PRESENT — gap
    "operator_role":         None,                # NOT PRESENT — gap (action audit, not reviewer audit)
    "decision_time_seconds": None,                # NOT PRESENT — gap
    "operator_confidence":   None,                # NOT PRESENT — gap
    "mve_text_shown":        None,                # NOT PRESENT — gap
    "shap_features_shown":   None,                # NOT PRESENT — gap
    "previous_hash":         "prev_hash",
    "entry_hash":            "integrity_hash",
    "timestamp":             "timestamp",
}


def _canonical_payload(record: dict) -> bytes:
    """Reconstruct the bytes that should have been hashed.

    Matches Module 5's `_canonical_json` (module5_pipeline.py:558) — sort
    keys + compact separators `(",", ":")`. Signature fields are popped
    out FIRST (they don't go into the integrity hash); then integrity_hash
    itself is excluded. See module5_pipeline.py:825-834 for the canonical
    verification logic.
    """
    clean = {k: v for k, v in record.items()
             if k not in ("integrity_hash", "signature", "signing_key_id", "signature_alg")}
    return json.dumps(clean, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _canonical_payload_legacy(record: dict) -> bytes:
    """Legacy JSON encoding (default separators) — used by pre-signature records.

    During the migration window the integrity hash was computed without
    the compact separators; accept that form when verifying unsigned
    records (see module5_pipeline.py:839-844).
    """
    clean = {k: v for k, v in record.items()
             if k not in ("integrity_hash", "signature", "signing_key_id", "signature_alg")}
    return json.dumps(clean, sort_keys=True).encode("utf-8")


def verify_chain(records: list) -> dict:
    """Walk the chain and report integrity status.

    Handles archive boundary: when a `prev_hash == "0"*64` is observed
    mid-stream, treat it as a chain restart (the previous log was archived
    to `audit_archive/` and the current log opens with a fresh genesis).
    This mirrors AuditLogger.verify_chain (module5_pipeline.py:806-814).
    """
    expected_prev = "0" * 64  # genesis prev_hash
    hash_mismatches = []
    chain_breaks = []
    archive_restarts = []

    for i, rec in enumerate(records):
        # 1. prev_hash must match the previous entry's integrity_hash —
        # OR be the genesis hash (chain restart from archive).
        rec_prev = rec.get("prev_hash")
        if rec_prev != expected_prev:
            if rec_prev == "0" * 64 and i > 0:
                # Treat as archive restart (legitimate per module5 logic)
                archive_restarts.append({"index": i, "alert_id": rec.get("alert_id")})
            else:
                chain_breaks.append({
                    "index": i,
                    "alert_id": rec.get("alert_id"),
                    "expected_prev": expected_prev,
                    "got_prev": rec_prev,
                })

        # 2. integrity_hash must equal SHA256(canonical_payload). Try the
        # canonical compact form first, fall back to legacy (default sep)
        # for unsigned records.
        claimed = rec.get("integrity_hash")
        actual = hashlib.sha256(_canonical_payload(rec)).hexdigest()
        if actual != claimed:
            # Try legacy form
            legacy = hashlib.sha256(_canonical_payload_legacy(rec)).hexdigest()
            if legacy == claimed:
                actual = claimed  # match via legacy variant
            else:
                hash_mismatches.append({
                    "index": i,
                    "alert_id": rec.get("alert_id"),
                    "expected_canonical": actual,
                    "expected_legacy": legacy,
                    "got": claimed,
                })

        expected_prev = claimed

    return {
        "n_records": len(records),
        "chain_breaks": chain_breaks,
        "hash_mismatches": hash_mismatches,
        "archive_restarts": archive_restarts,
        "chain_intact": len(chain_breaks) == 0,
        "all_hashes_valid": len(hash_mismatches) == 0,
    }


def verify_field_completeness(records: list) -> dict:
    """Compute spec-field-presence per record + aggregate gap analysis."""
    field_presence = {f: 0 for f in SPEC_FIELDS}
    n = len(records)
    for rec in records:
        for spec_field, actual_field in SPEC_FIELDS.items():
            if actual_field and actual_field in rec and rec[actual_field] is not None:
                field_presence[spec_field] += 1

    gap_summary = []
    for spec_field, count in field_presence.items():
        actual = SPEC_FIELDS[spec_field]
        if actual is None:
            gap_summary.append({
                "spec_field": spec_field,
                "status": "missing_from_schema",
                "note": (
                    "Field is not part of the Module-5 AuditTrailWriter schema. "
                    "Reviewer-attributed events go through the separate "
                    "HardenedAuditLogger chain (results/reports/audit_log.jsonl "
                    "events with event_type=reviewer_interaction or alert_responses)."
                ),
            })
        else:
            gap_summary.append({
                "spec_field": spec_field,
                "actual_field": actual,
                "present_count": count,
                "coverage_rate": count / n if n else 0.0,
                "status": "ok" if count == n else "partial",
            })

    return {
        "n_records": n,
        "spec_field_coverage": gap_summary,
        "missing_from_schema_count": sum(
            1 for g in gap_summary if g["status"] == "missing_from_schema"
        ),
    }


def main():
    records = []
    with open(LOG_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    print(f"Loaded {len(records)} audit records from {LOG_PATH}")

    integrity = verify_chain(records)
    completeness = verify_field_completeness(records)

    report = {
        "_meta": {
            "description": "RQ3 audit log integrity + RQ3 spec §3 field completeness",
            "source": str(LOG_PATH.relative_to(PROJECT_ROOT)),
            "n_records": len(records),
        },
        "hash_chain_integrity": integrity,
        "spec_field_completeness": completeness,
        "overall_status": (
            "PASS"
            if (integrity["chain_intact"] and integrity["all_hashes_valid"])
            else "FAIL"
        ),
    }

    with open(OUT, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print("\n=== Hash-chain integrity ===")
    print(f"  Chain intact:        {integrity['chain_intact']}")
    print(f"  All hashes valid:    {integrity['all_hashes_valid']}")
    print(f"  Archive restarts:    {len(integrity.get('archive_restarts', []))}")
    print(f"  Chain breaks:        {len(integrity['chain_breaks'])}")
    print(f"  Hash mismatches:     {len(integrity['hash_mismatches'])}")
    if integrity["chain_breaks"]:
        print(f"  First chain break:   index {integrity['chain_breaks'][0]['index']}")
    if integrity["hash_mismatches"]:
        print(f"  First hash mismatch: index {integrity['hash_mismatches'][0]['index']}")

    print("\n=== Spec §3 field completeness ===")
    for entry in completeness["spec_field_coverage"]:
        if entry["status"] == "ok":
            print(f"  ✓ {entry['spec_field']:25s} → {entry['actual_field']:25s} ({entry['coverage_rate']*100:.0f}%)")
        elif entry["status"] == "partial":
            print(f"  ⚠ {entry['spec_field']:25s} → {entry['actual_field']:25s} ({entry['coverage_rate']*100:.0f}%)")
        else:
            print(f"  ✗ {entry['spec_field']:25s} — {entry['status']}")

    print(f"\nOVERALL: {report['overall_status']}")
    print(f"  → wrote {OUT.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
