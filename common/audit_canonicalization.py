"""Canonical serialization + hash helpers for the audit log chain.

This module is the read-side counterpart to the writer at
``module5_responses.module5_pipeline.AuditLogger``. It exists so the
chain-verification scripts in ``analysis/`` can compute the same
``integrity_hash`` the writer computed, byte for byte.

Wire format (real, post-Path C adaptation; differs from
RQ3_AUDIT_INTEGRITY_SPEC.md template):

  - Hash chain fields:    ``prev_hash`` (NOT ``previous_hash``)
                          ``integrity_hash`` (NOT ``entry_hash``)
  - Genesis prev_hash:    "0" * 64
  - Canonical encoding:   json.dumps(obj, sort_keys=True,
                                     separators=(",", ":"),
                                     ensure_ascii=True).encode("utf-8")
  - Integrity hash:       SHA256(canonical_json(record))
                          where record contains prev_hash + body but
                          NOT YET integrity_hash, signature,
                          signing_key_id, or signature_alg.
  - Signature envelope:   ECDSA P-256 over the canonical JSON of the
                          record including integrity_hash. Verified by
                          the writer module, not here.

Both the writer (module5_pipeline._canonical_json) and this module use
the exact same json.dumps parameters; if they ever drift, the chain
breaks for the wrong reason. The spec's "previous_hash || body" hash
construction does NOT apply — the real writer puts prev_hash INSIDE the
record before hashing.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

GENESIS_PREV_HASH = "0" * 64

# Fields stripped from a record before recomputing integrity_hash.
# Matches what AuditLogger.verify pops in module5_pipeline.py.
SIGNATURE_ENVELOPE_FIELDS = ("signature", "signing_key_id", "signature_alg")
INTEGRITY_FIELD = "integrity_hash"


def canonical_json(obj: dict[str, Any]) -> bytes:
    """Encode a dict to the canonical JSON bytes used for hashing.

    Identical to ``module5_responses.module5_pipeline._canonical_json``.
    Caller-side stability guarantee: same dict (modulo key order) ->
    same bytes -> same hash.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


def compute_integrity_hash(record_with_prev_hash: dict[str, Any]) -> str:
    """Compute integrity_hash for a record whose ``prev_hash`` is set.

    The record passed in must already contain ``prev_hash`` and the
    payload fields. It must NOT yet contain ``integrity_hash`` or any
    of the signature envelope fields.
    """
    if "prev_hash" not in record_with_prev_hash:
        raise ValueError("record must contain prev_hash before hashing")
    forbidden = {INTEGRITY_FIELD, *SIGNATURE_ENVELOPE_FIELDS}
    overlap = forbidden.intersection(record_with_prev_hash.keys())
    if overlap:
        raise ValueError(
            "record must not contain hash/signature envelope fields when "
            f"computing integrity_hash; saw: {sorted(overlap)}"
        )
    return hashlib.sha256(canonical_json(record_with_prev_hash)).hexdigest()


def verify_entry_hash(entry: dict[str, Any],
                      expected_prev_hash: str) -> dict[str, Any]:
    """Verify a single entry against an expected previous hash.

    Replicates AuditLogger.verify's per-record check (module5_pipeline)
    but returns a diagnostic dict instead of mutating an outer result.
    Does NOT verify ECDSA signatures (separate concern; left to the
    writer module's classmethod).

    Returns dict with:
      is_valid: bool
      prev_hash_match: bool
      integrity_hash_match: bool
      stored_prev_hash, stored_integrity_hash, computed_integrity_hash
      failure_reason: str | None
    """
    stored_prev = entry.get("prev_hash")
    prev_match = stored_prev == expected_prev_hash
    if not prev_match:
        return {
            "is_valid": False,
            "prev_hash_match": False,
            "integrity_hash_match": False,
            "stored_prev_hash": stored_prev,
            "expected_prev_hash": expected_prev_hash,
            "stored_integrity_hash": entry.get(INTEGRITY_FIELD),
            "computed_integrity_hash": None,
            "failure_reason": (
                f"prev_hash mismatch: stored={str(stored_prev)[:12]}..., "
                f"expected={expected_prev_hash[:12]}..."
            ),
        }

    # Strip signature envelope + integrity_hash; recompute over remainder.
    body = {k: v for k, v in entry.items()
            if k not in SIGNATURE_ENVELOPE_FIELDS and k != INTEGRITY_FIELD}
    computed = hashlib.sha256(canonical_json(body)).hexdigest()
    stored_integrity = entry.get(INTEGRITY_FIELD)
    integrity_match = computed == stored_integrity

    return {
        "is_valid": prev_match and integrity_match,
        "prev_hash_match": prev_match,
        "integrity_hash_match": integrity_match,
        "stored_prev_hash": stored_prev,
        "expected_prev_hash": expected_prev_hash,
        "stored_integrity_hash": stored_integrity,
        "computed_integrity_hash": computed,
        "failure_reason": (
            None if (prev_match and integrity_match)
            else (
                f"integrity_hash mismatch: stored={str(stored_integrity)[:12]}..., "
                f"computed={computed[:12]}..."
            )
        ),
    }
