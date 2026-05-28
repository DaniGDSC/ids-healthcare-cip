"""Audit-log verification — hash chain + ECDSA signature walk."""
from __future__ import annotations

import base64
import hashlib
import json
import logging
from pathlib import Path

from .signing import (
    DEFAULT_PUBLIC_KEY_PATH,
    _canonical_json,
    _HAVE_CRYPTOGRAPHY,
)

try:
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
except ImportError:  # pragma: no cover
    pass

logger = logging.getLogger(__name__)


def _mark_break(result: dict, line_no: int, reason: str) -> None:
    result["broken"].append({"line": line_no, "reason": reason})
    if result["first_break_at"] is None:
        result["first_break_at"] = line_no


def verify_audit_log(
    path: Path,
    public_key_path: Path | None = None,
    *,
    legacy_ok: bool = False,
) -> dict:
    """Walk an audit log and verify hash chain + signatures.

    Args:
        path: path to the audit log JSONL file.
        public_key_path: PEM file containing the ECDSA P-256 public key.
            Defaults to :data:`DEFAULT_PUBLIC_KEY_PATH`.
        legacy_ok: when False (post-Sprint-3 default — tier 1 F1) any
            unsigned record is reported as broken. When True, unsigned
            records and unsigned ``prev_hash=0`` chain restarts are
            accepted as legacy migration markers. The migration default
            flipped to False after Sprint 3 sealed the historical log
            into an archive — set ``legacy_ok=True`` only when walking
            those archives.

    Returns:
        Dict with totals, the line number of the first break (if any),
        and a list of per-broken-line reasons.
    """
    path = Path(path)
    public_key_path = Path(public_key_path or DEFAULT_PUBLIC_KEY_PATH)

    result: dict = {
        "path": str(path),
        "public_key": str(public_key_path),
        "total": 0,
        "valid_signed": 0,
        "valid_legacy": 0,
        "broken": [],
        "first_break_at": None,
    }

    if not path.exists():
        result["broken"].append({"line": 0, "reason": "file does not exist"})
        return result

    public_key = None
    if _HAVE_CRYPTOGRAPHY and public_key_path.exists():
        try:
            public_key = serialization.load_pem_public_key(
                public_key_path.read_bytes()
            )
        except Exception as exc:  # noqa: BLE001
            result["broken"].append(
                {"line": 0, "reason": f"failed to load public key: {exc}"}
            )
            return result

    prev_hash_expected = "0" * 64
    result["legacy_chain_restarts"] = 0
    with open(path, "r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            result["total"] += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                _mark_break(result, line_no, f"json parse: {exc}")
                return result

            is_unsigned = "signature" not in record

            if record.get("prev_hash") != prev_hash_expected:
                # Legacy migration: the pre-hardening AuditLogger
                # reset the chain to genesis on every process start.
                # In legacy mode, accept a fresh genesis block as a
                # known-good restart marker rather than tampering.
                if (
                    legacy_ok
                    and is_unsigned
                    and record.get("prev_hash") == "0" * 64
                    and line_no > 1
                ):
                    result["legacy_chain_restarts"] += 1
                    prev_hash_expected = "0" * 64
                else:
                    _mark_break(
                        result,
                        line_no,
                        f"prev_hash mismatch (expected "
                        f"{prev_hash_expected[:12]}..., got "
                        f"{str(record.get('prev_hash'))[:12]}...)",
                    )
                    return result

            signature_b64 = record.pop("signature", None)
            record.pop("signing_key_id", None)
            record.pop("signature_alg", None)
            stored_integrity = record.get("integrity_hash")
            without_hash = {
                k: v for k, v in record.items() if k != "integrity_hash"
            }
            computed = hashlib.sha256(_canonical_json(without_hash)).hexdigest()
            if computed != stored_integrity and signature_b64 is None:
                legacy_payload = json.dumps(without_hash, sort_keys=True).encode(
                    "utf-8"
                )
                legacy_hash = hashlib.sha256(legacy_payload).hexdigest()
                if legacy_hash == stored_integrity:
                    computed = stored_integrity
            if computed != stored_integrity:
                _mark_break(
                    result,
                    line_no,
                    "integrity_hash mismatch (record body tampered)",
                )
                return result

            if signature_b64 is None:
                if legacy_ok:
                    result["valid_legacy"] += 1
                    prev_hash_expected = stored_integrity
                    continue
                _mark_break(result, line_no, "record is unsigned")
                return result

            if public_key is None:
                _mark_break(
                    result,
                    line_no,
                    "signature present but no public key available",
                )
                return result

            try:
                sig_record = dict(record)
                sig_payload = _canonical_json(sig_record)
                public_key.verify(
                    base64.b64decode(signature_b64),
                    sig_payload,
                    ec.ECDSA(hashes.SHA256()),
                )
                result["valid_signed"] += 1
            except InvalidSignature:
                _mark_break(result, line_no, "invalid signature")
                return result
            except Exception as exc:  # noqa: BLE001
                _mark_break(result, line_no, f"signature verify error: {exc}")
                return result

            prev_hash_expected = stored_integrity

    return result


__all__ = ["verify_audit_log"]
