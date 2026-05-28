"""AuditLogger — hash-chained, ECDSA-signed append-only JSONL audit log.

The class is intentionally kept as a single public type to preserve the
9-consumer API. Internally it delegates retention/rotation to ``retention``
and verification to ``verify``.
"""
from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from .signing import (
    DEFAULT_PUBLIC_KEY_PATH,
    OUTPUT_DIR,
    SIGNATURE_ALG,
    _canonical_json,
    _HAVE_CRYPTOGRAPHY,
    _load_signing_key,
)

try:
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import ec
except ImportError:  # pragma: no cover
    pass

logger = logging.getLogger(__name__)

ARCHIVE_DIR = OUTPUT_DIR / "audit_archive"

# Default retention: 365 days. Override per deployment via the
# IOMT_AUDIT_RETENTION_DAYS environment variable or the constructor
# argument ``retention_days``. See module docstring for jurisdiction notes.
DEFAULT_RETENTION_DAYS = 365


class AuditLogger:
    """Hash-chained, ECDSA-signed append-only JSONL audit log.

    Each record carries:
      - ``prev_hash``        : sha256 of the previous record (hash chain)
      - ``integrity_hash``   : sha256 of the current record (covers prev_hash)
      - ``signature``        : ECDSA P-256 signature over the canonical JSON
                               of the record (covers integrity_hash and
                               everything below it)
      - ``signing_key_id``   : stable id derived from the public key
      - ``signature_alg``    : ``"ECDSA_P256_SHA256"``

    Optional reviewer attribution: when callers pass ``reviewer_id`` /
    ``reviewer_role`` to :meth:`log`, a ``reviewer`` block is added to the
    record *before* signing, so reviewer attribution is bound to the
    signature.

    Restart safety: if the target file already exists, the constructor
    walks the last record and continues the chain from its
    ``integrity_hash``, so multiple invocations of the same pipeline do
    not produce a fake chain break.

    Retention: :meth:`rotate_and_purge` archives the active log into a
    sealed file under ``audit_archive/``, then starts a new active log
    whose first ``prev_hash`` points back at the last archived record so
    the cross-rotation chain remains walkable for forensics.
    """

    def __init__(
        self,
        path: Path,
        *,
        signing_key_path: Path | None = None,
        public_key_path: Path | None = None,
        retention_days: int | None = None,
        sign: bool = True,
        verify_on_open: bool = True,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        self.sign_enabled = sign and _HAVE_CRYPTOGRAPHY
        if sign and not _HAVE_CRYPTOGRAPHY:
            logger.warning(
                "AuditLogger: cryptography not installed; signing disabled. "
                "Records will be hash-chained only."
            )

        if self.sign_enabled:
            self._private_key, self.public_key_path, self.signing_key_id = (
                _load_signing_key(signing_key_path, public_key_path)
            )
        else:
            self._private_key = None
            self.public_key_path = public_key_path or DEFAULT_PUBLIC_KEY_PATH
            self.signing_key_id = "unsigned"

        env_days = os.environ.get("IOMT_AUDIT_RETENTION_DAYS")
        if retention_days is not None:
            self.retention_days = int(retention_days)
        elif env_days:
            self.retention_days = int(env_days)
        else:
            self.retention_days = DEFAULT_RETENTION_DAYS

        # Tier 1 F2: walk the existing log under the post-Sprint-3
        # default (legacy_ok=False) before we open it for append.
        # A broken chain raises before any new record can extend it.
        # `verify_on_open=False` is reserved for the rotation CLI which
        # owns the recovery dance (rotate_key writes to a fresh file).
        if verify_on_open and self.path.exists() and self.path.stat().st_size > 0:
            from .verify import verify_audit_log

            report = verify_audit_log(self.path, self.public_key_path, legacy_ok=False)
            if report.get("first_break_at") is not None:
                raise RuntimeError(
                    f"AuditLogger refusing to append to a tampered chain: "
                    f"first break at line {report['first_break_at']} of "
                    f"{self.path}. Run `python -m module5_responses.audit."
                    f"rotate_key --i-understand-this-orphans-old-signatures` "
                    f"to seal the prior chain and start a new one."
                )

        self.prev_hash = self._recover_prev_hash()

    # ── chain recovery ─────────────────────────────────────────────

    def _recover_prev_hash(self) -> str:
        """Read the last record's integrity_hash to continue the chain.

        M5-6: reads only the last 4 KB of the file instead of streaming
        the entire JSONL from the beginning — O(1) disk I/O regardless of
        log size.
        """
        if not self.path.exists() or self.path.stat().st_size == 0:
            return "0" * 64
        try:
            with open(self.path, "rb") as f:
                f.seek(0, 2)
                size = f.tell()
                f.seek(max(0, size - 4096))
                tail = f.read()
        except OSError:
            return "0" * 64

        lines = [ln for ln in tail.split(b"\n") if ln.strip()]
        if not lines:
            return "0" * 64
        last_line = lines[-1].decode("utf-8", errors="ignore").strip()
        try:
            last_record = json.loads(last_line)
            recovered = last_record.get("integrity_hash")
            if isinstance(recovered, str) and len(recovered) == 64:
                return recovered
        except json.JSONDecodeError:
            logger.warning(
                "AuditLogger: tail of %s is unparseable; starting new "
                "chain at genesis.",
                self.path,
            )
        return "0" * 64

    # ── log ────────────────────────────────────────────────────────

    def log(
        self,
        record: dict,
        *,
        reviewer_id: str | None = None,
        reviewer_role: str | None = None,
        review_timestamp: str | None = None,
        review_action: str | None = None,
    ) -> dict:
        """Append a hash-chained, signed record to the audit log.

        Args:
            record: arbitrary JSON-serializable event payload.
            reviewer_id: optional human reviewer identifier (e.g. P03).
            reviewer_role: optional role (Security Analyst / Clinician
                / Administrator).
            review_timestamp: ISO-8601 timestamp; defaults to now() in
                UTC if any other reviewer field is provided.
            review_action: optional free-text action label
                (confirm / reject / acknowledge / ...).

        Returns:
            The record as it was written (with all envelope fields).
        """
        record = dict(record)

        if any(x is not None for x in (reviewer_id, reviewer_role, review_action)):
            if review_timestamp is None:
                review_timestamp = datetime.now(timezone.utc).isoformat()
            record["reviewer"] = {
                "reviewer_id": reviewer_id,
                "reviewer_role": reviewer_role,
                "review_timestamp": review_timestamp,
                "review_action": review_action,
            }

        record["prev_hash"] = self.prev_hash
        record["integrity_hash"] = hashlib.sha256(_canonical_json(record)).hexdigest()

        if self.sign_enabled:
            sig_payload = _canonical_json(record)
            signature_der = self._private_key.sign(
                sig_payload, ec.ECDSA(hashes.SHA256())
            )
            record["signature"] = base64.b64encode(signature_der).decode("ascii")
            record["signing_key_id"] = self.signing_key_id
            record["signature_alg"] = SIGNATURE_ALG

        self.prev_hash = record["integrity_hash"]
        is_new_file = not self.path.exists() or self.path.stat().st_size == 0
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        # Tier 1 F3 / tier 2 F7: chmod 0640 on first write so the audit
        # log is not world-readable by default umask. Skip on subsequent
        # writes where the mode is already set.
        if is_new_file:
            try:
                os.chmod(self.path, 0o640)
            except OSError as exc:
                logger.warning(
                    "AuditLogger: chmod 0640 on %s failed: %s. Tighten "
                    "permissions manually so the chain is not group/world "
                    "readable.",
                    self.path, exc,
                )
        return record

    # ── verification (delegates) ───────────────────────────────────

    @classmethod
    def verify(
        cls,
        path: Path,
        public_key_path: Path | None = None,
        *,
        legacy_ok: bool = False,
    ) -> dict:
        """Walk an audit log and verify hash chain + signatures.

        Tier 1 F1: post-Sprint-3 default is ``legacy_ok=False``. Callers
        walking archived pre-migration logs (under ``audit_archive/``)
        should opt in by passing ``legacy_ok=True`` explicitly.
        """
        from .verify import verify_audit_log
        return verify_audit_log(path, public_key_path, legacy_ok=legacy_ok)

    @staticmethod
    def _mark_break(result: dict, line_no: int, reason: str) -> None:
        """Append a break entry; first break wins ``first_break_at``."""
        result["broken"].append({"line": line_no, "reason": reason})
        if result["first_break_at"] is None:
            result["first_break_at"] = line_no

    # ── rotation (delegates) ───────────────────────────────────────

    def rotate_and_purge(
        self,
        retention_days: int | None = None,
        archive_dir: Path | None = None,
    ) -> dict:
        """Archive the current active log if older than the retention cutoff.

        Returns a dict describing what happened. Delegates to
        :func:`audit.retention.rotate_and_purge`.
        """
        from .retention import rotate_and_purge as _do_rotate
        return _do_rotate(self, retention_days=retention_days, archive_dir=archive_dir)


__all__ = [
    "AuditLogger",
    "ARCHIVE_DIR",
    "OUTPUT_DIR",
    "DEFAULT_RETENTION_DAYS",
]
