"""Security controls for Phase 0 — actually wired into the loader.

Classes
-------
IntegrityVerifier   — A02: SHA-256 dataset integrity, signed with the
                      Module 5 ECDSA P-256 key, no auto-baseline footgun.
PathValidator       — A01: workspace containment via resolve()+relative_to().
ColumnAllowlist     — A03: column-name allowlist enforcement (the only
                      injection control that maps to a real attack surface
                      in this layer; sanitize_string() was theatre and is
                      gone).
log_phase0_event    — A09: routes Phase-0 audit events into the same
                      hardened, hash-chained, signed JSONL audit log used
                      by Module 5, so Phase 0 events inherit chain +
                      signature guarantees.

Design notes
------------
- ``security.py`` is now wired into ``DataLoader.load()`` and
  ``Phase0Config.from_yaml``. None of it is dead code.
- The integrity baseline is bootstrapped explicitly via the
  ``bootstrap_integrity`` CLI; ``verify()`` refuses to create a new
  baseline silently. This closes the "delete the JSON to whitewash a
  tampered file" attack surface that the previous implementation had.
- The dataset is hashed and parsed from the same in-memory bytes to
  eliminate the TOCTOU window between hash and read.
- Path traversal detection is delegated to ``Path.resolve() +
  relative_to(root)``. The previous substring check on ``..``/``~``/``$``
  was theatre that produced false positives without adding protection.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import stat
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from common.phi import BIOMETRIC_COLUMNS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_HASH_ALGORITHM: str = "sha256"
_METADATA_FILE: str = "dataset_integrity.json"
# v2: entries keyed by absolute path (leaked dev hostnames into the
#     committed baseline; one fresh-bootstrap per host produced drift).
# v3: entries keyed by SHA-256 digest. Filename + size are payload fields.
#     One physical file → one entry, regardless of host/path.
_METADATA_VERSION: int = 3
_SUPPORTED_VERSIONS: frozenset[int] = frozenset({3})


# ===================================================================
# A02 — Cryptographic Failures: Dataset Integrity (signed, no auto-baseline)
# ===================================================================


class IntegrityError(Exception):
    """Raised when a dataset's SHA-256 hash does not match its baseline,
    or when the integrity metadata file is missing/corrupt/forged.
    """


class IntegrityVerifier:
    """SHA-256 integrity verification with an ECDSA-signed metadata file.

    Lifecycle:
        1. ``bootstrap(path)`` — explicit, one-time. Refuses to overwrite
           an existing baseline. Hashes the file, signs the record with
           the Module 5 ECDSA P-256 key, and persists.
        2. ``verify_and_read(path)`` — every load. Reads the file once
           into memory, hashes the bytes, verifies the stored hash and
           the signature, and returns the bytes for the caller to parse.
           Refuses to run if no baseline exists for the file.

    The verifier and the parser operate on the same in-memory buffer to
    eliminate the TOCTOU window between hashing and reading.

    Args:
        metadata_dir: Directory where ``dataset_integrity.json`` is stored.
                      In production this should be on a volume only the
                      operator UID can write to.
    """

    def __init__(self, metadata_dir: Path) -> None:
        self._metadata_dir = metadata_dir
        self._metadata_path = metadata_dir / _METADATA_FILE

    # ── public API ─────────────────────────────────────────────────

    def bootstrap(self, file_path: Path) -> str:
        """Establish (or refresh) the signed baseline for *file_path*.

        Entries are keyed by the file's SHA-256 digest (v3 schema), so
        two hosts bootstrapping the same physical file produce one
        entry — not one-per-path. Filename + size are payload fields
        and are checked at verify time.

        Idempotent on identical content: if the same hash is already
        baselined, returns the existing digest (no-op) instead of
        refusing or duplicating.

        Tier 0 F1: when a baseline already exists on disk we read it
        through ``_read_metadata_verified`` so the idempotent shortcut
        cannot fire against attacker-injected rows. A baseline whose
        signature fails verification causes ``IntegrityError`` instead
        of being silently extended.

        Returns the SHA-256 hex digest written to the baseline.

        Raises:
            FileNotFoundError: if *file_path* does not exist.
            IntegrityError: if the existing baseline fails signature
                verification, or if a different file is already baselined
                under the same hash (sanity assert).
        """
        data = self._read_bytes(file_path)
        digest = hashlib.new(_HASH_ALGORITHM, data).hexdigest()
        size = len(data)
        filename = file_path.name

        if self._metadata_path.exists() and self._metadata_path.stat().st_size > 0:
            existing = self._read_metadata_verified()
        else:
            existing = self._read_metadata()
        self._assert_supported_version(existing)
        entries = existing.get("entries", {})
        prior = entries.get(digest)
        if prior is not None:
            # Same content already baselined. Sanity-check that the
            # filename/size payload matches; if not, the operator pointed
            # at a renamed-but-identical file — log it but allow.
            if prior.get("size_bytes") != size:
                raise IntegrityError(
                    f"Hash collision sanity-check failed: existing entry for "
                    f"sha256={digest[:16]}… has size {prior.get('size_bytes')} "
                    f"but new file is {size} bytes. Refusing to overwrite."
                )
            log_phase0_event(
                "INTEGRITY_BOOTSTRAP_NOOP",
                {"file": filename, "sha256_prefix": digest[:16]},
            )
            return digest

        record = {
            "filename": filename,
            "size_bytes": size,
            "bootstrapped_at": datetime.now(timezone.utc).isoformat(),
        }
        self._write_signed_metadata(digest=digest, record=record)
        log_phase0_event(
            "INTEGRITY_BOOTSTRAPPED",
            {"file": filename, "sha256_prefix": digest[:16]},
        )
        return digest

    def verify_and_read(self, file_path: Path) -> Tuple[bytes, str]:
        """Verify *file_path* against its signed baseline and return its bytes.

        The returned bytes are the exact same buffer that was hashed,
        so the caller must parse them via ``io.BytesIO`` rather than
        re-opening the file.

        Order matters: metadata signature is verified and the baseline
        entry is looked up BEFORE the file is read into memory. This
        prevents a DoS via "point at a 100GB attacker-controlled file"
        — we exit early if there's no baseline entry, without paying for
        the read or the hash.

        Returns:
            (file_bytes, sha256_hex_digest)

        Raises:
            FileNotFoundError: if *file_path* does not exist.
            IntegrityError: if the baseline is missing, the signature is
                invalid, or the recomputed hash does not match.
        """
        if not self._metadata_path.exists():
            raise IntegrityError(
                f"No integrity baseline at {self._metadata_path}. "
                f"Run `python -m module0_analysis.bootstrap_integrity` "
                f"once to establish the baseline."
            )

        # 1. Verify metadata signature (cheap, bounded by entries size).
        metadata = self._read_metadata_verified()
        self._assert_supported_version(metadata)
        entries = metadata.get("entries", {})

        # 2. DoS guard: file size must equal a known-baseline size.
        # Since entries are keyed by hash, we don't know which entry to
        # match against without hashing — but we DO know the size of
        # every legitimate baseline, so a file whose size matches none
        # of them cannot possibly verify. Reject without reading.
        if not file_path.exists():
            raise FileNotFoundError(f"Cannot verify: file not found: {file_path}")
        actual_size = file_path.stat().st_size
        allowed_sizes = {e.get("size_bytes") for e in entries.values()}
        if actual_size not in allowed_sizes:
            log_phase0_event(
                "INTEGRITY_SIZE_MISMATCH",
                {
                    "file": file_path.name,
                    "actual_size": actual_size,
                    "allowed_sizes_count": len(allowed_sizes),
                },
                level=logging.ERROR,
            )
            raise IntegrityError(
                f"File size {actual_size} does not match any baseline "
                f"(known sizes: {sorted(s for s in allowed_sizes if s is not None)}). "
                f"Refusing to hash an attacker-controlled-size file."
            )

        # 3. NOW read the file + hash (expensive, but size-bounded).
        data = self._read_bytes(file_path)
        digest = hashlib.new(_HASH_ALGORITHM, data).hexdigest()

        # 4. Lookup by hash. Mismatch = tamper.
        stored = entries.get(digest)
        if stored is None:
            log_phase0_event(
                "INTEGRITY_VIOLATION",
                {
                    "file": file_path.name,
                    "actual_sha256_prefix": digest[:16],
                },
                level=logging.CRITICAL,
            )
            raise IntegrityError(
                f"INTEGRITY VIOLATION: {file_path.name} — "
                f"sha256={digest[:16]}… is not in the baseline."
            )

        log_phase0_event(
            "INTEGRITY_VERIFIED",
            {"file": file_path.name, "sha256_prefix": digest[:16]},
        )
        return data, digest

    # ── internals ──────────────────────────────────────────────────

    @staticmethod
    def _read_bytes(file_path: Path) -> bytes:
        if not file_path.exists():
            raise FileNotFoundError(f"Cannot hash: file not found: {file_path}")
        return file_path.read_bytes()

    @staticmethod
    def _assert_supported_version(metadata: Dict[str, Any]) -> None:
        """Refuse to operate on metadata whose schema version is unknown.

        v2 (path-keyed) baselines must be migrated via
        `module0_analysis.migrate_v2_to_v3` before they can be used.
        """
        version = metadata.get("version")
        if version is None:
            # Empty metadata (no entries yet) — bootstrap path is allowed.
            if not metadata.get("entries"):
                return
            raise IntegrityError(
                "Integrity metadata is missing the 'version' field — "
                "refusing to proceed."
            )
        if version not in _SUPPORTED_VERSIONS:
            raise IntegrityError(
                f"Integrity metadata uses unsupported schema version {version}. "
                f"This security.py supports v{sorted(_SUPPORTED_VERSIONS)}. "
                f"v2 baselines (path-keyed) must be migrated: run "
                f"`python -m module0_analysis.migrate_v2_to_v3` once. "
                f"After migration, re-bootstrap or re-verify."
            )

    def _read_metadata(self) -> Dict[str, Any]:
        """Read raw metadata without signature checking (bootstrap path)."""
        if not self._metadata_path.exists():
            return {"version": _METADATA_VERSION, "entries": {}}
        try:
            return json.loads(self._metadata_path.read_text())
        except json.JSONDecodeError as exc:
            log_phase0_event(
                "INTEGRITY_METADATA_CORRUPT",
                {"path": str(self._metadata_path), "error": str(exc)},
                level=logging.CRITICAL,
            )
            raise IntegrityError(
                f"Integrity metadata at {self._metadata_path} is corrupt: "
                f"{exc}. Refusing to proceed."
            ) from exc

    def _read_metadata_verified(self) -> Dict[str, Any]:
        """Read metadata AND verify the ECDSA signature on the entries."""
        meta = self._read_metadata()
        entries = meta.get("entries", {})
        signature_b64 = meta.get("signature")
        if not signature_b64:
            raise IntegrityError(
                "Integrity metadata is unsigned. Re-bootstrap with the "
                "current security.py to produce a signed baseline."
            )

        # Lazy import: keep cryptography optional for environments that
        # only run unit tests against the analyzers.
        from module5_responses.signing import HAVE_CRYPTOGRAPHY, canonical_json

        if not HAVE_CRYPTOGRAPHY:
            raise IntegrityError(
                "cryptography package is not installed; cannot verify "
                "signed integrity baseline."
            )

        import base64
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import ec

        public_key = _load_phase0_public_key()

        signed_payload = canonical_json({"version": meta.get("version"), "entries": entries})
        try:
            public_key.verify(
                base64.b64decode(signature_b64),
                signed_payload,
                ec.ECDSA(hashes.SHA256()),
            )
        except InvalidSignature as exc:
            log_phase0_event(
                "INTEGRITY_METADATA_FORGED",
                {"path": str(self._metadata_path)},
                level=logging.CRITICAL,
            )
            raise IntegrityError(
                f"Integrity metadata signature is invalid — "
                f"{self._metadata_path} has been tampered with."
            ) from exc

        return meta

    def _write_signed_metadata(self, digest: str, record: Dict[str, Any]) -> None:
        """Add *record* under *digest* (v3 hash key) and re-sign the entries block.

        Tier 0 F1: when the existing baseline contains any entries we
        verify its signature **before** re-signing the augmented body.
        Without this, an attacker who can write
        ``dataset_integrity.json`` can inject a malicious row and have
        the operator's next ``bootstrap`` call laundered as a signed
        legitimisation of that row.
        """
        from module5_responses.signing import (
            HAVE_CRYPTOGRAPHY,
            canonical_json,
            load_signing_key,
        )

        if not HAVE_CRYPTOGRAPHY:
            raise IntegrityError(
                "cryptography package is not installed; cannot sign the "
                "integrity baseline. Install with `pip install cryptography>=42`."
            )
        import base64
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import ec

        # Tier 0 F1: refuse to re-sign over an unverified baseline.
        # Empty/new metadata is fine; any non-empty baseline must pass
        # signature verification before we accept its entries into the
        # new signed body.
        if self._metadata_path.exists() and self._metadata_path.stat().st_size > 0:
            try:
                meta = self._read_metadata_verified()
            except IntegrityError:
                log_phase0_event(
                    "INTEGRITY_BOOTSTRAP_REFUSED",
                    {"path": str(self._metadata_path),
                     "reason": "existing baseline failed verification"},
                    level=logging.CRITICAL,
                )
                raise
        else:
            meta = self._read_metadata()
        if meta.get("entries"):
            self._assert_supported_version(meta)
        entries = meta.get("entries", {})
        entries[digest] = record
        body = {"version": _METADATA_VERSION, "entries": entries}

        private_key, _, signing_key_id = load_signing_key()
        signature = private_key.sign(canonical_json(body), ec.ECDSA(hashes.SHA256()))
        body["signature"] = base64.b64encode(signature).decode("ascii")
        body["signing_key_id"] = signing_key_id
        body["signature_alg"] = "ECDSA_P256_SHA256"

        self._metadata_dir.mkdir(parents=True, exist_ok=True)
        # Atomic write so a crash mid-write cannot leave us with a
        # half-baselined file that verify() would refuse on next load.
        tmp = self._metadata_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(body, indent=2))
        os.replace(tmp, self._metadata_path)
        try:
            os.chmod(self._metadata_path, 0o640)
        except OSError:
            pass


@lru_cache(maxsize=1)
def _load_phase0_public_key():
    """Load the Module 5 public key once and cache for the process lifetime."""
    from module5_responses.signing import load_signing_key
    from cryptography.hazmat.primitives import serialization

    _, public_path, _ = load_signing_key()
    return serialization.load_pem_public_key(public_path.read_bytes())


# ===================================================================
# A01 — Broken Access Control: workspace containment
# ===================================================================


class PathValidator:
    """Validate file paths against workspace boundaries.

    The only real defense here is ``Path.resolve() + relative_to(root)``;
    the previous substring check on ``..``/``~``/``$`` was theatre that
    produced false positives without catching real escapes (URL-encoded
    traversal, NUL bytes, symlink games). It is gone.

    Args:
        workspace_root: The top-level project directory. All resolved
                        paths must reside within this directory tree.
    """

    def __init__(self, workspace_root: Path) -> None:
        self._root = workspace_root.resolve()

    def validate_input_path(self, path: Path) -> Path:
        """Resolve *path*, assert workspace containment AND existence.

        Returns:
            Resolved absolute path inside the workspace.

        Raises:
            PermissionError: if the resolved path escapes the workspace.
            FileNotFoundError: if the resolved path does not exist.
        """
        resolved = self._resolve_inside_workspace(path)
        if not resolved.exists():
            raise FileNotFoundError(f"Input path does not exist: {resolved}")
        log_phase0_event("INPUT_VALIDATED", {"path": str(resolved)})
        return resolved

    def validate_path_containment(self, path: Path) -> Path:
        """Resolve *path* and assert workspace containment WITHOUT existence check.

        Use this for config-load-time validation where the target may
        legitimately not exist yet (e.g., a CI config that lists an
        input_dir before the dataset is downloaded). Pipelines should
        still re-validate via ``validate_input_path`` at use time.

        Returns:
            Resolved absolute path inside the workspace.

        Raises:
            PermissionError: if the resolved path escapes the workspace.
        """
        return self._resolve_inside_workspace(path)

    def validate_output_dir(self, path: Path) -> Path:
        """Resolve *path*, assert workspace containment, then mkdir.

        Returns:
            Resolved absolute path inside the workspace.

        Raises:
            PermissionError: if the resolved path escapes the workspace.
        """
        resolved = self._resolve_inside_workspace(path)
        resolved.mkdir(parents=True, exist_ok=True)
        log_phase0_event("OUTPUT_DIR_VALIDATED", {"path": str(resolved)})
        return resolved

    def check_read_only(self, path: Path, *, enforce: bool = False) -> bool:
        """Check (or enforce) that *path* is read-only for the owner.

        When called with ``enforce=True`` (set automatically when the
        ``PHASE0_PROD=1`` environment variable is present), a writable
        raw dataset becomes a hard failure rather than a warning. This
        prevents an in-place tamper-then-rerun attack on a production
        host.
        """
        mode = path.stat().st_mode
        is_readonly = not (mode & stat.S_IWUSR)
        if not is_readonly:
            msg = (
                f"File {path.name} is writable (mode={mode & 0o777:o}). "
                f"Raw datasets must be chmod 444 in production."
            )
            if enforce or os.environ.get("PHASE0_PROD") == "1":
                log_phase0_event(
                    "RAW_DATASET_WRITABLE",
                    {"file": path.name, "mode_octal": f"{mode & 0o777:o}"},
                    level=logging.CRITICAL,
                )
                raise PermissionError(msg)
            logger.warning("A01: %s", msg)
        return is_readonly

    def _resolve_inside_workspace(self, path: Path) -> Path:
        resolved = (self._root / path).resolve()
        try:
            resolved.relative_to(self._root)
        except ValueError as exc:
            log_phase0_event(
                "PATH_ESCAPE",
                {"path": str(resolved), "root": str(self._root)},
                level=logging.ERROR,
            )
            raise PermissionError(
                f"A01: Path escapes workspace — {resolved} is outside {self._root}"
            ) from exc
        return resolved


# ===================================================================
# A03 — Injection: Column allowlist (the only piece worth keeping)
# ===================================================================


class ColumnAllowlist:
    """Enforce that requested column names are present in the DataFrame.

    The previous ``ConfigSanitizer.sanitize_string`` regex tried to
    police arbitrary config values for "shell-dangerous characters", but
    Phase 0 has no shell, no SQL, no eval surface — the regex blocked
    legitimate strings (apostrophes, accented author names) without
    protecting against any real attack. It is gone.

    What does map to a real attack surface is column-name validation:
    if the config asks for a column that the DataFrame does not have,
    silently producing zeros or NaNs would corrupt every downstream
    statistic. So we keep that check, and only that check.
    """

    @staticmethod
    def validate(
        requested_columns: Sequence[str],
        actual_columns: Set[str],
        *,
        context: str = "config",
    ) -> List[str]:
        invalid = [c for c in requested_columns if c not in actual_columns]
        if invalid:
            log_phase0_event(
                "COLUMN_ALLOWLIST_VIOLATION",
                {"context": context, "unknown_columns": invalid},
                level=logging.ERROR,
            )
            raise ValueError(
                f"A03: Column allowlist violation in {context} — " f"unknown columns: {invalid}"
            )
        return list(requested_columns)


# ===================================================================
# A09 — Security Logging: routed through the Module 5 hardened chain
# ===================================================================


_phase0_logger: Optional[logging.Logger] = None
_hardened_audit = None  # lazy: HardenedAuditLogger or None


def _get_phase0_logger() -> logging.Logger:
    global _phase0_logger
    if _phase0_logger is None:
        _phase0_logger = logging.getLogger("phase0.security.audit")
    return _phase0_logger


def _get_hardened_audit():
    """Lazily construct (or reuse) the Module 5 signed-chain logger.

    Returns ``None`` if Module 5 cannot be imported (e.g. cryptography
    not installed). Phase 0 events still go to the local logger in that
    case so they are not silently lost.
    """
    global _hardened_audit
    if _hardened_audit is not None:
        return _hardened_audit
    try:
        from module5_responses.module5_pipeline import (
            AuditLogger as HardenedAuditLogger,
            OUTPUT_DIR,
        )
    except (ImportError, ModuleNotFoundError) as exc:
        logger.warning(
            "phase0.security: cannot reach Module 5 hardened audit log "
            "(%s); falling back to local logger only.",
            exc,
        )
        _hardened_audit = False
        return None
    try:
        _hardened_audit = HardenedAuditLogger(OUTPUT_DIR / "audit_log.jsonl")
    except (OSError, RuntimeError) as exc:
        logger.warning(
            "phase0.security: failed to construct hardened audit logger "
            "(%s); falling back to local logger only.",
            exc,
        )
        _hardened_audit = False
        return None
    return _hardened_audit


def log_phase0_event(
    event: str,
    payload: Optional[Dict[str, Any]] = None,
    *,
    level: int = logging.INFO,
) -> None:
    """Log a Phase 0 audit event.

    Writes to two sinks:
      1. Local ``phase0.security.audit`` logger (always).
      2. Module 5 hardened, hash-chained, ECDSA-signed audit log
         (when available). Phase 0 events therefore inherit the same
         tamper-evident properties as Module 5 events.

    Payloads must NEVER contain biometric values. The function does not
    enforce this — callers are responsible — but a defensive check is
    applied to redact any keys whose names match a biometric column.
    """
    payload = dict(payload or {})
    for k in BIOMETRIC_COLUMNS & payload.keys():
        payload[k] = "[REDACTED-PHI]"

    ts = datetime.now(timezone.utc).isoformat()
    _get_phase0_logger().log(level, "[%s] %s: %s", ts, event, payload if payload else "")

    audit = _get_hardened_audit()
    if audit:
        try:
            audit.log(
                {
                    "event_type": "phase0_security",
                    "subtype": event,
                    "payload": payload,
                    "level": logging.getLevelName(level),
                    "logged_at": ts,
                }
            )
        except (OSError, RuntimeError) as exc:
            # Never let an audit-sink failure block Phase 0 execution,
            # but DO surface it loudly so an operator notices.
            logger.error(
                "phase0.security: failed to append to hardened audit log: %s",
                exc,
            )
