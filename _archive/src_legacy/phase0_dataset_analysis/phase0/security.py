"""Security controls for Phase 0 — OWASP Top 10 + HIPAA compliance.

Classes
-------
IntegrityVerifier   — A02: SHA-256 dataset integrity hashing and verification
PathValidator       — A01: path traversal protection, workspace containment
ConfigSanitizer     — A03: input sanitization, column allowlist enforcement
AuditLogger         — A09: HIPAA-compliant security event logging

Design Principles
-----------------
- Single Responsibility: each class addresses exactly one OWASP risk.
- No ``eval()`` or ``exec()`` anywhere in this module or its callers.
- Biometric *values* are never logged — only column *names*.
- All file access events are logged with ISO-8601 timestamps.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import stat
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_HASH_ALGORITHM: str = "sha256"
_HASH_CHUNK_SIZE: int = 65_536  # 64 KiB read chunks for streaming hash
_METADATA_FILE: str = "dataset_integrity.json"

# Characters allowed in config string values (printable ASCII minus shell-dangerous)
_SAFE_STRING_RE = re.compile(r"^[a-zA-Z0-9 _\-./,;:=+\[\](){}@#%&*!?]+$")

# Patterns that signal path traversal
_TRAVERSAL_PATTERNS = ("..", "~", "$")

# Biometric columns whose *values* must never appear in logs
BIOMETRIC_COLUMNS: frozenset = frozenset({
    "Temp", "SpO2", "Pulse_Rate", "SYS", "DIA",
    "Heart_rate", "Resp_Rate", "ST",
})


# ===================================================================
# A02 — Cryptographic Failures: Dataset Integrity
# ===================================================================


class IntegrityVerifier:
    """SHA-256 integrity verification for the raw dataset.

    On first load, computes and stores the hash in a metadata JSON file.
    On subsequent loads, recomputes and verifies against the stored hash,
    raising ``IntegrityError`` on mismatch.

    Args:
        metadata_dir: Directory where ``dataset_integrity.json`` is stored.
    """

    def __init__(self, metadata_dir: Path) -> None:
        self._metadata_dir = metadata_dir
        self._metadata_path = metadata_dir / _METADATA_FILE

    def compute_hash(self, file_path: Path) -> str:
        """Compute the SHA-256 hex digest of a file.

        Args:
            file_path: Path to the file to hash.

        Returns:
            Lowercase hexadecimal SHA-256 digest string.

        Raises:
            FileNotFoundError: If *file_path* does not exist.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Cannot hash: file not found: {file_path}")

        h = hashlib.new(_HASH_ALGORITHM)
        with open(file_path, "rb") as f:
            while chunk := f.read(_HASH_CHUNK_SIZE):
                h.update(chunk)

        digest = h.hexdigest()
        AuditLogger.log_file_access("HASH_COMPUTED", file_path, extra=f"sha256={digest[:16]}…")
        return digest

    def store_hash(self, file_path: Path, digest: str) -> None:
        """Persist the computed hash to the metadata JSON file.

        Args:
            file_path: Original file that was hashed.
            digest: SHA-256 hex digest to store.
        """
        self._metadata_dir.mkdir(parents=True, exist_ok=True)
        metadata: Dict[str, Any] = {}
        if self._metadata_path.exists():
            metadata = json.loads(self._metadata_path.read_text())

        metadata[str(file_path)] = {
            "sha256": digest,
            "verified_at": datetime.now(timezone.utc).isoformat(),
        }
        self._metadata_path.write_text(json.dumps(metadata, indent=2))
        AuditLogger.log_file_access("HASH_STORED", self._metadata_path)

    def verify(self, file_path: Path) -> str:
        """Compute hash and verify against stored value.

        On first invocation (no stored hash), computes and stores the hash.
        On subsequent invocations, verifies the recomputed hash matches.

        Args:
            file_path: Path to the dataset file.

        Returns:
            The verified SHA-256 hex digest.

        Raises:
            IntegrityError: If the recomputed hash does not match the stored one.
        """
        current_digest = self.compute_hash(file_path)

        if not self._metadata_path.exists():
            self.store_hash(file_path, current_digest)
            logger.info("First load — integrity baseline stored for %s", file_path.name)
            return current_digest

        metadata = json.loads(self._metadata_path.read_text())
        stored = metadata.get(str(file_path))

        if stored is None:
            self.store_hash(file_path, current_digest)
            logger.info("New file — integrity baseline stored for %s", file_path.name)
            return current_digest

        stored_digest = stored["sha256"]
        if current_digest != stored_digest:
            msg = (
                f"INTEGRITY VIOLATION: {file_path.name} — "
                f"expected {stored_digest[:16]}…, got {current_digest[:16]}…"
            )
            logger.critical(msg)
            raise IntegrityError(msg)

        AuditLogger.log_file_access(
            "INTEGRITY_VERIFIED", file_path,
            extra=f"sha256={current_digest[:16]}… ✓",
        )
        return current_digest


class IntegrityError(Exception):
    """Raised when a dataset file's SHA-256 hash does not match the stored baseline."""


# ===================================================================
# A01 — Broken Access Control: Path Validation
# ===================================================================


class PathValidator:
    """Validate file paths against workspace boundaries and traversal attacks.

    Args:
        workspace_root: The top-level project directory.  All resolved paths
                        must reside within this directory tree.
    """

    def __init__(self, workspace_root: Path) -> None:
        self._root = workspace_root.resolve()

    def validate_input_path(self, path: Path) -> Path:
        """Validate and resolve an input file path.

        Args:
            path: Raw path from configuration (may be relative).

        Returns:
            Resolved absolute path guaranteed to be within the workspace.

        Raises:
            ValueError: If the path contains traversal patterns.
            PermissionError: If the resolved path escapes the workspace.
            FileNotFoundError: If the resolved file does not exist.
        """
        self._check_traversal(path)
        resolved = (self._root / path).resolve()
        self._check_containment(resolved)

        if not resolved.exists():
            raise FileNotFoundError(f"Input path does not exist: {resolved}")

        AuditLogger.log_file_access("INPUT_VALIDATED", resolved)
        return resolved

    def validate_output_dir(self, path: Path) -> Path:
        """Validate and resolve an output directory path.

        Args:
            path: Raw output directory path from configuration.

        Returns:
            Resolved absolute path guaranteed to be within the workspace.
            The directory is created if it does not exist.

        Raises:
            ValueError: If the path contains traversal patterns.
            PermissionError: If the resolved path escapes the workspace.
        """
        self._check_traversal(path)
        resolved = (self._root / path).resolve()
        self._check_containment(resolved)
        resolved.mkdir(parents=True, exist_ok=True)
        AuditLogger.log_file_access("OUTPUT_DIR_VALIDATED", resolved)
        return resolved

    def check_read_only(self, path: Path) -> bool:
        """Check whether a file has read-only permissions (chmod 444 or similar).

        Args:
            path: Resolved path to check.

        Returns:
            True if the file is read-only for the owner, False otherwise.
        """
        mode = path.stat().st_mode
        is_readonly = not (mode & stat.S_IWUSR)
        if not is_readonly:
            logger.warning(
                "A01: File %s is writable (mode=%o). "
                "Consider chmod 444 for raw datasets.",
                path.name, mode & 0o777,
            )
        return is_readonly

    def _check_traversal(self, path: Path) -> None:
        """Reject paths containing traversal patterns.

        Raises:
            ValueError: If any segment contains '..', '~', or '$'.
        """
        path_str = str(path)
        for pattern in _TRAVERSAL_PATTERNS:
            if pattern in path_str:
                msg = f"A01: Path traversal detected — '{pattern}' in path: {path}"
                logger.error(msg)
                raise ValueError(msg)

    def _check_containment(self, resolved: Path) -> None:
        """Ensure a resolved path is inside the workspace.

        Raises:
            PermissionError: If the path escapes the workspace root.
        """
        try:
            resolved.relative_to(self._root)
        except ValueError:
            msg = (
                f"A01: Path escapes workspace — {resolved} "
                f"is outside {self._root}"
            )
            logger.error(msg)
            raise PermissionError(msg)


# ===================================================================
# A03 — Injection: Config Sanitization
# ===================================================================


class ConfigSanitizer:
    """Sanitize configuration inputs and enforce column allowlists.

    All string values from ``config.yaml`` are validated against a safe
    character regex.  Column names used in analysis are checked against
    the actual DataFrame columns (allowlist), preventing injection of
    arbitrary strings into SQL-like operations or shell commands.
    """

    @staticmethod
    def sanitize_string(value: str, field_name: str) -> str:
        """Validate that a config string contains only safe characters.

        Args:
            value: The raw string from config.yaml.
            field_name: Name of the config field (for error messages).

        Returns:
            The original string if it passes validation.

        Raises:
            ValueError: If the string contains disallowed characters.
        """
        if not _SAFE_STRING_RE.match(value):
            msg = f"A03: Unsafe characters in config field '{field_name}': {value!r}"
            logger.error(msg)
            raise ValueError(msg)
        return value

    @staticmethod
    def validate_column_allowlist(
        requested_columns: List[str],
        actual_columns: Set[str],
        context: str = "config",
    ) -> List[str]:
        """Validate that all requested column names exist in the DataFrame.

        Args:
            requested_columns: Column names from configuration.
            actual_columns: Set of actual DataFrame column names.
            context: Label for log messages (e.g. "required_columns").

        Returns:
            The validated column list (unchanged).

        Raises:
            ValueError: If any requested column is not in the allowlist.
        """
        invalid = [c for c in requested_columns if c not in actual_columns]
        if invalid:
            msg = (
                f"A03: Column allowlist violation in {context} — "
                f"unknown columns: {invalid}"
            )
            logger.error(msg)
            raise ValueError(msg)
        return requested_columns

    @staticmethod
    def sanitize_config_dict(raw: Dict[str, Any], prefix: str = "") -> None:
        """Recursively sanitize all string values in a config dict.

        Args:
            raw: Configuration dictionary (mutated in-place via validation).
            prefix: Dot-separated key path for error context.

        Raises:
            ValueError: If any string value contains disallowed characters.
        """
        for key, value in raw.items():
            full_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, str):
                ConfigSanitizer.sanitize_string(value, full_key)
            elif isinstance(value, dict):
                ConfigSanitizer.sanitize_config_dict(value, full_key)
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, str):
                        ConfigSanitizer.sanitize_string(item, f"{full_key}[{i}]")


# ===================================================================
# A09 — Security Logging: HIPAA-Compliant Audit Trail
# ===================================================================


class AuditLogger:
    """HIPAA-compliant security event logger.

    Design constraints:
        - All events include ISO-8601 UTC timestamps.
        - Biometric *values* are NEVER logged — only column *names*.
        - File access events are logged at INFO level.
        - Security violations are logged at ERROR / CRITICAL.
    """

    _audit_logger: Optional[logging.Logger] = None

    @classmethod
    def _get_logger(cls) -> logging.Logger:
        """Return (or lazily create) the dedicated audit logger."""
        if cls._audit_logger is None:
            cls._audit_logger = logging.getLogger("phase0.security.audit")
        return cls._audit_logger

    @classmethod
    def log_file_access(
        cls,
        event: str,
        path: Path,
        extra: str = "",
    ) -> None:
        """Log a file access event with an ISO-8601 timestamp.

        Args:
            event: Short event label (e.g. "READ", "HASH_COMPUTED").
            path: Path to the file being accessed.
            extra: Optional additional context (never biometric values).
        """
        ts = datetime.now(timezone.utc).isoformat()
        msg = f"[{ts}] {event}: {path}"
        if extra:
            msg += f"  ({extra})"
        cls._get_logger().info(msg)

    @classmethod
    def log_security_event(
        cls,
        event: str,
        detail: str,
        level: int = logging.WARNING,
    ) -> None:
        """Log a security-relevant event.

        Args:
            event: Short event label (e.g. "PATH_TRAVERSAL", "INTEGRITY_FAIL").
            detail: Human-readable description (must not contain PHI).
            level: Logging level (default WARNING).
        """
        ts = datetime.now(timezone.utc).isoformat()
        cls._get_logger().log(level, "[%s] %s: %s", ts, event, detail)

    @staticmethod
    def redact_biometric_values(
        columns: List[str],
        values: Any,
    ) -> Dict[str, str]:
        """Return a redacted dict safe for logging.

        Biometric columns have their values replaced with ``"[REDACTED]"``.
        Non-biometric columns are represented as ``"<present>"``
        (column name only — no raw values).

        Args:
            columns: List of column names.
            values: Ignored — values are never included.

        Returns:
            Dict mapping column name to a safe placeholder string.
        """
        result: Dict[str, str] = {}
        for col in columns:
            if col in BIOMETRIC_COLUMNS:
                result[col] = "[REDACTED-PHI]"
            else:
                result[col] = "<present>"
        return result
