"""Audit-log primitives for Module 5 — ECDSA P-256 signed, hash-chained JSONL.

Public surface re-exports key signing helpers and the :class:`AuditLogger`.
"""
from __future__ import annotations

from .logger import (
    ARCHIVE_DIR,
    DEFAULT_RETENTION_DAYS,
    OUTPUT_DIR,
    AuditLogger,
)
from .signing import (
    DEFAULT_PRIVATE_KEY_PATH,
    DEFAULT_PUBLIC_KEY_PATH,
    SIGNATURE_ALG,
    _bootstrap_local_key,
    _canonical_json,
    _HAVE_CRYPTOGRAPHY,
    _load_signing_key,
    _require_cryptography,
)

__all__ = [
    "AuditLogger",
    "ARCHIVE_DIR",
    "OUTPUT_DIR",
    "DEFAULT_RETENTION_DAYS",
    "DEFAULT_PRIVATE_KEY_PATH",
    "DEFAULT_PUBLIC_KEY_PATH",
    "SIGNATURE_ALG",
    "_canonical_json",
    "_load_signing_key",
    "_bootstrap_local_key",
    "_require_cryptography",
    "_HAVE_CRYPTOGRAPHY",
]
