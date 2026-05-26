"""Public signing API for Module 5's ECDSA P-256 audit-log primitives.

Wraps internal ``module5_responses.audit.signing`` symbols
(``_canonical_json``, ``_load_signing_key``, ``_HAVE_CRYPTOGRAPHY``)
under stable public names so other modules don't depend on private API.

Stable contract for cross-module callers
----------------------------------------
- ``HAVE_CRYPTOGRAPHY``: bool — True iff the ``cryptography`` package
  is importable in the current environment.
- ``canonical_json(record: dict) -> bytes``: deterministic JSON encoding
  (sorted keys, compact separators) used for hashing and signing.
- ``load_signing_key(private_path=None, public_path=None)
  -> (private_key, public_path, key_id)``: load (or bootstrap) the
  ECDSA P-256 keypair. See ``audit.signing._load_signing_key`` for
  resolution order.

Module 0, Module 1, common/signed_pickle.py, and tests import from here.
"""
from __future__ import annotations

from module5_responses.audit.signing import (
    _HAVE_CRYPTOGRAPHY as HAVE_CRYPTOGRAPHY,
    _canonical_json as canonical_json,
    _load_signing_key as load_signing_key,
)

__all__ = [
    "HAVE_CRYPTOGRAPHY",
    "canonical_json",
    "load_signing_key",
]
