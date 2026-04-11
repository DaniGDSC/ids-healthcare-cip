"""ECDSA-signed pickle I/O for the IDS pipeline.

Why this exists
---------------
Several Phase 2 model artefacts (sklearn ``Pipeline`` objects with
fitted classifiers and SMOTE wrappers) are loaded at inference time by
Module 3/4 via ``joblib.load``. ``joblib.load`` is a thin wrapper
around ``pickle.load`` and ``pickle.load`` deserialises arbitrary
Python — a malicious pickle that lands in ``results/models/`` (via a
compromised CI runner, a malicious PR that touches that directory, or
a tampered build host) gives the attacker code execution on every
machine that subsequently runs inference.

We cannot eliminate the pickle entirely for these objects (the fitted
sklearn ``Pipeline`` does not have a JSON-serialisable representation)
without rewriting the entire training stack. Instead, we add an
**ECDSA P-256 signature** alongside each pickle — produced at write
time with the same key Module 5 uses for its audit chain — and
**refuse to deserialise** unless the signature verifies on load.

This does NOT make the pickle bytes safe to read from a hostile party
who controls the file. It makes the pickle's authorship cryptographically
verifiable: if an attacker tampers with the file (or substitutes a
malicious replacement), ``loads_signed`` raises and the deserialiser
is never called. The trust boundary is now the private key, not the
filesystem.

Sidecar format
--------------
Each signed pickle ``foo.pkl`` is accompanied by ``foo.pkl.sig``
containing JSON of the shape::

    {
      "format": "phase2.signed_pickle.v1",
      "format_version": 1,
      "signature_alg": "ECDSA_P256_SHA256",
      "signing_key_id": "<id from Module 5>",
      "sha256":         "<hex digest of the pickle bytes>",
      "signature":      "<base64 ECDSA signature over sha256 bytes>",
      "signed_at":      "<ISO-8601 UTC>",
    }

The signature is computed over the SHA-256 digest (not the raw bytes)
so verification on load is cheap and constant in pickle size.

Threat model
------------
- ✅ Tampered file (attacker rewrites bytes)         → caught
- ✅ Substituted file (attacker replaces wholesale)  → caught
- ✅ Stale signature (attacker keeps old sig.json)   → caught (sha256 check)
- ❌ Compromise of the private key                   → game over
- ❌ Malicious authoring at training time            → unrelated to this layer
- ❌ Bugs in the model code that the pickle restores → unrelated to this layer

Use ``loads_signed`` for every load site that touches a Phase 2 model
artefact. Plain ``joblib.load`` of these files is forbidden.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib

logger = logging.getLogger(__name__)

_SIDECAR_FORMAT = "phase2.signed_pickle.v1"
_SIG_SUFFIX = ".sig"


class SignedPickleError(Exception):
    """Raised when a signed pickle cannot be verified.

    Distinct from ``IOError`` so callers can decide whether to refuse
    the load (production) or to fall back to a quarantine action
    (forensic recovery).
    """


# ─────────────────────────────────────────────────────────────────────
# Internal: ECDSA key access via Module 5
# ─────────────────────────────────────────────────────────────────────

def _get_signing_key():
    """Lazily load the Module 5 ECDSA private key.

    Returns ``(private_key, signing_key_id)``. Raises ``RuntimeError``
    if the ``cryptography`` package is missing or the key is not
    available — the caller MUST treat that as a fatal error rather
    than skipping the signature.
    """
    from pipeline.module5_responses.module5_pipeline import (
        _HAVE_CRYPTOGRAPHY,
        _load_signing_key,
    )
    if not _HAVE_CRYPTOGRAPHY:
        raise RuntimeError(
            "signed_pickle requires the `cryptography` package. "
            "Install it with `pip install cryptography>=42.0`."
        )
    private_key, _public_path, signing_key_id = _load_signing_key()
    return private_key, signing_key_id


def _get_verifying_key():
    """Load the Module 5 public key for verification.

    Returns ``(public_key, signing_key_id)``. Same lazy import pattern
    as ``_get_signing_key`` so callers that only need to verify don't
    pull cryptography until first use.
    """
    from pipeline.module5_responses.module5_pipeline import (
        _HAVE_CRYPTOGRAPHY,
        _load_signing_key,
    )
    if not _HAVE_CRYPTOGRAPHY:
        raise RuntimeError(
            "signed_pickle requires the `cryptography` package. "
            "Install it with `pip install cryptography>=42.0`."
        )
    from cryptography.hazmat.primitives import serialization

    _private, public_path, signing_key_id = _load_signing_key()
    public_key = serialization.load_pem_public_key(public_path.read_bytes())
    return public_key, signing_key_id


# ─────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────

def dumps_signed(obj: Any, path: Path) -> Path:
    """Pickle *obj* to *path* and write a signature sidecar next to it.

    Atomic write via ``tmp + os.replace`` so a crash mid-write cannot
    leave a half-written pickle that the verifier would refuse on next
    load — and a half-missing sidecar that the operator would have to
    manually clean up.

    Args:
        obj: Object to pickle. Joblib's protocol-5 serialisation is
            used so large numpy arrays land in efficient buffers.
        path: Destination ``.pkl`` path. The sidecar is written to
            ``path.with_suffix(path.suffix + ".sig")``.

    Returns:
        The path to the written pickle.

    Raises:
        RuntimeError: if the signing key is unavailable.
    """
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import ec

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    private_key, signing_key_id = _get_signing_key()

    # Step 1: serialise to a temp file, hash, then atomically rename.
    tmp_pkl = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(obj, tmp_pkl)
    raw = tmp_pkl.read_bytes()
    digest = hashlib.sha256(raw).digest()
    digest_hex = digest.hex()

    # Step 2: sign the digest (not the raw bytes — keeps the
    # signature step constant-time in pickle size).
    signature = private_key.sign(digest, ec.ECDSA(hashes.SHA256()))

    sidecar = {
        "format":         _SIDECAR_FORMAT,
        "format_version": 1,
        "signature_alg":  "ECDSA_P256_SHA256",
        "signing_key_id": signing_key_id,
        "sha256":         digest_hex,
        "signature":      base64.b64encode(signature).decode("ascii"),
        "signed_at":      datetime.now(timezone.utc).isoformat(),
    }
    sig_path = path.with_suffix(path.suffix + _SIG_SUFFIX)
    tmp_sig = sig_path.with_suffix(sig_path.suffix + ".tmp")
    tmp_sig.write_text(json.dumps(sidecar, indent=2, sort_keys=True))

    # Step 3: atomically promote both files. Sidecar first so a
    # consumer that races us never sees a pickle without its
    # signature.
    os.replace(tmp_sig, sig_path)
    os.replace(tmp_pkl, path)

    logger.info(
        "signed_pickle: wrote %s (sha256=%s, key=%s)",
        path.name, digest_hex[:16], signing_key_id,
    )
    return path


def loads_signed(path: Path) -> Any:
    """Verify the signature for *path* and deserialise.

    The pickle bytes are NOT passed to ``joblib.load`` until the
    signature has verified against the Module 5 public key. If
    verification fails for any reason (missing sidecar, sha256
    mismatch, invalid signature, foreign format) ``SignedPickleError``
    is raised and the deserialiser is never invoked.

    Args:
        path: Path to a ``.pkl`` previously written by ``dumps_signed``.

    Returns:
        The unpickled object.

    Raises:
        FileNotFoundError: if the pickle does not exist.
        SignedPickleError: if signature verification fails for any reason.
    """
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import ec

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Signed pickle not found: {path}")

    sig_path = path.with_suffix(path.suffix + _SIG_SUFFIX)
    if not sig_path.exists():
        raise SignedPickleError(
            f"No signature sidecar at {sig_path}. Refusing to deserialise "
            f"{path.name}: an unsigned pickle is exactly the RCE sink "
            f"this module exists to close."
        )

    try:
        sidecar = json.loads(sig_path.read_text())
    except json.JSONDecodeError as exc:
        raise SignedPickleError(
            f"Signature sidecar {sig_path} is not valid JSON: {exc}"
        ) from exc

    if sidecar.get("format") != _SIDECAR_FORMAT:
        raise SignedPickleError(
            f"{sig_path} is not a {_SIDECAR_FORMAT} sidecar "
            f"(got format={sidecar.get('format')!r})"
        )

    raw = path.read_bytes()
    actual_digest = hashlib.sha256(raw).hexdigest()
    expected_digest = sidecar.get("sha256", "")
    if actual_digest != expected_digest:
        raise SignedPickleError(
            f"sha256 mismatch for {path.name}: pickle bytes are stale "
            f"or tampered (expected {expected_digest[:16]}, got "
            f"{actual_digest[:16]}). Refusing to deserialise."
        )

    public_key, expected_key_id = _get_verifying_key()
    sidecar_key_id = sidecar.get("signing_key_id")
    if sidecar_key_id != expected_key_id:
        raise SignedPickleError(
            f"signing_key_id mismatch for {path.name}: sidecar claims "
            f"{sidecar_key_id!r} but the local public key is "
            f"{expected_key_id!r}. The pickle was signed by a different "
            f"key (or the local key has been rotated and the artefact "
            f"needs to be re-signed)."
        )

    try:
        signature = base64.b64decode(sidecar["signature"])
        public_key.verify(
            signature,
            bytes.fromhex(actual_digest),
            ec.ECDSA(hashes.SHA256()),
        )
    except InvalidSignature as exc:
        raise SignedPickleError(
            f"ECDSA verification failed for {path.name}: the signature "
            f"in {sig_path.name} does not validate. Refusing to "
            f"deserialise."
        ) from exc
    except Exception as exc:  # noqa: BLE001
        raise SignedPickleError(
            f"signature verification raised an unexpected error for "
            f"{path.name}: {exc}"
        ) from exc

    # Only NOW do we touch joblib.load.
    obj = joblib.load(path)
    logger.info(
        "signed_pickle: verified %s (sha256=%s, key=%s)",
        path.name, actual_digest[:16], sidecar_key_id,
    )
    return obj
