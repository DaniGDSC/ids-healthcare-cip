"""ECDSA-signed sidecar for non-pickle artefact pairs.

Counterpart to ``common.signed_pickle`` for artefacts that cannot be
pickled (or shouldn't be — see tier 0 F2 and tier 2 F1). Designed for:

  * JSON + Keras-weights pairs (the DAE detector).
  * NPZ + JSON-metadata pairs (the risk-scores artefact).

The trust model is identical to ``signed_pickle``:
  * The signer hashes a *content digest* — for a pair, that's
    ``sha256(canonical_json(meta_bytes) || sha256(payload_bytes))`` — and
    signs the digest under the Module 5 ECDSA P-256 key.
  * A ``.sig`` JSON sidecar carries the digest, the base64 signature,
    the signing key id, the algorithm, and a UTC timestamp.
  * Verification: recompute the content digest from the *file bytes on
    disk*, compare to the sidecar digest, then verify the signature
    against the digest. If anything fails the artefact is refused.

Use ``write_signed_pair`` / ``verify_signed_pair`` to attach a sidecar to
any (meta_path, payload_path) pair. Use ``write_signed_single`` /
``verify_signed_single`` for a single-file artefact (e.g. JSON without
a binary companion). The single-file form is what
``module3_risk_scoring/io.py`` uses for the new
``risk_scores.meta.json`` sidecar.
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

logger = logging.getLogger(__name__)

_SIDECAR_FORMAT = "phase2.signed_sidecar.v1"
_SIG_SUFFIX = ".sig"


class SignedSidecarError(Exception):
    """Raised when a signed-sidecar artefact cannot be verified."""


def _content_digest_pair(meta_bytes: bytes, payload_bytes: bytes) -> str:
    payload_sha = hashlib.sha256(payload_bytes).digest()
    return hashlib.sha256(meta_bytes + payload_sha).hexdigest()


def _content_digest_single(meta_bytes: bytes) -> str:
    return hashlib.sha256(meta_bytes).hexdigest()


def _sign(digest_hex: str) -> tuple[str, str]:
    """Return (base64_signature, signing_key_id)."""
    from module5_responses.signing import HAVE_CRYPTOGRAPHY, load_signing_key

    if not HAVE_CRYPTOGRAPHY:
        raise RuntimeError(
            "signed_sidecar requires the `cryptography` package. "
            "Install it with `pip install cryptography>=42`."
        )
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import ec

    private_key, _public_path, key_id = load_signing_key()
    signature = private_key.sign(bytes.fromhex(digest_hex), ec.ECDSA(hashes.SHA256()))
    return base64.b64encode(signature).decode("ascii"), key_id


def _verify(digest_hex: str, signature_b64: str, sidecar_key_id: str) -> None:
    """Verify the signature, raising SignedSidecarError on mismatch."""
    from module5_responses.signing import HAVE_CRYPTOGRAPHY, load_signing_key

    if not HAVE_CRYPTOGRAPHY:
        raise SignedSidecarError(
            "cryptography package not installed; cannot verify."
        )
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec

    _private, public_path, expected_key_id = load_signing_key()
    if sidecar_key_id != expected_key_id:
        raise SignedSidecarError(
            f"signing_key_id mismatch: sidecar claims {sidecar_key_id!r} "
            f"but local key is {expected_key_id!r}. Re-sign or restore."
        )
    public_key = serialization.load_pem_public_key(public_path.read_bytes())
    try:
        public_key.verify(
            base64.b64decode(signature_b64),
            bytes.fromhex(digest_hex),
            ec.ECDSA(hashes.SHA256()),
        )
    except InvalidSignature as exc:
        raise SignedSidecarError("ECDSA verification failed.") from exc


def _atomic_write(path: Path, payload: bytes) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(payload)
    os.replace(tmp, path)
    try:
        os.chmod(path, 0o640)
    except OSError as exc:
        logger.warning("chmod 0640 on %s failed: %s", path, exc)


def _write_sidecar(sig_path: Path, sidecar: dict[str, Any]) -> None:
    _atomic_write(sig_path, json.dumps(sidecar, indent=2, sort_keys=True).encode("utf-8"))


def _read_sidecar(sig_path: Path) -> dict[str, Any]:
    if not sig_path.exists():
        raise SignedSidecarError(
            f"No signature sidecar at {sig_path}. Refusing to consume the "
            "artefact: an unsigned non-pickle bundle is the exact gap "
            "tier 0 F2 / tier 2 F1 closed."
        )
    try:
        body = json.loads(sig_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SignedSidecarError(
            f"Signature sidecar {sig_path} is not valid JSON: {exc}"
        ) from exc
    if body.get("format") != _SIDECAR_FORMAT:
        raise SignedSidecarError(
            f"{sig_path} is not a {_SIDECAR_FORMAT} sidecar "
            f"(got format={body.get('format')!r})"
        )
    return body


# ── public API: pair ───────────────────────────────────────────────


def write_signed_pair(
    meta_path: Path,
    payload_path: Path,
    *,
    sig_path: Path | None = None,
) -> Path:
    """Sign the pair (meta_path, payload_path) and write the sidecar.

    Both files must already exist on disk; this helper only attaches a
    signature. Use the typical write order:

        save_meta(meta_path)
        save_payload(payload_path)
        write_signed_pair(meta_path, payload_path)

    Returns the path to the sidecar.
    """
    meta_path = Path(meta_path)
    payload_path = Path(payload_path)
    if sig_path is None:
        sig_path = meta_path.with_suffix(meta_path.suffix + _SIG_SUFFIX)
    meta_bytes = meta_path.read_bytes()
    payload_bytes = payload_path.read_bytes()
    digest_hex = _content_digest_pair(meta_bytes, payload_bytes)
    sig_b64, key_id = _sign(digest_hex)
    _write_sidecar(
        sig_path,
        {
            "format": _SIDECAR_FORMAT,
            "format_version": 1,
            "kind": "pair",
            "meta_file": meta_path.name,
            "payload_file": payload_path.name,
            "signature_alg": "ECDSA_P256_SHA256",
            "signing_key_id": key_id,
            "content_digest_sha256": digest_hex,
            "signature": sig_b64,
            "signed_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    logger.info(
        "signed_sidecar.pair: wrote %s (digest=%s, key=%s)",
        sig_path.name, digest_hex[:16], key_id,
    )
    return sig_path


def verify_signed_pair(meta_path: Path, payload_path: Path) -> None:
    """Verify the signed pair (meta_path, payload_path). Raise on failure."""
    meta_path = Path(meta_path)
    payload_path = Path(payload_path)
    sig_path = meta_path.with_suffix(meta_path.suffix + _SIG_SUFFIX)
    sidecar = _read_sidecar(sig_path)
    if sidecar.get("kind") != "pair":
        raise SignedSidecarError(
            f"{sig_path} is not a pair sidecar (kind={sidecar.get('kind')!r})"
        )
    if not meta_path.exists():
        raise FileNotFoundError(f"Cannot verify pair: meta not found: {meta_path}")
    if not payload_path.exists():
        raise FileNotFoundError(f"Cannot verify pair: payload not found: {payload_path}")
    meta_bytes = meta_path.read_bytes()
    payload_bytes = payload_path.read_bytes()
    digest_hex = _content_digest_pair(meta_bytes, payload_bytes)
    if digest_hex != sidecar.get("content_digest_sha256"):
        raise SignedSidecarError(
            f"content digest mismatch for ({meta_path.name}, "
            f"{payload_path.name}): bytes are stale or tampered."
        )
    _verify(digest_hex, sidecar["signature"], sidecar.get("signing_key_id", ""))
    logger.info(
        "signed_sidecar.pair: verified %s + %s (digest=%s)",
        meta_path.name, payload_path.name, digest_hex[:16],
    )


# ── public API: single ────────────────────────────────────────────


def write_signed_single(meta_path: Path, *, sig_path: Path | None = None) -> Path:
    """Sign a single-file artefact and write the sidecar."""
    meta_path = Path(meta_path)
    if sig_path is None:
        sig_path = meta_path.with_suffix(meta_path.suffix + _SIG_SUFFIX)
    meta_bytes = meta_path.read_bytes()
    digest_hex = _content_digest_single(meta_bytes)
    sig_b64, key_id = _sign(digest_hex)
    _write_sidecar(
        sig_path,
        {
            "format": _SIDECAR_FORMAT,
            "format_version": 1,
            "kind": "single",
            "meta_file": meta_path.name,
            "signature_alg": "ECDSA_P256_SHA256",
            "signing_key_id": key_id,
            "content_digest_sha256": digest_hex,
            "signature": sig_b64,
            "signed_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    logger.info(
        "signed_sidecar.single: wrote %s (digest=%s, key=%s)",
        sig_path.name, digest_hex[:16], key_id,
    )
    return sig_path


def verify_signed_single(meta_path: Path) -> None:
    meta_path = Path(meta_path)
    sig_path = meta_path.with_suffix(meta_path.suffix + _SIG_SUFFIX)
    sidecar = _read_sidecar(sig_path)
    if sidecar.get("kind") != "single":
        raise SignedSidecarError(
            f"{sig_path} is not a single sidecar (kind={sidecar.get('kind')!r})"
        )
    if not meta_path.exists():
        raise FileNotFoundError(f"Cannot verify single: not found: {meta_path}")
    meta_bytes = meta_path.read_bytes()
    digest_hex = _content_digest_single(meta_bytes)
    if digest_hex != sidecar.get("content_digest_sha256"):
        raise SignedSidecarError(
            f"content digest mismatch for {meta_path.name}: bytes stale or tampered."
        )
    _verify(digest_hex, sidecar["signature"], sidecar.get("signing_key_id", ""))
    logger.info("signed_sidecar.single: verified %s (digest=%s)", meta_path.name, digest_hex[:16])


__all__ = [
    "SignedSidecarError",
    "write_signed_pair",
    "verify_signed_pair",
    "write_signed_single",
    "verify_signed_single",
]
