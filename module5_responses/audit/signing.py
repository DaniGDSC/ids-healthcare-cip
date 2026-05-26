"""ECDSA P-256 signing primitives for the audit log."""
from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path

try:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import ec

    _HAVE_CRYPTOGRAPHY = True
except ImportError:  # pragma: no cover
    _HAVE_CRYPTOGRAPHY = False

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "results/reports"

# Default key locations. The private key is auto-bootstrapped on first
# run if no operator-provided key is available via IOMT_AUDIT_SIGNING_KEY.
# The public key is written next to the audit log so verifiers find it
# without configuration.
DEFAULT_PRIVATE_KEY_PATH = Path.home() / ".iomt-ids" / "audit_signing_key.pem"
DEFAULT_PUBLIC_KEY_PATH = OUTPUT_DIR / "audit_signing_key.pub.pem"

SIGNATURE_ALG = "ECDSA_P256_SHA256"


def _require_cryptography() -> None:
    if not _HAVE_CRYPTOGRAPHY:
        raise RuntimeError(
            "audit log signing requires the `cryptography` package. "
            "Install it with `pip install cryptography>=42.0`."
        )


def _bootstrap_local_key(private_path: Path, public_path: Path) -> None:
    """Generate a fresh ECDSA P-256 keypair on first run.

    Private key is written with 0600 permissions to a user-local directory
    (default: ~/.iomt-ids). The public key is written next to the audit
    log so verifiers find it without extra configuration.

    SECURITY WARNING: an auto-generated key is convenient for development
    but offers no protection against an attacker who already has shell
    access on the host. For production, set IOMT_AUDIT_SIGNING_KEY to a
    key issued by your operator (HSM, KMS, or operator-provisioned PEM)
    so the private key never lives next to the data it signs.
    """
    _require_cryptography()
    private_path.parent.mkdir(parents=True, exist_ok=True)
    public_path.parent.mkdir(parents=True, exist_ok=True)

    private_key = ec.generate_private_key(ec.SECP256R1())
    pem_priv = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    pem_pub = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    private_path.write_bytes(pem_priv)
    try:
        os.chmod(private_path, 0o600)
    except OSError:
        pass
    public_path.write_bytes(pem_pub)
    logger.warning(
        "AuditLogger: bootstrapped a local ECDSA P-256 signing key at %s. "
        "Replace with an operator-provisioned key for production.",
        private_path,
    )


def _load_signing_key(
    private_path: Path | None = None,
    public_path: Path | None = None,
):
    """Load (or bootstrap) the ECDSA private key used to sign records.

    Resolution order:
      1. IOMT_AUDIT_SIGNING_KEY environment variable (operator override)
      2. Explicit ``private_path`` argument
      3. DEFAULT_PRIVATE_KEY_PATH (~/.iomt-ids/audit_signing_key.pem)
      4. Bootstrap a fresh key at the default path
    """
    _require_cryptography()
    env_path = os.environ.get("IOMT_AUDIT_SIGNING_KEY")
    if env_path:
        private_path = Path(env_path)
    elif private_path is None:
        private_path = DEFAULT_PRIVATE_KEY_PATH

    public_path = public_path or DEFAULT_PUBLIC_KEY_PATH

    if not private_path.exists():
        _bootstrap_local_key(private_path, public_path)

    private_key = serialization.load_pem_private_key(
        private_path.read_bytes(), password=None
    )

    # Always (re)export the matching public key next to the audit log so
    # verification works without operator intervention. Idempotent.
    pem_pub = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    public_path.parent.mkdir(parents=True, exist_ok=True)
    if not public_path.exists() or public_path.read_bytes() != pem_pub:
        public_path.write_bytes(pem_pub)

    key_id = "ecdsa-p256-" + hashlib.sha256(pem_pub).hexdigest()[:16]
    return private_key, public_path, key_id


def _canonical_json(record: dict) -> bytes:
    """Deterministic JSON encoding used for hashing and signing."""
    return json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")


__all__ = [
    "_HAVE_CRYPTOGRAPHY",
    "SIGNATURE_ALG",
    "DEFAULT_PRIVATE_KEY_PATH",
    "DEFAULT_PUBLIC_KEY_PATH",
    "OUTPUT_DIR",
    "_require_cryptography",
    "_bootstrap_local_key",
    "_load_signing_key",
    "_canonical_json",
]
