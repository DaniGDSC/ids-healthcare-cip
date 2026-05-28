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

# Tier 0 F3 / tier 3 F1 — VCS-pinned expected key fingerprint. When this
# file is present, _load_signing_key refuses any key whose derived id
# does not match. Rotation becomes a deliberate one-line diff reviewed
# in the same PR as the new key fingerprint.
EXPECTED_KEY_ID_FILE = PROJECT_ROOT / "config" / "signing_key_id.txt"

# Artefact-presence markers used to refuse silent re-bootstrap. If ANY
# of these exist on disk but the local private key does not, the system
# is in a "key lost, artefacts present" state — auto-bootstrapping
# would (a) invalidate every existing artefact's signature and (b) let
# an attacker who deleted the key sign whatever they want next.
_ARTEFACT_PRESENCE_GLOBS = (
    "results/models/*.pkl.sig",
    "results/reports/audit_log.jsonl",
    "module0_analysis/dataset_integrity.json",
)

SIGNATURE_ALG = "ECDSA_P256_SHA256"


class SigningKeyTrustError(RuntimeError):
    """Raised when the signing key fails a VCS-pinned trust check or
    auto-bootstrap is refused because signed artefacts already exist
    on disk."""


def _read_pinned_key_id() -> str | None:
    """Read the VCS-tracked expected key id (None when no pin)."""
    if not EXPECTED_KEY_ID_FILE.exists():
        return None
    try:
        return EXPECTED_KEY_ID_FILE.read_text(encoding="utf-8").strip() or None
    except OSError as exc:
        logger.warning(
            "_load_signing_key: failed to read %s (%s); skipping pin check",
            EXPECTED_KEY_ID_FILE, exc,
        )
        return None


def _artefacts_present() -> list[str]:
    """Return the relative paths of any signed artefacts on disk."""
    found: list[str] = []
    for pattern in _ARTEFACT_PRESENCE_GLOBS:
        for path in PROJECT_ROOT.glob(pattern):
            if path.exists():
                found.append(str(path.relative_to(PROJECT_ROOT)))
    return found


def _require_cryptography() -> None:
    if not _HAVE_CRYPTOGRAPHY:
        raise RuntimeError(
            "audit log signing requires the `cryptography` package. "
            "Install it with `pip install cryptography>=42.0`."
        )


def _bootstrap_local_key(private_path: Path, public_path: Path) -> None:
    """Generate a fresh ECDSA P-256 keypair on first run.

    Tier 0 F3 / tier 3 F4 hardening:
      - Refuses to bootstrap when signed artefacts already exist on
        disk (the "key was lost or wiped" case). The operator must
        explicitly run a rotation CLI in that scenario so the chain
        break is a deliberate, auditable event.
      - chmod 0700 on the parent directory so a group member cannot
        replace the key behind the operator's back.
      - chmod 0600 on the private key file (existing behaviour).

    SECURITY WARNING: an auto-generated key is convenient for first-
    install bootstrap but offers no protection against an attacker who
    already has shell access on the host. For production, set
    IOMT_AUDIT_SIGNING_KEY to a key issued by your operator (HSM, KMS,
    or operator-provisioned PEM) so the private key never lives next
    to the data it signs.
    """
    _require_cryptography()

    artefacts = _artefacts_present()
    if artefacts:
        raise SigningKeyTrustError(
            "Refusing to auto-bootstrap a new signing key while signed "
            "artefacts already exist on disk:\n  "
            + "\n  ".join(artefacts)
            + "\nBootstrapping now would orphan every existing signature. "
            "If the previous key is genuinely lost, run the rotation CLI "
            "(operator runbook) so the break is recorded as a CRITICAL "
            "audit event. If the previous key is recoverable, restore it "
            "to "
            + str(private_path)
            + " before retrying."
        )

    private_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(private_path.parent, 0o700)
    except OSError as exc:
        logger.warning(
            "_bootstrap_local_key: chmod 0700 on %s failed: %s. "
            "Tighten directory permissions manually before relying on "
            "the chain.",
            private_path.parent, exc,
        )
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
    except OSError as exc:
        logger.error(
            "_bootstrap_local_key: chmod 0600 on %s failed: %s. "
            "ABORTING — refusing to leave a world-readable signing key on "
            "disk.",
            private_path, exc,
        )
        try:
            private_path.unlink()
        except OSError:
            pass
        raise SigningKeyTrustError(
            f"Could not enforce 0600 on {private_path}: {exc}"
        ) from exc
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
      4. Bootstrap a fresh key at the default path — REFUSED if signed
         artefacts already exist on disk (tier 0 F3 hardening).

    Trust pin (tier 0 F3 / tier 3 F1): if
    ``config/signing_key_id.txt`` exists, the derived ``key_id`` MUST
    match its contents. Mismatch raises :class:`SigningKeyTrustError`
    rather than silently trusting a re-bootstrapped key.
    """
    _require_cryptography()
    env_path = os.environ.get("IOMT_AUDIT_SIGNING_KEY")
    if env_path:
        private_path = Path(env_path)
    elif private_path is None:
        private_path = DEFAULT_PRIVATE_KEY_PATH

    public_path = public_path or DEFAULT_PUBLIC_KEY_PATH

    if not private_path.exists():
        # _bootstrap_local_key itself raises SigningKeyTrustError when
        # signed artefacts are on disk; otherwise it writes a new pair.
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

    # Tier 0 F3 / tier 3 F1: VCS-pinned key id enforcement.
    pinned = _read_pinned_key_id()
    if pinned and pinned != key_id:
        raise SigningKeyTrustError(
            f"Signing key id mismatch: loaded key has id {key_id!r} but "
            f"the VCS pin in {EXPECTED_KEY_ID_FILE} expects {pinned!r}. "
            "Either restore the previous key, OR update the pin file in "
            "a code-reviewed commit if this is an intentional rotation."
        )

    return private_key, public_path, key_id


def _canonical_json(record: dict) -> bytes:
    """Deterministic JSON encoding used for hashing and signing."""
    return json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")


__all__ = [
    "_HAVE_CRYPTOGRAPHY",
    "SIGNATURE_ALG",
    "DEFAULT_PRIVATE_KEY_PATH",
    "DEFAULT_PUBLIC_KEY_PATH",
    "EXPECTED_KEY_ID_FILE",
    "OUTPUT_DIR",
    "SigningKeyTrustError",
    "_require_cryptography",
    "_bootstrap_local_key",
    "_load_signing_key",
    "_canonical_json",
    "_read_pinned_key_id",
    "_artefacts_present",
]
