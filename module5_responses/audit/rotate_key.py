"""Operator CLI to rotate the audit-chain signing key.

Sprint-1 hardening: ``_bootstrap_local_key`` refuses to silently
generate a fresh key when signed artefacts already exist on disk
(tier 0 F3). The only sanctioned way to break a chain is through this
CLI, which:

  1. Demands explicit operator acknowledgement
     (``--i-understand-this-orphans-old-signatures``).
  2. Archives the current ``audit_log.jsonl`` so the prior chain is
     preserved alongside its ``last_integrity_hash``.
  3. Moves the existing private key to a retired location.
  4. Bootstraps a fresh keypair with the same on-disk hardening
     (``0700`` on the directory, ``0600`` on the key file).
  5. Writes a ``SIGNING_KEY_ROTATED`` event into the new chain whose
     ``cross_rotation_anchor`` references the archived chain's last
     integrity hash.
  6. Prints the new key id so the operator can update
     ``config/signing_key_id.txt`` in a code-reviewed commit.

For the LOST-key emergency (no archived copy of the prior private
key), pass ``--key-lost --i-acknowledge-chain-break``. The CLI will
emit ``SIGNING_KEY_LOST`` instead and skip the archive-with-prior-key
step (because we cannot sign the rotation event under the old key).

Usage::

    python -m module5_responses.audit.rotate_key \\
        --i-understand-this-orphans-old-signatures

    python -m module5_responses.audit.rotate_key \\
        --key-lost --i-acknowledge-chain-break
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


def _archive_current_log(log_path: Path, archive_dir: Path) -> Path | None:
    """Move the active log under archive_dir with a timestamp suffix.

    Returns the archive path (or None when no active log exists).
    """
    if not log_path.exists() or log_path.stat().st_size == 0:
        return None
    archive_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archived = archive_dir / f"{log_path.stem}.{stamp}.rotation.jsonl"
    log_path.rename(archived)
    return archived


def _read_last_integrity_hash(path: Path) -> str | None:
    if not path or not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 4096))
            tail = f.read()
    except OSError:
        return None
    for raw in reversed(tail.split(b"\n")):
        s = raw.strip()
        if not s:
            continue
        try:
            rec = json.loads(s)
        except (json.JSONDecodeError, ValueError):
            continue
        ih = rec.get("integrity_hash")
        if isinstance(ih, str) and len(ih) == 64:
            return ih
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m module5_responses.audit.rotate_key",
        description=(
            "Rotate the audit-chain ECDSA signing key. Orphans every "
            "existing signature; only run with operator authorisation."
        ),
    )
    parser.add_argument(
        "--i-understand-this-orphans-old-signatures",
        action="store_true",
        dest="ack_orphan",
        help="Explicit acknowledgement that existing signatures will no "
             "longer verify after rotation.",
    )
    parser.add_argument(
        "--key-lost",
        action="store_true",
        help="The prior private key is gone; skip the under-old-key sign step.",
    )
    parser.add_argument(
        "--i-acknowledge-chain-break",
        action="store_true",
        dest="ack_break",
        help="Required with --key-lost; the new chain is genesis-rooted.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from module5_responses.audit.signing import (
        DEFAULT_PRIVATE_KEY_PATH,
        DEFAULT_PUBLIC_KEY_PATH,
        OUTPUT_DIR,
        _artefacts_present,
        _bootstrap_local_key,
        _load_signing_key,
    )
    from module5_responses.audit.logger import AuditLogger

    log_path = OUTPUT_DIR / "audit_log.jsonl"
    archive_dir = OUTPUT_DIR / "audit_archive"

    if args.key_lost:
        if not args.ack_break:
            print(
                "REFUSED: --key-lost requires --i-acknowledge-chain-break",
                file=sys.stderr,
            )
            return 2
        event = "SIGNING_KEY_LOST"
    else:
        if not args.ack_orphan:
            print(
                "REFUSED: pass --i-understand-this-orphans-old-signatures "
                "to authorise rotation.",
                file=sys.stderr,
            )
            return 2
        event = "SIGNING_KEY_ROTATED"

    # 1. Capture the prior chain's last hash before we move the file.
    prior_anchor = _read_last_integrity_hash(log_path)
    archived = _archive_current_log(log_path, archive_dir)

    # 2. Retire the prior private key (if it still exists).
    retired_path: Path | None = None
    if DEFAULT_PRIVATE_KEY_PATH.exists():
        retired_path = DEFAULT_PRIVATE_KEY_PATH.with_suffix(
            f".retired-{datetime.now(timezone.utc).strftime('%Y%m%d')}.pem"
        )
        DEFAULT_PRIVATE_KEY_PATH.rename(retired_path)

    # 3. Bootstrap a fresh keypair. Artefacts-present guard is bypassed
    # because we just moved the active log out of the way; signed
    # pickles remain in results/models/ but those are not in the guard
    # set after the archive move (they are still on disk but will fail
    # verify under the new key — which is the intended behaviour).
    #
    # _artefacts_present scans live paths; after the archive move,
    # `audit_log.jsonl` is gone. `dataset_integrity.json` and the
    # `*.pkl.sig` files remain — those need explicit re-signing post
    # rotation, but their presence still trips the guard. So we call
    # _bootstrap_local_key directly with the guard relaxed by checking
    # presence ourselves and emitting a warning rather than raising.
    remaining = _artefacts_present()
    if remaining:
        logger.warning(
            "rotate_key: %d signed artefact(s) will be orphaned by this "
            "rotation: %s. Re-sign with tools.resign_models and re-run "
            "module0_analysis.bootstrap_integrity after the new pin is "
            "committed.",
            len(remaining), remaining,
        )

    # Direct EC generation here so we don't loop on the bootstrap guard.
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    import hashlib
    import os

    DEFAULT_PRIVATE_KEY_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(DEFAULT_PRIVATE_KEY_PATH.parent, 0o700)
    except OSError:
        pass
    pk = ec.generate_private_key(ec.SECP256R1())
    pem_priv = pk.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    pem_pub = pk.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    DEFAULT_PRIVATE_KEY_PATH.write_bytes(pem_priv)
    os.chmod(DEFAULT_PRIVATE_KEY_PATH, 0o600)
    DEFAULT_PUBLIC_KEY_PATH.write_bytes(pem_pub)
    new_key_id = "ecdsa-p256-" + hashlib.sha256(pem_pub).hexdigest()[:16]

    # 4. Open a fresh AuditLogger and write the rotation marker.
    # AuditLogger.__init__ now reads the new key (the pin file still
    # points at the OLD id — the operator updates it in the next step).
    # To allow this initial write the operator must NOT have committed
    # the new pin yet; we surface that constraint clearly.
    try:
        audit = AuditLogger(log_path)
    except Exception as exc:  # noqa: BLE001
        print(
            f"WARNING: opening AuditLogger raised {exc}. Likely the pin "
            f"in config/signing_key_id.txt still points at the OLD key id. "
            f"Update it to {new_key_id!r} and re-run this CLI; the new key "
            f"is already on disk at {DEFAULT_PRIVATE_KEY_PATH}.",
            file=sys.stderr,
        )
        return 3

    marker = {
        "event_type": event,
        "rotated_at": datetime.now(timezone.utc).isoformat(),
        "archived_chain": str(archived) if archived else None,
        "prior_chain_last_integrity_hash": prior_anchor,
        "cross_rotation_anchor": prior_anchor,
        "retired_private_key": str(retired_path) if retired_path else None,
        "new_key_id": new_key_id,
        "operator_ack": (
            "i-acknowledge-chain-break" if args.key_lost
            else "i-understand-this-orphans-old-signatures"
        ),
    }
    audit.log(marker)

    print(f"OK: rotated. New key id = {new_key_id}")
    print(
        "Next steps:\n"
        f"  1. Update config/signing_key_id.txt to {new_key_id!r}.\n"
        "  2. Commit the new pin alongside the public key in\n"
        "     results/reports/audit_signing_key.pub.pem.\n"
        "  3. Re-sign existing model pickles:\n"
        "       python -m tools.resign_models\n"
        "  4. Re-bootstrap the dataset integrity baseline:\n"
        "       python -m module0_analysis.bootstrap_integrity"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
