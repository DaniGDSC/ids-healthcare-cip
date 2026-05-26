"""One-time migration: v2 (path-keyed) → v3 (sha256-keyed) integrity baseline.

Why
---
v2 keyed entries by absolute filesystem path:
    "/home/un1/.../dataset.csv": {"sha256": "abc...", ...}
    "/home/dev/.../dataset.csv": {"sha256": "abc...", ...}

This drifted: each developer who fresh-bootstrapped created a new entry
on their host. The committed baseline accumulated hostname-specific
paths that revealed dev usernames and bloated the file.

v3 keys by SHA-256 digest:
    "abc...": {"filename": "dataset.csv", "size_bytes": ..., ...}

One physical file → one entry, regardless of where it lives on disk.

Usage
-----
    python -m module0_analysis.migrate_v2_to_v3 \\
        [--in module0_analysis/dataset_integrity.json] \\
        [--out module0_analysis/dataset_integrity.json] \\
        [--keep-backup]

What this does
--------------
1. Read existing baseline; refuse if not v2.
2. Verify the existing v2 signature (refuse to migrate forged metadata).
3. Collapse path-keyed entries by SHA-256: identical hash across paths
   becomes one entry. If two paths claim same hash but different size,
   fail loudly (corruption indicator).
4. Pick filename from the first entry of each hash group.
5. Re-sign the v3 body with the current Module 5 ECDSA P-256 key.
6. Atomic write to *out*; rename old to ``.v2.bak`` if --keep-backup.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from module5_responses.signing import (
    HAVE_CRYPTOGRAPHY,
    canonical_json,
    load_signing_key,
)
from .security import IntegrityError, _METADATA_VERSION

_DEFAULT_PATH = Path(__file__).resolve().parent / "dataset_integrity.json"


def _verify_v2_signature(meta: dict) -> None:
    """Refuse to migrate a forged or unsigned v2 baseline."""
    if not HAVE_CRYPTOGRAPHY:
        raise IntegrityError(
            "cryptography package missing — cannot verify v2 signature."
        )
    sig_b64 = meta.get("signature")
    if not sig_b64:
        raise IntegrityError("v2 metadata is unsigned. Refusing to migrate.")

    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec

    _, public_path, _ = load_signing_key()
    public_key = serialization.load_pem_public_key(public_path.read_bytes())
    payload = canonical_json({"version": meta["version"], "entries": meta["entries"]})
    try:
        public_key.verify(base64.b64decode(sig_b64), payload, ec.ECDSA(hashes.SHA256()))
    except InvalidSignature as exc:
        raise IntegrityError(
            "v2 metadata signature is invalid — file has been tampered with. "
            "Refusing to migrate. Investigate before bootstrapping a fresh v3."
        ) from exc


def _collapse_to_v3(v2_entries: dict[str, dict]) -> dict[str, dict]:
    """Group v2 path-keyed entries by SHA-256 digest into v3 records."""
    by_hash: dict[str, dict] = {}
    for path_str, rec in v2_entries.items():
        digest = rec.get("sha256")
        size = rec.get("size_bytes")
        if not digest or size is None:
            raise IntegrityError(
                f"v2 entry for {path_str!r} missing sha256/size_bytes — "
                f"refusing to migrate a corrupt baseline."
            )
        existing = by_hash.get(digest)
        if existing is None:
            by_hash[digest] = {
                "filename": Path(path_str).name,
                "size_bytes": size,
                "bootstrapped_at": rec.get(
                    "bootstrapped_at", datetime.now(timezone.utc).isoformat()
                ),
                "migrated_from_v2_paths": [path_str],
            }
        else:
            if existing["size_bytes"] != size:
                raise IntegrityError(
                    f"v2 has two entries claiming the same SHA-256 ({digest[:16]}…) "
                    f"with different sizes ({existing['size_bytes']} vs {size}). "
                    f"Corrupt baseline — refusing to migrate."
                )
            existing["migrated_from_v2_paths"].append(path_str)
    return by_hash


def _sign_and_write(body: dict, out_path: Path) -> None:
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import ec

    private_key, _, signing_key_id = load_signing_key()
    signature = private_key.sign(canonical_json(body), ec.ECDSA(hashes.SHA256()))
    body["signature"] = base64.b64encode(signature).decode("ascii")
    body["signing_key_id"] = signing_key_id
    body["signature_alg"] = "ECDSA_P256_SHA256"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(body, indent=2))
    tmp.replace(out_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m module0_analysis.migrate_v2_to_v3",
        description=(
            "Migrate Phase 0 integrity baseline from v2 (path-keyed) to "
            "v3 (sha256-keyed). One-time; idempotent — refuses to run "
            "on already-v3 baselines."
        ),
    )
    parser.add_argument("--in", dest="in_path", type=Path, default=_DEFAULT_PATH)
    parser.add_argument("--out", dest="out_path", type=Path, default=_DEFAULT_PATH)
    parser.add_argument(
        "--keep-backup",
        action="store_true",
        help="Rename original to dataset_integrity.json.v2.bak",
    )
    args = parser.parse_args(argv)

    if not args.in_path.exists():
        print(f"REFUSED: input file not found: {args.in_path}", file=sys.stderr)
        return 2

    try:
        meta = json.loads(args.in_path.read_text())
    except json.JSONDecodeError as exc:
        print(f"REFUSED: input is not valid JSON: {exc}", file=sys.stderr)
        return 2

    version = meta.get("version")
    if version == _METADATA_VERSION:
        print(f"Already v{_METADATA_VERSION}; nothing to migrate.", file=sys.stderr)
        return 0
    if version != 2:
        print(
            f"REFUSED: unsupported source version {version!r} "
            f"(this script migrates v2 → v{_METADATA_VERSION} only).",
            file=sys.stderr,
        )
        return 2

    try:
        _verify_v2_signature(meta)
    except IntegrityError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 3

    try:
        v3_entries = _collapse_to_v3(meta["entries"])
    except IntegrityError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 4

    v3_body: dict = {"version": _METADATA_VERSION, "entries": v3_entries}

    if args.keep_backup:
        backup = args.in_path.with_suffix(".json.v2.bak")
        args.in_path.rename(backup)
        print(f"  → backup: {backup}")

    _sign_and_write(v3_body, args.out_path)

    print(
        f"OK: migrated {len(meta['entries'])} v2 entries "
        f"→ {len(v3_entries)} v{_METADATA_VERSION} entries"
    )
    for digest, rec in v3_entries.items():
        paths_summary = ", ".join(rec.get("migrated_from_v2_paths", []))
        print(
            f"  sha256={digest[:16]}…  filename={rec['filename']}  paths=[{paths_summary}]"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
