"""One-time migration: seal the pre-Sprint-3 audit log into an archive.

Sprint 3 flipped ``verify_audit_log`` and ``AuditLogger.__init__`` to
``legacy_ok=False`` (tier 1 F1). Any active log carrying unsigned
records or pre-hardening chain restarts will now fail
``verify_on_open`` and break every subsequent process. This CLI is the
sanctioned one-shot migration:

  1. Walk the active log under ``legacy_ok=True`` (the pre-Sprint-3
     semantics) and report breaks. A break aborts the seal — operators
     must investigate before sealing a tampered legacy log.
  2. Move the file into ``audit_archive/`` with a timestamp suffix and
     drop a sealed manifest that captures the first / last integrity
     hashes + the legacy-restart count.
  3. The next ``AuditLogger`` instantiation opens a fresh file whose
     genesis ``AUDIT_LOG_MIGRATED`` record carries
     ``cross_rotation_anchor`` pointing at the archived chain's last
     integrity hash, so a forensic walker can trace archive →
     post-migration without trusting filesystem ordering.

Usage::

    python -m module5_responses.audit.seal_legacy

    python -m module5_responses.audit.seal_legacy --dry-run
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


def _read_last_integrity_hash(path: Path) -> str | None:
    if not path.exists() or path.stat().st_size == 0:
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
        prog="python -m module5_responses.audit.seal_legacy",
        description=(
            "One-time archive of the pre-Sprint-3 audit log so the new "
            "legacy_ok=False default does not lock the system out."
        ),
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Inspect without touching the file.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from module5_responses.audit.signing import OUTPUT_DIR
    from module5_responses.audit.verify import verify_audit_log
    from module5_responses.audit.logger import AuditLogger

    log_path = OUTPUT_DIR / "audit_log.jsonl"
    archive_dir = OUTPUT_DIR / "audit_archive"

    if not log_path.exists() or log_path.stat().st_size == 0:
        print("OK: no active audit_log.jsonl to seal.")
        return 0

    # 1. Walk under the legacy default to surface any tamper.
    report = verify_audit_log(log_path, legacy_ok=True)
    print(
        f"Pre-seal verification: total={report['total']}, "
        f"valid_signed={report['valid_signed']}, "
        f"valid_legacy={report['valid_legacy']}, "
        f"legacy_chain_restarts={report.get('legacy_chain_restarts', 0)}, "
        f"first_break_at={report['first_break_at']}"
    )
    if report["first_break_at"] is not None:
        print(
            "REFUSED: the legacy log appears broken before line "
            f"{report['first_break_at']}. Investigate or restore from "
            "backup before sealing.",
            file=sys.stderr,
        )
        return 2

    last_anchor = _read_last_integrity_hash(log_path)

    if args.dry_run:
        print(f"DRY RUN: would archive {log_path} (anchor={last_anchor[:16] if last_anchor else '-'}…).")
        return 0

    # 2. Move the file into the archive with manifest.
    archive_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archived = archive_dir / f"{log_path.stem}.{stamp}.legacy-seal.jsonl"
    log_path.rename(archived)
    manifest = {
        "archived_path": str(archived),
        "sealed_at": datetime.now(timezone.utc).isoformat(),
        "reason": "Sprint 3 legacy_ok=False migration",
        "verifier_summary": {
            "total": report["total"],
            "valid_signed": report["valid_signed"],
            "valid_legacy": report["valid_legacy"],
            "legacy_chain_restarts": report.get("legacy_chain_restarts", 0),
        },
        "last_integrity_hash": last_anchor,
    }
    (archived.with_suffix(".manifest.json")).write_text(
        json.dumps(manifest, indent=2), encoding="utf-8",
    )

    # 3. Open a fresh AuditLogger (verify_on_open passes — file is gone)
    # and write the migration marker carrying cross_rotation_anchor.
    audit = AuditLogger(log_path)
    audit.log(
        {
            "event_type": "AUDIT_LOG_MIGRATED",
            "subtype": "sprint3_legacy_seal",
            "archived_path": str(archived),
            "archived_n_records": report["total"],
            "cross_rotation_anchor": last_anchor,
            "sealed_at": manifest["sealed_at"],
        }
    )

    print(
        f"OK: sealed {report['total']} legacy records into {archived.name}. "
        f"Fresh chain anchored at {last_anchor[:16] + '…' if last_anchor else 'genesis'}."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
