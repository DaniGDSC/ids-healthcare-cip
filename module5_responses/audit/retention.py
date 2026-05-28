"""Audit-log retention + rotation."""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .logger import ARCHIVE_DIR
from .verify import verify_audit_log


def _record_ts(rec: dict, mtime_fallback: Path) -> datetime:
    """Best-effort timestamp parse with mtime fallback for legacy records."""
    for key in ("timestamp", "review_timestamp"):
        v = rec.get(key)
        if isinstance(v, str):
            try:
                return datetime.fromisoformat(v.replace("Z", "+00:00"))
            except ValueError:
                pass
    r = rec.get("reviewer", {})
    if isinstance(r, dict):
        v = r.get("review_timestamp")
        if isinstance(v, str):
            try:
                return datetime.fromisoformat(v.replace("Z", "+00:00"))
            except ValueError:
                pass
    return datetime.fromtimestamp(mtime_fallback.stat().st_mtime, tz=timezone.utc)


def rotate_and_purge(
    audit,
    retention_days: int | None = None,
    archive_dir: Path | None = None,
) -> dict:
    """Archive ``audit.path`` if oldest record is past the retention cutoff.

    The new active log starts at genesis (``prev_hash="0"*64``); the
    cross-rotation forensic link is preserved via the sealed manifest
    sidecar + the signed ``AUDIT_LOG_ROTATED`` marker that the caller
    writes after a successful archive.
    """
    days = retention_days if retention_days is not None else audit.retention_days
    archive_dir = Path(archive_dir or ARCHIVE_DIR)
    archive_dir.mkdir(parents=True, exist_ok=True)

    report: dict = {
        "rotated": False,
        "reason": None,
        "archived_path": None,
        "manifest_path": None,
        "retention_days": days,
        "verify_before_rotate": None,
    }

    if not audit.path.exists() or audit.path.stat().st_size == 0:
        report["reason"] = "active log empty or missing"
        return report

    # Tier 1 F1: post-Sprint-3 default is legacy_ok=False. If the active
    # log has been carrying legacy unsigned records we expect the
    # operator to have already sealed it via the rotate_key CLI before
    # invoking rotate_and_purge.
    verify_report = verify_audit_log(audit.path, audit.public_key_path, legacy_ok=False)
    report["verify_before_rotate"] = {
        "total": verify_report["total"],
        "valid_signed": verify_report["valid_signed"],
        "valid_legacy": verify_report["valid_legacy"],
        "first_break_at": verify_report["first_break_at"],
    }
    if verify_report["first_break_at"] is not None:
        report["reason"] = (
            f"refusing to rotate a tampered log (first break at "
            f"line {verify_report['first_break_at']})"
        )
        audit.log(
            {
                "event_type": "SECURITY_INCIDENT",
                "subtype": "rotate_refused_chain_broken",
                "first_break_at": verify_report["first_break_at"],
                "broken_count": len(verify_report["broken"]),
            }
        )
        return report

    # Tier 1 F1 follow-up: legacy_chain_restarts > 0 was an informational
    # counter pre-Sprint 3. Post-flip it must be 0 — any non-zero count
    # is a chain-restart marker that should not exist in the post-
    # migration log.
    if verify_report.get("legacy_chain_restarts", 0) > 0:
        report["reason"] = (
            f"refusing to rotate a log with "
            f"{verify_report['legacy_chain_restarts']} legacy chain "
            f"restart(s); seal via rotate_key CLI first"
        )
        audit.log(
            {
                "event_type": "SECURITY_INCIDENT",
                "subtype": "rotate_refused_legacy_restart_present",
                "legacy_chain_restarts": verify_report["legacy_chain_restarts"],
            }
        )
        return report

    first_record = None
    last_record = None
    with open(audit.path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            rec = json.loads(line)
            if first_record is None:
                first_record = rec
            last_record = rec

    if first_record is None or last_record is None:
        report["reason"] = "active log has no parseable records"
        return report

    first_ts = _record_ts(first_record, audit.path)
    last_ts = _record_ts(last_record, audit.path)
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    if first_ts.tzinfo is None:
        first_ts = first_ts.replace(tzinfo=timezone.utc)
    if last_ts.tzinfo is None:
        last_ts = last_ts.replace(tzinfo=timezone.utc)

    if first_ts >= cutoff:
        report["reason"] = (
            f"oldest record ({first_ts.isoformat()}) is within "
            f"the {days}-day retention window; nothing to rotate"
        )
        return report

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archived_path = archive_dir / f"{audit.path.stem}.{stamp}.jsonl"
    shutil.move(str(audit.path), str(archived_path))

    manifest = {
        "archived_path": str(archived_path),
        "first_record_ts": first_ts.isoformat(),
        "last_record_ts": last_ts.isoformat(),
        "n_records": verify_report["total"],
        "first_integrity_hash": first_record.get("integrity_hash"),
        "last_integrity_hash": last_record.get("integrity_hash"),
        "signing_key_id": last_record.get("signing_key_id"),
        "sealed_at": datetime.now(timezone.utc).isoformat(),
        "verifier_summary": report["verify_before_rotate"],
    }
    manifest_path = archived_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Tier 1 F2: write a cross_rotation_anchor that binds the new
    # chain's genesis to the archived chain's last integrity_hash.
    # Forensic walkers can follow archives → active chain without
    # trusting filesystem ordering: the anchor field in the first
    # record points at the archive that immediately preceded it.
    archived_last_integrity_hash = last_record.get("integrity_hash")
    audit.prev_hash = "0" * 64
    audit.log(
        {
            "event_type": "AUDIT_LOG_ROTATED",
            "archived_path": str(archived_path),
            "archived_first_ts": first_ts.isoformat(),
            "archived_last_ts": last_ts.isoformat(),
            "archived_n_records": verify_report["total"],
            "archived_last_integrity_hash": archived_last_integrity_hash,
            "archived_first_integrity_hash": first_record.get("integrity_hash"),
            "cross_rotation_anchor": archived_last_integrity_hash,
            "manifest_path": str(manifest_path),
            "retention_days": days,
            "rotated_at": datetime.now(timezone.utc).isoformat(),
        }
    )

    report["rotated"] = True
    report["archived_path"] = str(archived_path)
    report["manifest_path"] = str(manifest_path)
    report["reason"] = (
        f"archived {verify_report['total']} records spanning "
        f"{first_ts.isoformat()} → {last_ts.isoformat()}"
    )
    return report


__all__ = ["rotate_and_purge"]
