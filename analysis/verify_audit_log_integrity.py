"""Verify the SHA256 hash chain of the audit log.

Detects ALL chain breaks (not just the first, unlike the writer
module's AuditLogger.verify which short-circuits). For each break,
records ±3-entry forensic context so a reviewer can identify what
was tampered with.

Output: results/rq3_audit_chain_verification.json
Exit code 0 always (CI test interprets the JSON).
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from common.audit_canonicalization import (
    GENESIS_PREV_HASH,
    INTEGRITY_FIELD,
    verify_entry_hash,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG = REPO_ROOT / "logs" / "llm_audit.jsonl"
OUT = REPO_ROOT / "results" / "rq3_audit_chain_verification.json"

FORENSIC_CONTEXT_WIDTH = 3


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _entry_summary(entry: dict, line_no: int) -> dict:
    h = entry.get(INTEGRITY_FIELD) or ""
    return {
        "line": line_no,
        "alert_id": entry.get("alert_id"),
        "timestamp": entry.get("timestamp"),
        "integrity_hash_prefix": (h[:16] + "...") if h else None,
    }


def _verify_one_log(log_path: Path) -> dict:
    out_meta: dict[str, Any] = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generated_by": "analysis/verify_audit_log_integrity.py",
        "log_path": str(log_path.relative_to(REPO_ROOT))
                    if log_path.is_relative_to(REPO_ROOT) else str(log_path),
        "canonicalization_module": "common.audit_canonicalization",
        "hash_algorithm": "SHA256",
        "genesis_prev_hash": GENESIS_PREV_HASH,
        "wire_format_note": (
            "Hash chain uses prev_hash + integrity_hash field names (real "
            "wire format of module5_responses.module5_pipeline.AuditLogger), "
            "not the previous_hash + entry_hash names from the spec."
        ),
    }

    if not log_path.exists() or log_path.stat().st_size == 0:
        return {
            "_meta": out_meta,
            "headline": {
                "chain_intact": None,
                "n_entries": 0,
                "n_breaks": 0,
                "verification_complete": False,
                "_note": "Audit log missing or empty — nothing to verify.",
            },
            "breaks": [],
            "parse_errors": [],
            "segments": [],
        }

    out_meta["log_sha256"] = _file_sha256(log_path)

    entries: list[dict] = []
    parse_errors: list[dict] = []
    with log_path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                entries.append(json.loads(raw))
            except json.JSONDecodeError as exc:
                parse_errors.append({
                    "line_number": line_no,
                    "error": str(exc),
                    "raw_excerpt": raw[:120],
                })

    breaks: list[dict] = []
    segments: list[dict] = [{"start_line": 1, "n_entries": 0}]
    expected_prev = GENESIS_PREV_HASH

    for idx, entry in enumerate(entries):
        line_no = idx + 1
        result = verify_entry_hash(entry, expected_prev)
        if result["is_valid"]:
            segments[-1]["n_entries"] += 1
            segments[-1]["end_line"] = line_no
            expected_prev = result["stored_integrity_hash"]
        else:
            preceding = [_entry_summary(entries[j], j + 1)
                         for j in range(max(0, idx - FORENSIC_CONTEXT_WIDTH), idx)]
            following = [_entry_summary(entries[j], j + 1)
                         for j in range(idx + 1,
                                        min(len(entries),
                                            idx + 1 + FORENSIC_CONTEXT_WIDTH))]
            breaks.append({
                "entry_line_number": line_no,
                "entry_alert_id": entry.get("alert_id"),
                "reason": result["failure_reason"],
                "stored_integrity_hash": result["stored_integrity_hash"],
                "computed_integrity_hash": result["computed_integrity_hash"],
                "stored_prev_hash": result["stored_prev_hash"],
                "expected_prev_hash": result["expected_prev_hash"],
                "forensic_context": {
                    "preceding_entries": preceding,
                    "broken_entry": _entry_summary(entry, line_no),
                    "following_entries": following,
                },
            })
            # Close the intact segment and start a new one after the break.
            if segments[-1]["n_entries"] > 0:
                segments.append({"start_line": line_no + 1, "n_entries": 0})
            else:
                segments[-1]["start_line"] = line_no + 1
            # Continue forensics: pivot expected_prev to broken entry's
            # stored integrity_hash so we can detect subsequent breaks.
            expected_prev = entry.get(INTEGRITY_FIELD) or expected_prev

    # Drop trailing empty segment if present.
    segments = [s for s in segments if s["n_entries"] > 0]

    out_meta["n_entries_scanned"] = len(entries)
    headline = {
        "chain_intact": len(breaks) == 0 and len(parse_errors) == 0,
        "n_entries": len(entries),
        "n_breaks": len(breaks),
        "n_parse_errors": len(parse_errors),
        "first_entry_timestamp": (entries[0].get("timestamp") if entries
                                  else None),
        "last_entry_timestamp": (entries[-1].get("timestamp") if entries
                                 else None),
        "verification_complete": True,
        "tamper_evidence_claim": (
            "tamper-evident (detection); not tamper-resistant (prevention)"
        ),
    }
    return {
        "_meta": out_meta,
        "headline": headline,
        "breaks": breaks,
        "parse_errors": parse_errors,
        "segments": segments,
    }


def main(log_path: Optional[Path] = None) -> None:
    log_path = log_path or DEFAULT_LOG
    result = _verify_one_log(log_path)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, default=str))
    h = result["headline"]
    print(f"Wrote {OUT.relative_to(REPO_ROOT)}")
    if h["chain_intact"] is None:
        print(f"Chain: NO-OP ({h.get('_note', 'no log to verify')})")
    elif h["chain_intact"]:
        print(f"Chain: INTACT ({h['n_entries']} entries, 0 breaks)")
    else:
        print(f"Chain: BROKEN ({h['n_entries']} entries, "
              f"{h['n_breaks']} break(s), "
              f"{h['n_parse_errors']} parse error(s))")


if __name__ == "__main__":
    main()
