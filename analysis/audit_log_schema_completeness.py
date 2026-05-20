"""Validate every audit log entry against configs/audit_log_schema.yaml.

Hard strict per mode. Mode A and Mode B have different required field
sets inside the optional mve_audit block. Per-block conditional rules:

  - required_always (top-level sections)        — always enforced
  - required_when_signed (signature_envelope)   — when any signature
                                                  field is present
  - required_when_present (mve_audit, reviewer) — when the block is in
                                                  the record at all
  - required_when_mode_a_llm (mve_audit)        — when mve_audit.mve_mode
                                                  == "A_llm"

Output: results/rq3_audit_schema_audit.json
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
LOG_PATH = REPO_ROOT / "logs" / "llm_audit.jsonl"
SCHEMA_PATH = REPO_ROOT / "configs" / "audit_log_schema.yaml"
OUT = REPO_ROOT / "results" / "rq3_audit_schema_audit.json"

_TYPE_MAP: dict[str, Any] = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "float": float,
    "array": list,
    "object": dict,
    "boolean": bool,
    "null": type(None),
}


def _type_matches(value: Any, type_names: list[str] | str) -> bool:
    if not isinstance(type_names, list):
        type_names = [type_names]
    for tn in type_names:
        py = _TYPE_MAP.get(tn)
        if py is None:
            continue
        # Exclude bool when integer is required (bool is subclass of int).
        if tn == "integer" and isinstance(value, bool):
            continue
        if isinstance(value, py):
            return True
    return False


def _validate_field(spec: dict, value: Any) -> list[dict]:
    violations: list[dict] = []
    field = spec["field"]
    if not _type_matches(value, spec.get("type")):
        violations.append({
            "field": field, "kind": "type",
            "expected": spec.get("type"),
            "got": type(value).__name__,
        })
        return violations
    if "enum" in spec and value not in spec["enum"]:
        violations.append({
            "field": field, "kind": "enum",
            "expected": spec["enum"], "got": value,
        })
    if "range" in spec and isinstance(value, (int, float)):
        lo, hi = spec["range"]
        if not (lo <= value <= hi):
            violations.append({
                "field": field, "kind": "range",
                "expected": [lo, hi], "got": value,
            })
    if "length" in spec and isinstance(value, (str, list)):
        if len(value) != spec["length"]:
            violations.append({
                "field": field, "kind": "length",
                "expected": spec["length"], "got": len(value),
            })
    if "pattern" in spec and isinstance(value, str):
        if not re.match(spec["pattern"], value):
            violations.append({
                "field": field, "kind": "pattern",
                "expected": spec["pattern"], "got": value[:20] + "...",
            })
    return violations


def _validate_block(block: dict | None, specs: list[dict],
                    block_name: str) -> tuple[list[str], list[dict]]:
    """Validate a list of field specs against a sub-dict (or the
    top-level entry, when block is the entry itself).

    Returns (missing_field_names, violations).
    """
    missing: list[str] = []
    violations: list[dict] = []
    if block is None:
        return [s["field"] for s in specs], []
    for spec in specs:
        if spec["field"] not in block:
            missing.append(f"{block_name}.{spec['field']}")
            continue
        for v in _validate_field(spec, block[spec["field"]]):
            v["field"] = f"{block_name}.{v['field']}"
            violations.append(v)
    return missing, violations


def _validate_entry(entry: dict, schema: dict) -> dict:
    sections = schema.get("sections") or {}
    missing: list[str] = []
    violations: list[dict] = []

    # alert_context — flat fields on the entry itself
    for spec in sections.get("alert_context", {}).get("required_always") or []:
        if spec["field"] not in entry:
            missing.append(f"alert_context.{spec['field']}")
            continue
        for v in _validate_field(spec, entry[spec["field"]]):
            v["field"] = f"alert_context.{v['field']}"
            violations.append(v)

    # decision_capture — flat on entry
    for spec in (sections.get("decision_capture", {}).get("required_always")
                 or []):
        if spec["field"] not in entry:
            missing.append(f"decision_capture.{spec['field']}")
            continue
        for v in _validate_field(spec, entry[spec["field"]]):
            v["field"] = f"decision_capture.{v['field']}"
            violations.append(v)

    # forward_compat — flat on entry
    for spec in (sections.get("forward_compat", {}).get("required_always")
                 or []):
        if spec["field"] not in entry:
            missing.append(f"forward_compat.{spec['field']}")
            continue
        for v in _validate_field(spec, entry[spec["field"]]):
            v["field"] = f"forward_compat.{v['field']}"
            violations.append(v)

    # tamper_evidence — flat on entry
    for spec in (sections.get("tamper_evidence", {}).get("required_always")
                 or []):
        if spec["field"] not in entry:
            missing.append(f"tamper_evidence.{spec['field']}")
            continue
        for v in _validate_field(spec, entry[spec["field"]]):
            v["field"] = f"tamper_evidence.{v['field']}"
            violations.append(v)

    # signature_envelope — required when ANY signature field present
    sig_block_keys = ("signature", "signing_key_id", "signature_alg")
    any_sig_present = any(k in entry for k in sig_block_keys)
    if any_sig_present:
        sig_specs = (sections.get("signature_envelope", {})
                     .get("required_when_signed") or [])
        for spec in sig_specs:
            if spec["field"] not in entry:
                missing.append(f"signature_envelope.{spec['field']}")
                continue
            for v in _validate_field(spec, entry[spec["field"]]):
                v["field"] = f"signature_envelope.{v['field']}"
                violations.append(v)

    # mve_audit — nested block, optional. When present, validate.
    mve = entry.get("mve_audit")
    if mve is not None:
        if not isinstance(mve, dict):
            violations.append({"field": "mve_audit", "kind": "type",
                               "expected": "object",
                               "got": type(mve).__name__})
        else:
            base = (sections.get("mve_audit", {})
                    .get("required_when_present") or [])
            m_miss, m_vio = _validate_block(mve, base, "mve_audit")
            missing.extend(m_miss)
            violations.extend(m_vio)
            if mve.get("mve_mode") == "A_llm":
                llm = (sections.get("mve_audit", {})
                       .get("required_when_mode_a_llm") or [])
                m2, v2 = _validate_block(mve, llm, "mve_audit")
                missing.extend(m2)
                violations.extend(v2)

    # reviewer — nested block, optional.
    rev = entry.get("reviewer")
    if rev is not None:
        if not isinstance(rev, dict):
            violations.append({"field": "reviewer", "kind": "type",
                               "expected": "object",
                               "got": type(rev).__name__})
        else:
            rspecs = (sections.get("reviewer", {})
                      .get("required_when_present") or [])
            r_miss, r_vio = _validate_block(rev, rspecs, "reviewer")
            missing.extend(r_miss)
            violations.extend(r_vio)

    return {
        "is_valid": not missing and not violations,
        "missing_required_fields": missing,
        "violations": violations,
        "mode": (mve or {}).get("mve_mode") if isinstance(mve, dict) else None,
    }


def main() -> None:
    schema = yaml.safe_load(SCHEMA_PATH.read_text())
    meta = {
        "schema_version": schema.get("schema_version", "1.0"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generated_by": "analysis/audit_log_schema_completeness.py",
        "log_path": str(LOG_PATH.relative_to(REPO_ROOT)),
        "schema_path": str(SCHEMA_PATH.relative_to(REPO_ROOT)),
        "taxonomy_locked_on": schema.get("taxonomy_locked_on"),
    }

    if not LOG_PATH.exists() or LOG_PATH.stat().st_size == 0:
        result = {
            "_meta": meta,
            "headline": {
                "all_entries_pass_schema": None,
                "n_entries_validated": 0,
                "n_entries_failing": 0,
                "_note": "Audit log missing or empty — nothing to validate.",
            },
            "failures": [],
        }
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(result, indent=2, default=str))
        print(f"Wrote {OUT.relative_to(REPO_ROOT)} (no-op)")
        return

    failures: list[dict] = []
    by_mode: dict[str, dict[str, int]] = {
        "A_llm": {"n_validated": 0, "n_failing": 0},
        "B_rule": {"n_validated": 0, "n_failing": 0},
        "_no_mve_audit": {"n_validated": 0, "n_failing": 0},
    }

    with LOG_PATH.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                entry = json.loads(raw)
            except json.JSONDecodeError as exc:
                failures.append({
                    "line_number": line_no,
                    "_status": "json_parse_error",
                    "error": str(exc),
                })
                continue

            verdict = _validate_entry(entry, schema)
            mode_key = verdict["mode"] or "_no_mve_audit"
            by_mode.setdefault(mode_key, {"n_validated": 0, "n_failing": 0})
            by_mode[mode_key]["n_validated"] += 1
            if not verdict["is_valid"]:
                by_mode[mode_key]["n_failing"] += 1
                failures.append({
                    "line_number": line_no,
                    "alert_id": entry.get("alert_id"),
                    "mode": verdict["mode"],
                    "missing_required_fields": verdict["missing_required_fields"],
                    "violations": verdict["violations"],
                })

    n_total = sum(b["n_validated"] for b in by_mode.values())
    n_failing = len(failures)

    result = {
        "_meta": meta,
        "headline": {
            "all_entries_pass_schema": n_failing == 0,
            "n_entries_validated": n_total,
            "n_entries_failing": n_failing,
            "by_mode": by_mode,
        },
        "failures": failures[:50],
        "failures_truncated_at": 50,
        "failures_total_count": n_failing,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, default=str))
    print(f"Wrote {OUT.relative_to(REPO_ROOT)}")
    if n_failing == 0:
        print(f"Schema audit: PASS ({n_total} entries)")
    else:
        print(f"Schema audit: FAIL ({n_failing}/{n_total} entries violate schema)")


if __name__ == "__main__":
    main()
