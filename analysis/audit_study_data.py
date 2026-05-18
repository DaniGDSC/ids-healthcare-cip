"""Audit RQ2.c LLM-persona user-study data: schema validation + error-rate audit.

Data shape (LLM-persona simulation, not human study):
  survey/study_responses_*.json = {persona_id, n_alerts, rows:[{alert_id,
      condition, correct_action, response:{action, severity_assessment,
      confidence, rationale}, error}]}

Exclusion criteria (Path C — LLM-persona variant):
  EX-3 Persona-level schema invalid (missing required fields, mixed condition,
       or all rows failed with API errors).

Attention-check and duration-based exclusions (spec EX-1, EX-2) do not apply
to LLM-persona data; documented as a known limitation rather than enforced.

Writes:
  survey/study_data_audit.json     full audit
  survey/rq2c_exclusions.json      excluded participants with reasons
"""
from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SURVEY_DIR = REPO_ROOT / "survey"
AUDIT_OUT = SURVEY_DIR / "study_data_audit.json"
EXCL_OUT = SURVEY_DIR / "rq2c_exclusions.json"

REQUIRED_TOP_LEVEL = ["persona_id", "n_alerts", "rows"]
REQUIRED_PER_ROW = ["alert_id", "condition", "correct_action"]
REQUIRED_RESPONSE_FIELDS = ["action", "confidence"]
VALID_ROLES = {"biomed_engineer", "IT_generalist", "nurse_manager"}
VALID_CONDITIONS = {"A", "B"}


def _role_from_pid(pid: str) -> str:
    """Strip a trailing ``_P\\d+`` suffix to recover the role."""
    parts = pid.split("_")
    if parts and re.match(r"^P\d+$", parts[-1]):
        parts = parts[:-1]
    return "_".join(parts) if parts else pid


def _validate_persona(record: dict) -> tuple[list[dict], str | None, str | None]:
    errors: list[dict] = []
    for key in REQUIRED_TOP_LEVEL:
        if key not in record:
            errors.append({"check": "missing_top_level", "field": key})

    pid = record.get("persona_id")
    role = _role_from_pid(pid) if isinstance(pid, str) else None
    if role not in VALID_ROLES:
        errors.append({"check": "unknown_role", "value": role})

    rows = record.get("rows", [])
    if not isinstance(rows, list) or not rows:
        errors.append({"check": "rows_empty_or_invalid"})
        return errors, role, None

    conditions_seen: set[str] = set()
    for i, r in enumerate(rows):
        for k in REQUIRED_PER_ROW:
            if k not in r:
                errors.append({"check": "missing_row_field",
                               "row_index": i, "field": k})
                break
        cond = r.get("condition")
        if cond in VALID_CONDITIONS:
            conditions_seen.add(cond)
        else:
            errors.append({"check": "invalid_condition",
                           "row_index": i, "value": cond})
        resp = r.get("response")
        # response may be None when error is set — only validate the shape
        # when the row succeeded.
        if isinstance(resp, dict):
            for k in REQUIRED_RESPONSE_FIELDS:
                if k not in resp:
                    errors.append({"check": "missing_response_field",
                                   "row_index": i, "field": k})
                    break

    if len(conditions_seen) > 1:
        errors.append({"check": "mixed_condition_within_persona",
                       "values": sorted(conditions_seen)})

    condition = next(iter(conditions_seen)) if len(conditions_seen) == 1 else None
    return errors, role, condition


def _row_success_stats(record: dict) -> dict:
    rows = record.get("rows", [])
    n_total = len(rows)
    n_error = sum(1 for r in rows if r.get("error"))
    n_success = sum(1 for r in rows
                    if r.get("error") is None and isinstance(r.get("response"), dict))
    return {
        "n_rows": n_total,
        "n_success": n_success,
        "n_error": n_error,
        "success_rate": round(n_success / n_total, 4) if n_total else 0.0,
    }


def _summarize_field(records: list[dict], field: str) -> dict[str, int]:
    counts: Counter = Counter()
    for r in records:
        if r.get("excluded"):
            continue
        counts[str(r.get(field) or "UNKNOWN")] += 1
    return dict(counts)


def _summarize_role_x_condition(records: list[dict]) -> dict[str, int]:
    counts: Counter = Counter()
    for r in records:
        if r.get("excluded"):
            continue
        counts[f"{r.get('role') or 'UNKNOWN'}|{r.get('condition') or 'UNKNOWN'}"] += 1
    return dict(counts)


def main() -> None:
    response_files = sorted(SURVEY_DIR.glob("study_responses_*.json"))
    audit_records: list[dict] = []
    exclusions: list[dict] = []

    for path in response_files:
        try:
            rec = json.loads(path.read_text())
        except json.JSONDecodeError as e:
            audit_records.append({
                "file": path.name,
                "persona_id": path.stem,
                "excluded": True,
                "exclusion_reasons": ["EX-3 schema invalid (JSON parse error)"],
                "schema_errors": [{"check": "json_parse_error", "msg": str(e)}],
            })
            exclusions.append({
                "persona_id": path.stem,
                "file": path.name,
                "reasons": ["EX-3 schema invalid (JSON parse error)"],
            })
            continue

        errors, role, condition = _validate_persona(rec)
        stats = _row_success_stats(rec)

        # A persona with zero successful rows can't contribute to metrics.
        zero_success = stats["n_rows"] > 0 and stats["n_success"] == 0
        excluded = bool(errors) or zero_success
        reasons: list[str] = []
        if errors:
            reasons.append("EX-3 schema invalid")
        if zero_success:
            reasons.append("EX-3 zero successful rows (all API errors)")

        audit_records.append({
            "file": path.name,
            "persona_id": rec.get("persona_id"),
            "role": role,
            "condition": condition,
            "n_rows": stats["n_rows"],
            "n_success": stats["n_success"],
            "n_error": stats["n_error"],
            "success_rate": stats["success_rate"],
            "schema_errors": errors,
            "excluded": excluded,
            "exclusion_reasons": reasons,
        })

        if excluded:
            exclusions.append({
                "persona_id": rec.get("persona_id", path.stem),
                "file": path.name,
                "reasons": reasons,
            })

    n_included = len(response_files) - len(exclusions)
    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generated_by": "analysis/audit_study_data.py",
        "data_source": "LLM-persona simulation (gpt-4o-mini); not human study",
        "n_participant_files": len(response_files),
        "n_excluded": len(exclusions),
        "exclusion_rules": {
            "EX-3": "Schema validation failed OR zero successful rows",
            "EX-1_attention_check": "N/A — LLM personas have no attention check",
            "EX-2_duration_outlier": "N/A — LLM personas have no response timing",
        },
    }

    audit = {
        "_meta": meta,
        "summary": {
            "n_total": len(response_files),
            "n_included": n_included,
            "n_excluded": len(exclusions),
            "exclusion_rate": round(len(exclusions) / len(response_files), 4)
                              if response_files else 0.0,
            "by_role": _summarize_field(audit_records, "role"),
            "by_condition": _summarize_field(audit_records, "condition"),
            "by_role_x_condition": _summarize_role_x_condition(audit_records),
        },
        "audit_records": audit_records,
    }

    AUDIT_OUT.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_OUT.write_text(json.dumps(audit, indent=2, default=str))
    EXCL_OUT.write_text(json.dumps({"_meta": meta, "exclusions": exclusions},
                                   indent=2, default=str))

    print(f"Wrote {AUDIT_OUT.relative_to(REPO_ROOT)}")
    print(f"Wrote {EXCL_OUT.relative_to(REPO_ROOT)}")
    print(f"Total personas: {len(response_files)}  "
          f"Included: {n_included}  Excluded: {len(exclusions)}")


if __name__ == "__main__":
    main()
