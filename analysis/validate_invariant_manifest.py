"""Validate configs/invariants_manifest.yaml against 10 structural rules.

Output: results/rq3_invariant_manifest_validation.json
Exit code 1 if any check fails (CI-blocking).
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "configs" / "invariants_manifest.yaml"
OUT = REPO_ROOT / "results" / "rq3_invariant_manifest_validation.json"

VALID_SEVERITY = {"safety_critical", "quality"}
VALID_STATUS = {"enforced", "pending", "documented"}
VALID_RQS = {1, 2, 3}
VALID_VERIFICATION = {"pytest", "grep", "grep_and_pytest"}


def _add(findings: list, check_id: str, ok: bool, desc: str, **details) -> None:
    findings.append({
        "check_id": check_id,
        "severity": "PASS" if ok else "FAIL",
        "description": desc,
        "details": details or None,
    })


def main() -> None:
    findings: list = []

    # V1: parse
    try:
        doc = yaml.safe_load(MANIFEST.read_text())
        _add(findings, "V1", True, "Manifest parsed")
    except Exception as e:
        _add(findings, "V1", False, "Manifest failed to parse", error=str(e))
        _finalize(findings, [])
        sys.exit(1)

    # V2: pre-registration / lock date present (taxonomy_locked_on or
    # preregistered_date are both accepted)
    locked_on = doc.get("taxonomy_locked_on") or doc.get("preregistered_date")
    _add(findings, "V2", bool(locked_on),
         "taxonomy_locked_on / preregistered_date present",
         value=locked_on)

    invs = doc.get("invariants") or []

    # V3: exactly 9 invariants
    _add(findings, "V3", len(invs) == 9,
         "Exactly 9 invariants", count=len(invs))

    # V4: IDs unique 1-9
    ids = [inv.get("id") for inv in invs]
    _add(findings, "V4", set(ids) == set(range(1, 10)),
         "IDs unique and complete 1-9", ids=ids)

    # V5-V10: per-invariant
    for inv in invs:
        inv_id = inv.get("id")
        prefix = f"Inv {inv_id}"

        missing = [f for f in ("title", "text", "rationale")
                   if not (inv.get(f) or "").strip()]
        _add(findings, f"V5-{inv_id}", not missing,
             f"{prefix} has title/text/rationale",
             missing_fields=missing)

        has_test = bool(inv.get("test_files"))
        has_grep = bool(inv.get("grep_audit"))
        _add(findings, f"V6-{inv_id}", has_test or has_grep,
             f"{prefix} has at least one test_file or grep_audit",
             test_files=inv.get("test_files"),
             has_grep_audit=has_grep)

        sev = inv.get("severity")
        _add(findings, f"V7-{inv_id}", sev in VALID_SEVERITY,
             f"{prefix} severity valid", value=sev)

        status = inv.get("status")
        _add(findings, f"V8-{inv_id}", status in VALID_STATUS,
             f"{prefix} status valid", value=status)

        # V9: enforced → all listed test files exist on disk
        if status == "enforced":
            missing_files = [tf for tf in (inv.get("test_files") or [])
                             if not (REPO_ROOT / tf).exists()]
            _add(findings, f"V9-{inv_id}", not missing_files,
                 f"{prefix} test files exist (enforced)",
                 missing_files=missing_files)

        rqs = set(inv.get("serves_rqs") or [])
        _add(findings, f"V10-{inv_id}", rqs.issubset(VALID_RQS) and bool(rqs),
             f"{prefix} serves_rqs valid", value=sorted(rqs))

    _finalize(findings, invs)


def _finalize(findings: list, invs: list) -> None:
    n_fail = sum(1 for f in findings if f["severity"] == "FAIL")
    audit = {
        "_meta": {
            "schema_version": "1.0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "generated_by": "analysis/validate_invariant_manifest.py",
            "manifest_path": str(MANIFEST.relative_to(REPO_ROOT)),
        },
        "headline": {
            "validation_pass": n_fail == 0,
            "n_invariants": len(invs),
            "n_checks": len(findings),
            "n_fail": n_fail,
        },
        "findings": findings,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(audit, indent=2, default=str))
    print(f"Wrote {OUT.relative_to(REPO_ROOT)}")
    if n_fail == 0:
        print("Validation: PASS")
    else:
        print(f"Validation: FAIL ({n_fail} check(s))")
        for f in findings:
            if f["severity"] == "FAIL":
                print(f"  - {f['check_id']}: {f['description']} "
                      f"{f.get('details') or ''}")
        sys.exit(1)


if __name__ == "__main__":
    main()
