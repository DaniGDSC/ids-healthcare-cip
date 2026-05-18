"""RQ2.a — Render the compliance manifest into markdown + audit JSON
(RQ2_Compliance.md Phase 4).

Reads ``configs/rq2_compliance_manifest.yaml`` and writes:

  * ``results/rq2_compliance_mapping.md``   — paper-appendix table
  * ``results/rq2_compliance_audit.json``   — CI-gated evidence audit

Each requirement has two evidence lists:

  ``evidence_files``    — REQUIRED, must exist; missing files FAIL the
                          downstream test in tests/test_compliance_manifest.py.
  ``evidence_pending``  — OPTIONAL, may not yet exist (Track 1 / 2 work
                          in progress, or LLM audit log not yet started);
                          reported as INFO, never fails.

Schema-drift drift note: spec used ``config/`` (singular).  Repo uses
``configs/`` (plural) consistently — manifest path and all evidence
references honour the real paths.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "configs/rq2_compliance_manifest.yaml"
OUT_MD = REPO_ROOT / "results/rq2_compliance_mapping.md"
OUT_AUDIT = REPO_ROOT / "results/rq2_compliance_audit.json"


def _evidence_status(path_str: str) -> str:
    """Return ``"present"`` / ``"missing"`` for an evidence path."""
    return "present" if (REPO_ROOT / path_str).exists() else "missing"


def main() -> None:
    if not MANIFEST.exists():
        print(f"Manifest not found: {MANIFEST}", file=sys.stderr)
        sys.exit(1)

    doc = yaml.safe_load(MANIFEST.read_text(encoding="utf-8")) or {}
    reqs = doc.get("requirements", []) or []

    audit: dict = {
        "_meta": {
            "schema_version": "1.0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "generated_by": "analysis/make_rq2_compliance_table.py",
            "manifest_path": str(MANIFEST.relative_to(REPO_ROOT)),
            "manifest_schema_version": doc.get("schema_version"),
            "last_validated": doc.get("last_validated"),
        },
        "requirements_total": len(reqs),
        "evidence_audit": [],
    }

    for req in reqs:
        required = list(req.get("evidence_files") or [])
        pending = list(req.get("evidence_pending") or [])
        missing_required = [p for p in required if _evidence_status(p) == "missing"]
        missing_pending = [p for p in pending if _evidence_status(p) == "missing"]
        audit["evidence_audit"].append({
            "id": req.get("id"),
            "literature_term": req.get("literature_term"),
            "evidence_required_total": len(required),
            "evidence_required_present": len(required) - len(missing_required),
            "evidence_required_missing": missing_required,
            "evidence_pending_total": len(pending),
            "evidence_pending_present": len(pending) - len(missing_pending),
            "evidence_pending_missing": missing_pending,
        })

    audit["all_required_evidence_present"] = all(
        not e["evidence_required_missing"] for e in audit["evidence_audit"]
    )
    audit["any_pending_evidence_missing"] = any(
        e["evidence_pending_missing"] for e in audit["evidence_audit"]
    )

    # ── Render markdown table for paper appendix ─────────────────────
    lines = [
        "# RQ2 — Compliance Mapping (literature ↔ MVE)",
        "",
        f"*Generated from `{MANIFEST.relative_to(REPO_ROOT)}` "
        f"on {audit['_meta']['generated_at']}.*  ",
        f"*Manifest last validated: {doc.get('last_validated', 'unknown')}.*  ",
        f"*Required evidence present: "
        f"{'YES' if audit['all_required_evidence_present'] else 'NO'}.*",
        "",
        "| Requirement | Literature Term | MVE Implementation | Required Evidence | Pending |",
        "|---|---|---|---|---|",
    ]
    for req in reqs:
        required = req.get("evidence_files") or []
        pending = req.get("evidence_pending") or []
        impl = (req.get("mve_implementation") or "").strip().replace("\n", " ")
        req_str = "<br>".join(f"`{p}`" for p in required) or "—"
        pend_str = "<br>".join(f"`{p}`" for p in pending) or "—"
        lines.append(
            f"| **{req.get('id')}** | {req.get('literature_term')} | "
            f"{impl} | {req_str} | {pend_str} |"
        )
    lines.extend(["", "---", "", "## Detailed Descriptions", ""])
    for req in reqs:
        lines.extend([
            f"### {req.get('id')} — {req.get('literature_term')}",
            "",
            (req.get("description") or "").strip(),
            "",
            f"**MVE Implementation:** "
            f"{(req.get('mve_implementation') or '').strip()}",
            "",
            "**Required Evidence:**",
        ])
        for p in (req.get("evidence_files") or []):
            mark = "✅" if _evidence_status(p) == "present" else "❌"
            lines.append(f"- {mark} `{p}`")
        pending = req.get("evidence_pending") or []
        if pending:
            lines.append("")
            lines.append("**Pending Evidence (informational):**")
            for p in pending:
                mark = "✅" if _evidence_status(p) == "present" else "⏳"
                lines.append(f"- {mark} `{p}`")
        lines.append("")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    OUT_AUDIT.write_text(
        json.dumps(audit, indent=2, default=str), encoding="utf-8"
    )

    print(f"Wrote {OUT_MD.relative_to(REPO_ROOT)}")
    print(f"Wrote {OUT_AUDIT.relative_to(REPO_ROOT)}")
    if not audit["all_required_evidence_present"]:
        offenders = [
            (e["id"], e["evidence_required_missing"])
            for e in audit["evidence_audit"]
            if e["evidence_required_missing"]
        ]
        print(f"FAIL: required evidence missing: {offenders}")
    else:
        print("PASS: all required evidence present")
    if audit["any_pending_evidence_missing"]:
        pending = [
            (e["id"], e["evidence_pending_missing"])
            for e in audit["evidence_audit"]
            if e["evidence_pending_missing"]
        ]
        print(f"INFO: pending evidence not yet on disk: {pending}")


if __name__ == "__main__":
    main()
