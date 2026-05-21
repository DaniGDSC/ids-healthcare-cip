"""Render results/rq3_truth_table_reference.json -> Appendix B markdown.

Output: results/rq3_truth_table_appendix_b.md  (for thesis §5.6 / Appendix B).
"""
from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
JSON_IN = REPO_ROOT / "results" / "rq3_truth_table_reference.json"
OUT_MD = REPO_ROOT / "results" / "rq3_truth_table_appendix_b.md"

_STATUS_MARKER = {
    "pass": "PASS",
    "depends_ok": "depends",
    "fail": "FAIL",
    "row_missing": "MISSING",
}


def main() -> None:
    if not JSON_IN.exists():
        raise SystemExit(
            f"{JSON_IN.relative_to(REPO_ROOT)} missing — "
            "run `pytest tests/test_rq3_truth_table_completeness.py` first."
        )
    data = json.loads(JSON_IN.read_text())

    lines: list[str] = []
    lines.append("# Appendix B - Tier x Surfacing Truth Table (RQ3)")
    lines.append("")
    lines.append(
        f"*Generated from `{data['_meta']['source_csv']}` on "
        f"{data['_meta']['generated_at']}.*"
    )
    lines.append("")
    lines.append(
        "This table enumerates the system's `should_surface` decision for "
        "every combination of `risk_tier`, `patchable`, and "
        "`maintenance_active`. Rows derived from "
        "`RQ3_expected_outputs.md §4.2` are verified by "
        "`tests/test_rq3_truth_table_completeness.py` and serve as the "
        "safety-engineering evidence for Invariant 2 (safety floor) and the "
        "maintenance-window suppression policy."
    )
    lines.append("")

    h = data["headline"]
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- **Verification status:** "
                 f"{'PASS' if h['verification_pass'] else 'FAIL'}")
    lines.append(f"- **Claims verified:** {h['n_pass']} / {h['n_claims_total']}")
    lines.append(f"- **'Depends on threshold' rows (presence verified):** "
                 f"{h['n_depends_ok']}")
    lines.append(f"- **Failures:** {h['n_fail']}")
    lines.append("")

    lines.append("## Table")
    lines.append("")
    lines.append("| risk_tier | patchable | maintenance | should_surface "
                 "| reason | verification |")
    lines.append("|---|---|---|---|---|---|")

    for r in data["results"]:
        c = r["claim"]
        row = r.get("matched_row") or {}
        surface = row.get("should_surface", "-")
        reason = row.get("reason", "-")
        marker = _STATUS_MARKER.get(r["status"], r["status"] or "?")
        lines.append(
            f"| {c['tier']} | {c['patchable']} | {c['maintenance']} | "
            f"{surface} | {reason} | {marker} |"
        )
    lines.append("")

    lines.append("## Verification semantics")
    lines.append("")
    lines.append("- **PASS** - row exists with the expected `should_surface` "
                 "value and reason prefix.")
    lines.append("- **depends** - row exists; outcome is non-binary per "
                 "§4.2 ('depends on threshold').")
    lines.append("- **FAIL** - outcome or reason mismatch.")
    lines.append("- **MISSING** - expected row absent from the canonical CSV.")
    lines.append("")

    if h["n_fail"]:
        lines.append("## Failures")
        lines.append("")
        for r in data["results"]:
            if r["status"] in {"fail", "row_missing"}:
                lines.append(f"- **{r['claim']['source_claim']}**: "
                             f"{r['details']}")
        lines.append("")

    lines.append("## Cross-references")
    lines.append("")
    lines.append(f"- Canonical CSV: `{data['_meta']['source_csv']}` "
                 "(produced by `module6_evaluation/make_rq1_truth_table.py`).")
    lines.append("- Spec reference: "
                 f"{data['_meta']['rq3_section_reference']}.")
    lines.append("- Invariant 2 ('Safety floor') in "
                 "`configs/invariants_manifest.yaml` is enforced by the "
                 "CRITICAL+False rows in this table.")
    lines.append("")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"Wrote {OUT_MD.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
