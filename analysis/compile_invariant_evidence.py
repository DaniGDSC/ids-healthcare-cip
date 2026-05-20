"""Aggregate pytest results + grep checks for each invariant in the manifest.

Inputs:
  configs/invariants_manifest.yaml
  tests/_report.json   (from `pytest --json-report --json-report-file=tests/_report.json`)

Outputs:
  results/rq3_invariant_evidence.json
"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "configs" / "invariants_manifest.yaml"
PYTEST_REPORT = REPO_ROOT / "tests" / "_report.json"
OUT_JSON = REPO_ROOT / "results" / "rq3_invariant_evidence.json"


def _load_manifest() -> dict:
    if not MANIFEST.exists():
        raise SystemExit(
            f"Manifest missing: {MANIFEST.relative_to(REPO_ROOT)}. "
            "Run Phase 1 of RQ3_INVARIANT_EVIDENCE_SPEC.md."
        )
    return yaml.safe_load(MANIFEST.read_text())


def _load_pytest_report() -> Optional[dict]:
    if not PYTEST_REPORT.exists():
        return None
    try:
        return json.loads(PYTEST_REPORT.read_text())
    except json.JSONDecodeError:
        return None


def _tests_for_file(report: dict, test_file: str) -> list[dict]:
    """Return pytest-json-report entries whose nodeid begins with test_file."""
    return [t for t in report.get("tests", [])
            if t.get("nodeid", "").startswith(test_file)]


def _aggregate_test_results(report: Optional[dict],
                            test_files: list[str]) -> dict:
    if not report:
        return {
            "test_files": test_files,
            "n_tests_total": 0,
            "n_tests_passed": 0,
            "n_tests_failed": 0,
            "n_tests_skipped": 0,
            "outcome": "no_report",
        }
    all_tests: list[dict] = []
    for tf in test_files:
        all_tests.extend(_tests_for_file(report, tf))
    n_total = len(all_tests)
    n_passed = sum(1 for t in all_tests if t.get("outcome") == "passed")
    n_failed = sum(1 for t in all_tests if t.get("outcome") == "failed")
    n_skipped = sum(1 for t in all_tests if t.get("outcome") == "skipped")

    if n_total == 0:
        outcome = "no_tests_found"
    elif n_failed > 0:
        outcome = "fail"
    elif n_passed == n_total:
        outcome = "pass"
    elif n_passed > 0 and n_skipped == n_total - n_passed:
        outcome = "partial_skip"
    else:
        outcome = "unknown"

    return {
        "test_files": test_files,
        "n_tests_total": n_total,
        "n_tests_passed": n_passed,
        "n_tests_failed": n_failed,
        "n_tests_skipped": n_skipped,
        "outcome": outcome,
    }


def _run_grep_audit(grep_audit: dict) -> dict:
    target_dirs_rel = list(grep_audit.get("target_dirs") or [])
    target_dirs = [REPO_ROOT / d for d in target_dirs_rel
                   if (REPO_ROOT / d).exists()]
    if not target_dirs:
        return {
            "target_dirs": target_dirs_rel,
            "pattern_matches": [],
            "import_matches": [],
            "n_pattern_matches": 0,
            "n_import_matches": 0,
            "outcome": "no_target_dirs_exist",
        }

    pattern_matches: list[dict] = []
    for pat in grep_audit.get("forbidden_patterns") or []:
        for d in target_dirs:
            res = subprocess.run(
                ["grep", "-rnE", "--include=*.py", pat, str(d)],
                capture_output=True, text=True,
            )
            if res.stdout.strip():
                for line in res.stdout.strip().splitlines():
                    pattern_matches.append({"pattern": pat, "match": line[:200]})

    import_matches: list[dict] = []
    for pat in grep_audit.get("forbidden_imports") or []:
        for d in target_dirs:
            res = subprocess.run(
                ["grep", "-rnE", "--include=*.py", pat, str(d)],
                capture_output=True, text=True,
            )
            if res.stdout.strip():
                for line in res.stdout.strip().splitlines():
                    import_matches.append({"pattern": pat, "match": line[:200]})

    outcome = "pass" if not (pattern_matches or import_matches) else "fail"
    return {
        "target_dirs": [str(d.relative_to(REPO_ROOT)) for d in target_dirs],
        "pattern_matches": pattern_matches[:20],
        "import_matches": import_matches[:20],
        "n_pattern_matches": len(pattern_matches),
        "n_import_matches": len(import_matches),
        "outcome": outcome,
    }


def _determine_overall_status(inv: dict, test_results: Optional[dict],
                              grep_results: Optional[dict]) -> str:
    manifest_status = inv.get("status", "enforced")
    if manifest_status == "pending":
        return "pending"
    if manifest_status == "documented":
        return "documented"

    # partial_skip = no failures + at least one passed + remainder skipped.
    # Treated as "pass" because the invariant's tests are designed to skip
    # when the data preconditions aren't met (e.g. HIPAA gate skips on
    # LLM-persona-only study data).
    PYTEST_PASS_OUTCOMES = {"pass", "partial_skip"}

    method = inv.get("verification_method", "pytest")
    if method == "pytest":
        return ("pass"
                if test_results and test_results["outcome"] in PYTEST_PASS_OUTCOMES
                else "fail")
    if method == "grep":
        return ("pass" if grep_results and grep_results["outcome"] == "pass"
                else "fail")
    if method == "grep_and_pytest":
        pytest_ok = bool(test_results
                         and test_results["outcome"] in PYTEST_PASS_OUTCOMES)
        grep_ok = bool(grep_results and grep_results["outcome"] == "pass")
        return "pass" if (pytest_ok and grep_ok) else "fail"
    return "unknown"


def main() -> None:
    doc = _load_manifest()
    report = _load_pytest_report()

    invariant_outputs: list[dict] = []
    for inv in doc.get("invariants", []) or []:
        manifest_status = inv.get("status", "enforced")
        test_results: Optional[dict] = None
        grep_results: Optional[dict] = None

        if manifest_status != "pending":
            if inv.get("test_files"):
                test_results = _aggregate_test_results(
                    report, inv["test_files"])
            if inv.get("grep_audit"):
                grep_results = _run_grep_audit(inv["grep_audit"])

        overall = _determine_overall_status(inv, test_results, grep_results)

        entry: dict[str, Any] = {
            "id": inv["id"],
            "title": inv["title"],
            "severity": inv.get("severity"),
            "serves_rqs": inv.get("serves_rqs") or [],
            "status_manifest": manifest_status,
            "verification_method": inv.get("verification_method"),
            "test_results": test_results,
            "grep_results": grep_results,
            "_overall_status": overall,
        }
        if manifest_status == "pending":
            entry["_note"] = inv.get("_note") or "Test creation scheduled"
        invariant_outputs.append(entry)

    n_total = len(invariant_outputs)
    n_enforced = sum(1 for i in invariant_outputs
                     if i["status_manifest"] == "enforced")
    n_pending = sum(1 for i in invariant_outputs
                    if i["status_manifest"] == "pending")
    n_documented = sum(1 for i in invariant_outputs
                       if i["status_manifest"] == "documented")
    n_failed = sum(1 for i in invariant_outputs
                   if i["_overall_status"] == "fail")
    n_no_tests = sum(1 for i in invariant_outputs
                     if i["_overall_status"] == "no_tests_found")
    all_pass = (n_failed == 0) and all(
        i["_overall_status"] in {"pass", "pending", "documented"}
        for i in invariant_outputs
    )

    out = {
        "_meta": {
            "schema_version": "1.0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "generated_by": "analysis/compile_invariant_evidence.py",
            "manifest_path": str(MANIFEST.relative_to(REPO_ROOT)),
            "taxonomy_locked_on": doc.get("taxonomy_locked_on"),
            "pytest_report_path": str(PYTEST_REPORT.relative_to(REPO_ROOT)),
            "pytest_report_available": report is not None,
        },
        "headline": {
            "all_invariants_pass": all_pass,
            "n_invariants_total": n_total,
            "n_enforced": n_enforced,
            "n_pending": n_pending,
            "n_documented": n_documented,
            "n_failed": n_failed,
            "n_no_tests_found": n_no_tests,
            "_overall_status": (
                "all enforced invariants pass" if all_pass
                else f"{n_failed} invariant(s) failing"
            ),
        },
        "invariants": invariant_outputs,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2, default=str))
    print(f"Wrote {OUT_JSON.relative_to(REPO_ROOT)}")
    print(f"Headline: {out['headline']['_overall_status']}")
    marker_map = {"pass": "PASS", "fail": "FAIL", "pending": "PEND",
                  "documented": "DOC ", "no_tests_found": "????",
                  "unknown": "????"}
    for i in invariant_outputs:
        m = marker_map.get(i["_overall_status"], "????")
        print(f"  {m} #{i['id']:<2} [{i['status_manifest']:<10}] {i['title']}")


if __name__ == "__main__":
    main()
