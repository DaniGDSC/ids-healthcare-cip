#!/usr/bin/env python3
"""run_tests.py — Entry point for XAI-IDS-Healthcare prototype test suite.

Loads fixtures → runs harness → prints report → generates alignment_report.yaml.
Exit code 0 if all tests PASS/WARN, 1 if any FAIL.

Usage:
    python run_tests.py
    python run_tests.py --fixture tests/fixtures/sample_alerts.yaml
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def _recommendation(report) -> str:
    """Compute recommendation per alignment_report_format criteria."""
    any_fail = any(m["pass_fail"] == "FAIL" for m in report.metrics)
    any_neg_fail = any(t["pass_fail"] == "FAIL" for t in report.negative_tests)

    if any_fail or any_neg_fail:
        return "BLOCKED"

    any_warn = any(m["pass_fail"] == "WARN" for m in report.metrics)
    n_supported = sum(
        1 for a in report.alignment
        if a.get("verdict") == "SUPPORTED"
    )
    n_partial = sum(
        1 for a in report.alignment
        if a.get("verdict") == "PARTIAL"
    )

    if any_warn or n_partial >= 1:
        return "ITERATE"

    if n_supported >= 4:
        return "SHIP_TO_USER_STUDY"

    return "ITERATE"


def _print_report(report, recommendation: str) -> None:
    """Print human-readable test summary to stdout."""
    sep = "=" * 72

    print()
    print(sep)
    print("XAI-IDS-Healthcare Prototype — Test Report")
    print(sep)

    # Automated tests
    print("\nAUTOMATED TESTS")
    print("-" * 72)
    header = f"{'Metric':<8} {'Name':<38} {'Result':>8} {'Target':>8} {'Min':>7}  Status"
    print(header)
    print("-" * 72)
    for m in report.metrics:
        pf_marker = {"PASS": "✓", "WARN": "!", "FAIL": "✗"}.get(m["pass_fail"], "?")
        print(
            f"{m['metric_id']:<8} {m['metric_name']:<38} "
            f"{m['result_value']:>7.1%} {m['target']:>7.1%} {m['minimum']:>6.1%}  "
            f"{pf_marker} {m['pass_fail']}"
        )
        if m.get("detail"):
            print(f"         └─ {m['detail'][:80]}")

    # Negative tests
    print("\nNEGATIVE TESTS (scope boundary)")
    print("-" * 72)
    for t in report.negative_tests:
        marker = "✓" if t["pass_fail"] == "PASS" else "✗"
        violations_note = (
            f"({t['violations_found']} violations)"
            if t["violations_found"] > 0 else ""
        )
        print(f"  {marker} {t['test_name']} {violations_note}")
        for v in t["violations"][:3]:
            print(f"      └─ {v[:100]}")

    # Alignment
    print("\nCLAIM ALIGNMENT")
    print("-" * 72)
    for a in report.alignment:
        verdict = a.get("verdict", "UNKNOWN")
        marker = {"SUPPORTED": "✓", "PARTIAL": "!", "NOT_SUPPORTED": "✗",
                  "NOT_TESTED": "—"}.get(verdict, "?")
        tests = ", ".join(a["supported_by"]) if a["supported_by"] else "none"
        print(f"  {marker} {a['claim_id']}: {verdict}  (tests: {tests})")

    # Recommendation
    rec_marker = {"SHIP_TO_USER_STUDY": "✓", "ITERATE": "!", "BLOCKED": "✗"}.get(
        recommendation, "?"
    )
    print()
    print(sep)
    print(f"  RECOMMENDATION: {rec_marker} {recommendation}")
    print(sep)
    print()


def _write_alignment_report(report, recommendation: str, out_path: Path) -> None:
    """Write alignment_report.yaml per research_spec.yaml section 6."""
    doc = {
        "test_results": report.metrics,
        "negative_test_results": report.negative_tests,
        "claims_supported": [
            a for a in report.alignment if a.get("supported_by")
        ],
        "claims_not_tested": [
            {
                "claim_id": a["claim_id"],
                "claim_text": a["claim_text"],
                "reason": a.get("verdict", ""),
            }
            for a in report.alignment
            if not a.get("supported_by")
        ],
        "recommendation": recommendation,
    }
    out_path.write_text(
        yaml.dump(doc, default_flow_style=False, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    print(f"  Alignment report: {out_path}")


def main() -> int:
    """Entry point. Returns exit code (0=ok, 1=fail)."""
    parser = argparse.ArgumentParser(
        description="XAI-IDS-Healthcare prototype test runner"
    )
    parser.add_argument(
        "--fixture",
        type=Path,
        default=None,
        help="Path to sample_alerts.yaml (default: tests/fixtures/sample_alerts.yaml)",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("alignment_report.yaml"),
        help="Output path for alignment_report.yaml",
    )
    args = parser.parse_args()

    logger.info("Starting XAI-IDS-Healthcare test suite")

    from src.harness import run_simulation

    report = run_simulation(fixture_path=args.fixture)
    recommendation = _recommendation(report)

    _print_report(report, recommendation)
    _write_alignment_report(report, recommendation, args.report)

    # Exit 1 if any FAIL
    any_fail = (
        any(m["pass_fail"] == "FAIL" for m in report.metrics)
        or any(t["pass_fail"] == "FAIL" for t in report.negative_tests)
    )
    return 1 if any_fail else 0


if __name__ == "__main__":
    sys.exit(main())
