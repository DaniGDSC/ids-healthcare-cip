#!/usr/bin/env python3
"""Artifact schema-version gate (Sprint 6 / Tầng 3.5).

Walks every artifact in :data:`common.artifact_versioning.ARTIFACT_VERSIONS`
and verifies the on-disk ``_schema_version`` (or ``schema_version``)
matches the registry. Fails the build when any artifact is stale,
missing the version field, or carries an unknown version.

Usage:
    python -m tools.version_gate              # report + exit 0 (preview)
    python -m tools.version_gate --check      # exit 1 on any mismatch
    python -m tools.version_gate --reports DIR

Output:
    results/version_gate.json — machine-readable report
    stdout                    — tabular summary

Each artifact's check produces one row::

    OK   risk_scores.npz                v2.0 == v2.0
    OK   alert_responses.json           v3.2 == v3.2 (in _provenance)
    ✗    phase0_baseline.json           v2.0 ≠ v2.1 (stale — regen)
    skip analyst_report.json            no version yet (pending migration)
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from common.artifact_versioning import (  # noqa: E402
    ARTIFACT_VERSIONS,
    PENDING_ENVELOPE_MIGRATION,
    check_compatibility,
    read_version,
)


REPORTS_DIR = PROJECT_ROOT / "results" / "reports"
RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_OUT = RESULTS_DIR / "version_gate.json"


def _scan_one(name: str) -> dict:
    """Find the (possibly split-suffixed) artifact and check it."""
    # Probe common locations: results/reports/, results/, project root
    candidates = [
        REPORTS_DIR / name,
        RESULTS_DIR / name,
        PROJECT_ROOT / name,
    ]
    # Also try demo-suffix and test-suffix variants
    if name.endswith(".npz"):
        stem = name[: -len(".npz")]
        candidates.extend([
            REPORTS_DIR / f"{stem}_demo.npz",
            REPORTS_DIR / f"{stem}_test.npz",
        ])
        if stem == "risk_scores":
            candidates.append(REPORTS_DIR / "demo_scores.npz")
    elif name.endswith(".json"):
        stem = name[: -len(".json")]
        candidates.extend([
            REPORTS_DIR / f"{stem}_demo.json",
            REPORTS_DIR / f"{stem}_test.json",
        ])

    found = [p for p in candidates if p.exists()]
    rows = []
    for path in found:
        check = check_compatibility(path)
        rows.append({
            "path":     str(path.relative_to(PROJECT_ROOT)),
            "artifact": check.artifact,
            "on_disk":  check.on_disk,
            "expected": check.expected,
            "ok":       check.ok,
            "reason":   check.reason,
        })
    if not rows:
        rows.append({
            "path":     None,
            "artifact": name,
            "on_disk":  None,
            "expected": ARTIFACT_VERSIONS.get(name),
            "ok":       True,
            "reason":   "artifact not present — skipped",
        })
    return {"name": name, "rows": rows}


def _scan_pending() -> dict:
    """Special-case scan for the artifacts queued for envelope migration —
    the gate reports their state as ``skip`` rather than fail."""
    out = []
    for name in PENDING_ENVELOPE_MIGRATION:
        # Probe split variants
        candidates = [REPORTS_DIR / name]
        stem = name[: -len(".json")]
        candidates.extend([
            REPORTS_DIR / f"{stem}_demo.json",
            REPORTS_DIR / f"{stem}_test.json",
        ])
        for path in candidates:
            if path.exists():
                out.append({
                    "path":     str(path.relative_to(PROJECT_ROOT)),
                    "artifact": name,
                    "on_disk":  read_version(path),
                    "expected": None,
                    "ok":       True,  # skipped, not failed
                    "reason":   "pending envelope migration — gate inert",
                })
    return {"pending": out}


def _print(report: dict) -> None:
    n_total = sum(len(b["rows"]) for b in report["registered"])
    n_failing = sum(
        1 for b in report["registered"] for r in b["rows"] if not r["ok"]
    )
    print()
    print("=" * 84)
    print(f" SCHEMA VERSION GATE — {n_total} registered, {len(report['pending']['pending'])} pending")
    print("=" * 84)
    for block in report["registered"]:
        for r in block["rows"]:
            status = "  OK " if r["ok"] else "  ✗  "
            on, exp = r["on_disk"], r["expected"]
            line = f"{status}  {r['artifact']:<35s}  on_disk={on!s:<8} expected={exp!s:<8}"
            print(line)
            if not r["ok"]:
                print(f"          reason: {r['reason']}")
    if report["pending"]["pending"]:
        print()
        print("Pending envelope migration (skipped by gate):")
        for r in report["pending"]["pending"]:
            print(f"  skip  {r['artifact']:<35s}  ({r['path']})")
    print("=" * 84)
    print(f"  Result: {'PASS' if n_failing == 0 else 'FAIL'}  "
          f"({n_failing} mismatches out of {n_total} registered checks)")
    print("=" * 84)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--check", action="store_true",
                   help="Exit non-zero on any mismatch.")
    p.add_argument("--output", type=Path, default=DEFAULT_OUT,
                   help="Path for the JSON report.")
    args = p.parse_args()

    report = {
        "_meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "n_registered": len(ARTIFACT_VERSIONS),
            "n_pending":    len(PENDING_ENVELOPE_MIGRATION),
        },
        "registered": [_scan_one(name) for name in ARTIFACT_VERSIONS],
        "pending":    _scan_pending(),
    }

    _print(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(f"\n  wrote {args.output.relative_to(PROJECT_ROOT)}")

    n_failing = sum(
        1 for b in report["registered"] for r in b["rows"] if not r["ok"]
    )
    if args.check and n_failing:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
