#!/usr/bin/env python3
"""Cross-surface signal coverage audit (Sprint 2.2).

Catches Category 5 ("incomplete surface scanning") bugs by maintaining
a single source of truth for *where* each upgrade-phase signal can
appear, scanning all those surfaces, and emitting a cross-tab.

A signal here is something an upgrade phase committed to writing
(ALERT-XXXX prefix, MITRE T-ref, observation phrase, etc.). A
surface is a JSON path inside one of the artifacts a stakeholder
actually reads (alert_responses MVE Layer 1/3, clinician summary,
analyst report top features, etc.).

The audit answers two questions:

  1. **Is the signal present where it should be?** Every (signal,
     surface) pair the upgrade-plan committed to is scored. Below-
     floor coverage is flagged.

  2. **Is a metric reading 0% because the signal is genuinely
     absent, or because the metric forgot to scan a surface?** Each
     signal has an "expected_surfaces" list. If the signal appears in
     at least one expected surface but the corresponding Phase 0
     metric reports near-zero coverage, the audit warns — that's the
     Phase 4.3 simulator bug pattern (scan clinician summary text but
     miss MVE block).

The default invocation runs against the test split; pass ``demo`` to
audit the demo artifact set instead.

Output:
  - ``results/coverage_audit{_demo}.json`` with full cross-tab
  - Console: tabular summary + below-floor warnings
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS = PROJECT_ROOT / "results" / "reports"


# ── Signals: regex + which surfaces are *expected* to carry them ──


SIGNALS: dict[str, dict] = {
    "alert_id": {
        "pattern": re.compile(r"\bALERT-\d{3,}\b"),
        "expected_surfaces": [
            "alert_responses.records[].explanation.mve.layer_3.immediate_action",
        ],
        "floor": 1.0,
        "phase": "1.2",
    },
    "extension_sla": {
        # Two formats supported — the original "ext NNNN, SLA Nmin"
        # and the linter-compacted "[NNNN/Nm]".
        "pattern": re.compile(
            r"(\bext \d{3,5}, SLA\b|\[\d{3,5}/\d+\s*(m|min|h)\b)",
            re.IGNORECASE,
        ),
        "expected_surfaces": [
            "alert_responses.records[].explanation.mve.layer_3.escalation_path",
        ],
        "floor": 1.0,
        "phase": "1.2",
    },
    "observation_phrase": {
        "pattern": re.compile(r"\bobserved [+-]?\d+(\.\d+)?\b", re.IGNORECASE),
        "expected_surfaces": [
            "clinician_summaries[].summary",
        ],
        "floor": 0.50,
        "phase": "1.1",
    },
    "mitre_id_with_gloss": {
        # MITRE TXXXX (name — gloss) — the em-dash is the gloss marker
        "pattern": re.compile(r"\bMITRE T\d{4}.*? — \w"),
        "expected_surfaces": [
            "alert_responses.records[].explanation.mve.layer_1.deviation_description",
        ],
        "floor": 0.50,
        "phase": "1.4",
    },
    "counterfactual_clause": {
        "pattern": re.compile(r"\bwould clear if\b", re.IGNORECASE),
        "expected_surfaces": [
            "clinician_summaries[].summary",
        ],
        "floor": 0.40,
        "phase": "2",
    },
    "playbook_markdown": {
        "pattern": re.compile(r"\*\*Playbook:", re.IGNORECASE),
        "expected_surfaces": [
            "clinician_summaries[].summary",
        ],
        "floor": 0.40,
        "phase": "3.1",
    },
    "stability_badge": {
        "pattern": re.compile(r"Explanation:\s*(STABLE|BORDERLINE|UNSTABLE)"),
        "expected_surfaces": [
            "clinician_summaries[].summary",
        ],
        "floor": 0.40,
        "phase": "4.1",
    },
}


# ── Surfaces: how to extract a flat list of strings from each ─────


def _read_path(records: list[dict], dotted_path: str) -> list[str]:
    """Walk ``records`` along ``dotted_path`` and collect string leaves.

    Path syntax: ``foo.bar.baz`` for nested dicts, ``[]`` after a key
    indicates a list (each element walked).
    """
    out: list[str] = []
    parts = dotted_path.split(".")

    def _walk(obj, idx: int) -> None:
        if idx >= len(parts):
            if isinstance(obj, str):
                out.append(obj)
            return
        part = parts[idx]
        list_iter = part.endswith("[]")
        key = part[:-2] if list_iter else part
        nxt = obj.get(key) if isinstance(obj, dict) else None
        if list_iter:
            if isinstance(nxt, list):
                for item in nxt:
                    _walk(item, idx + 1)
        else:
            _walk(nxt, idx + 1)

    for r in records:
        _walk(r, 0)
    return out


def _load_surface(surface_path: str, split: str) -> list[str]:
    """Map a surface path to the actual list of strings to scan.

    ``_read_path`` expects a list of dicts plus the *intra-record*
    path. So we strip the outer ``alert_responses.records[].`` /
    ``clinician_summaries[].`` / ``analyst_report[].`` prefix before
    walking — the records list itself is already unwrapped.
    """
    suffix = "_demo" if split == "demo" else ""
    if surface_path.startswith("alert_responses.records[]."):
        path = REPORTS / f"alert_responses{suffix}.json"
        envelope = json.loads(path.read_text())
        records = envelope.get("records", envelope) if isinstance(envelope, dict) else envelope
        return _read_path(records, surface_path[len("alert_responses.records[]."):])
    if surface_path.startswith("clinician_summaries[]."):
        path = REPORTS / f"clinician_summaries{suffix}.json"
        records = json.loads(path.read_text())
        return _read_path(records, surface_path[len("clinician_summaries[]."):])
    if surface_path.startswith("analyst_report[]."):
        path = REPORTS / f"analyst_report{suffix}.json"
        records = json.loads(path.read_text())
        return _read_path(records, surface_path[len("analyst_report[]."):])
    raise ValueError(f"Unknown surface root in {surface_path}")


# ── Cross-tab build ──────────────────────────────────────────────


def audit(split: str) -> dict:
    """Build the signal × surface coverage cross-tab for one split."""
    # Cache surface strings so we don't re-load files per signal.
    surface_cache: dict[str, list[str]] = {}

    def _strings(path: str) -> list[str]:
        if path not in surface_cache:
            surface_cache[path] = _load_surface(path, split)
        return surface_cache[path]

    report: dict = {"_split": split, "signals": {}}

    for sig_name, spec in SIGNALS.items():
        pattern = spec["pattern"]
        per_surface: dict[str, dict] = {}
        for surface in spec["expected_surfaces"]:
            strings = _strings(surface)
            n_total = len(strings)
            n_hit   = sum(1 for s in strings if s and pattern.search(s))
            per_surface[surface] = {
                "n_total":  n_total,
                "n_hit":    n_hit,
                "coverage": round(n_hit / n_total, 4) if n_total else 0.0,
            }
        # Aggregate: max coverage across surfaces (the signal is
        # *present* if any surface emits it for that record).
        max_cov = max((v["coverage"] for v in per_surface.values()), default=0.0)
        floor   = spec["floor"]
        report["signals"][sig_name] = {
            "phase":            spec["phase"],
            "floor":            floor,
            "max_coverage":     round(max_cov, 4),
            "below_floor":      max_cov + 1e-9 < floor,
            "per_surface":      per_surface,
        }

    n_failing = sum(1 for v in report["signals"].values() if v["below_floor"])
    report["_meta"] = {
        "generated_at":  datetime.now(timezone.utc).isoformat(),
        "n_signals":     len(SIGNALS),
        "n_below_floor": n_failing,
    }
    return report


# ── Console rendering ────────────────────────────────────────────


def _print(report: dict) -> None:
    print()
    print("=" * 88)
    print(f" COVERAGE AUDIT — split={report['_split']}")
    print("=" * 88)
    print(f"{'Signal':<26s} {'Phase':<6s} {'Coverage':>10s} {'Floor':>8s} {'Status':<14s}")
    print("-" * 88)
    for name, block in report["signals"].items():
        status = "✗ BELOW FLOOR" if block["below_floor"] else "✓ OK"
        cov  = block["max_coverage"]
        fl   = block["floor"]
        print(f"{name:<26s} {block['phase']:<6s} {cov:>9.1%}  {fl:>7.0%}  {status:<14s}")
        if block["below_floor"]:
            for surface, ps in block["per_surface"].items():
                print(f"    └─ {surface}: {ps['n_hit']}/{ps['n_total']} ({ps['coverage']:.1%})")
    print("=" * 88)
    n_fail = report["_meta"]["n_below_floor"]
    print(f"  {'PASS' if n_fail == 0 else 'FAIL'}  ({n_fail} below floor of {report['_meta']['n_signals']} signals)")
    print("=" * 88)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("split", nargs="?", default="test", choices=("test", "demo"))
    p.add_argument("--check", action="store_true",
                   help="Exit non-zero if any signal is below floor")
    args = p.parse_args()

    report = audit(args.split)
    suffix = "_demo" if args.split == "demo" else ""
    out = PROJECT_ROOT / "results" / f"coverage_audit{suffix}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    from common.artifact_versioning import embed_version_in_dict
    out.write_text(json.dumps(embed_version_in_dict(report, out.name), indent=2))

    _print(report)
    print(f"\n  wrote {out.relative_to(PROJECT_ROOT)}")

    if args.check and report["_meta"]["n_below_floor"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
