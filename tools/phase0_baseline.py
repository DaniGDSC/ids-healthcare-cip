#!/usr/bin/env python3
"""Phase-0 baseline runner for the faithfulness/actionability upgrade.

Reads ``results/reports/{analyst_report,clinician_summaries,alert_responses}.json``
and writes ``results/phase0_baseline.json`` with three metrics:

  - narrative_faithfulness
  - action_specificity
  - counterfactual_coverage

Run before merging any Phase-1+ work so the lift is measurable.

Usage:
    python -m tools.phase0_baseline                # writes results/phase0_baseline.json
    python -m tools.phase0_baseline --check        # exits non-zero if metrics regress
    python -m tools.phase0_baseline --reports DIR  # custom reports directory
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from module4_explanations.phase0_metrics import collect_baseline  # noqa: E402


DEFAULT_REPORTS_DIR = PROJECT_ROOT / "results" / "reports"
DEFAULT_OUTPUT      = PROJECT_ROOT / "results" / "phase0_baseline.json"

_FLOORS_KEY = "_floors"


def _git_rev() -> str | None:
    import subprocess
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def _print_summary(baseline: dict) -> None:
    nf = baseline["narrative_faithfulness"]
    asp = baseline["action_specificity"]
    cf = baseline["counterfactual_coverage"]

    print(f"\n  narrative_faithfulness   {nf['rate']:.1%}"
          f"  ({nf['n_matched']}/{nf['n']} matched"
          f", {nf['n_unknown_narrative']} unknown narrative"
          f", {nf['n_unknown_shap_category']} unknown SHAP cat)")

    print(f"  action_specificity       {asp['overall_rate']:.1%}  overall")
    for src, rate in asp["per_source_rate"].items():
        seen = asp["per_source_seen"].get(src, 0)
        print(f"    - {src:24s} {rate:>6.1%}   (n={seen})")
    if any(asp["signal_breakdown"].values()):
        print("    signals present:")
        for sig, count in sorted(asp["signal_breakdown"].items(), key=lambda x: -x[1]):
            if count:
                print(f"      {sig:18s} {count}")

    print(f"  counterfactual_coverage  {cf['rate']:.1%}"
          f"  ({cf['n_with_counterfactual']}/{cf['n']})")
    if "actionable_feasible_rate" in cf:
        print(f"    actionable_feasible       {cf['actionable_feasible_rate']:.1%}"
              f"  (CRITICAL+HIGH+MEDIUM denominator)")
        for tier, b in cf.get("by_severity", {}).items():
            if b["seen"]:
                cov = b["feasible"] / b["seen"]
                print(f"      {tier:10s}  feasible {b['feasible']}/{b['seen']}  ({cov:.1%})")

    # Sprint 2.4 — operational_health
    oh = baseline.get("operational_health", {})
    if oh.get("n"):
        print(f"  operational_health       op_prec={oh['operational_precision']:.1%}, "
              f"op_recall={oh['operational_recall']:.1%}, "
              f"surf_prec={oh['surfaced_precision']:.1%}")
        print(f"    LOW-tier attack density:  {oh['low_tier_attack_density']:.1%} "
              f"({oh['low_tier_attacks']}/{oh['low_tier_n']})")


def _check_against_floors(baseline: dict, prior: dict) -> tuple[bool, list[str]]:
    floors = prior.get(_FLOORS_KEY) or {}
    regressions: list[str] = []

    def _check(metric: str, current: float) -> None:
        floor = floors.get(metric)
        if floor is None:
            return
        if current + 1e-9 < floor:
            regressions.append(
                f"{metric}: current={current:.4f} < floor={floor:.4f}"
            )

    _check("narrative_faithfulness",  baseline["narrative_faithfulness"]["rate"])
    _check("action_specificity",      baseline["action_specificity"]["overall_rate"])
    _check("counterfactual_coverage", baseline["counterfactual_coverage"]["rate"])
    # Phase 2 — additional floor on the actionable-tier counterfactual
    # rate (CRITICAL+HIGH+MEDIUM).
    af = baseline["counterfactual_coverage"].get("actionable_feasible_rate")
    if af is not None:
        _check("counterfactual_actionable_feasible_rate", af)
    # Sprint 2.4 — operational precision floor. This is the bottom-up
    # health check that would have flagged the pre-formula-fix
    # 0.125 noise floor; lifting the floor here ensures we never
    # silently regress to that state again.
    oh = baseline.get("operational_health", {})
    if oh.get("n"):
        _check("operational_precision", oh["operational_precision"])

    return (len(regressions) == 0), regressions


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reports", type=Path, default=DEFAULT_REPORTS_DIR,
                   help="Reports directory (default: results/reports)")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                   help="Baseline JSON output path")
    p.add_argument("--check", action="store_true",
                   help="Compare against floors stored in output file; "
                        "exit non-zero on regression. No-op on first run.")
    p.add_argument("--update-floors", action="store_true",
                   help="Overwrite the floors block with current metric values. "
                        "Use after a Phase-N PR that intentionally raises floors.")
    args = p.parse_args()

    if not args.reports.exists():
        print(f"ERROR: reports dir not found: {args.reports}", file=sys.stderr)
        return 2

    print(f"[phase0] Reading reports from {args.reports.relative_to(PROJECT_ROOT)}")
    baseline = collect_baseline(args.reports)
    baseline["_meta"]["generated_at"] = datetime.now(timezone.utc).isoformat()
    baseline["_meta"]["git_rev"] = _git_rev()

    prior: dict = {}
    if args.output.exists():
        try:
            prior = json.loads(args.output.read_text())
        except Exception:
            prior = {}

    if args.update_floors or _FLOORS_KEY not in prior:
        baseline[_FLOORS_KEY] = {
            "narrative_faithfulness":  baseline["narrative_faithfulness"]["rate"],
            "action_specificity":      baseline["action_specificity"]["overall_rate"],
            "counterfactual_coverage": baseline["counterfactual_coverage"]["rate"],
            "counterfactual_actionable_feasible_rate":
                baseline["counterfactual_coverage"].get("actionable_feasible_rate", 0.0),
            # Sprint 2.4 — operational precision floor (the
            # bottom-up "did the formula start spamming?" sentinel)
            "operational_precision":
                baseline.get("operational_health", {}).get("operational_precision", 0.0),
        }
        floors_action = "wrote new" if args.update_floors else "initialised"
        print(f"[phase0] {floors_action} floors block")
    else:
        baseline[_FLOORS_KEY] = prior[_FLOORS_KEY]

    _print_summary(baseline)

    if args.check:
        ok, regressions = _check_against_floors(baseline, prior)
        if not ok:
            print("\n[phase0] REGRESSION DETECTED:", file=sys.stderr)
            for r in regressions:
                print(f"  - {r}", file=sys.stderr)
            return 1
        print("\n[phase0] All metrics ≥ recorded floors")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Sprint 6 / Tầng 3.5 — embed _schema_version so the version gate
    # can detect stale baseline files.
    from common.artifact_versioning import embed_version_in_dict
    args.output.write_text(json.dumps(
        embed_version_in_dict(baseline, args.output.name), indent=2,
    ))
    print(f"[phase0] Wrote {args.output.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
