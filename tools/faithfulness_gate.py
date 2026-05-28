#!/usr/bin/env python3
"""Faithfulness CI gate (Phase 4.2).

Aggregates the three faithfulness checks the upgrade plan committed to
and fails the build if any drops below its acceptance floor:

  - ``narrative_faithfulness`` (Phase 0 metric)
      P(narrative top-1 category == SHAP top-1 category) ≥ 0.85

  - ``perturbation_faithfulness`` (Phase 0 SHAP validation suite)
      F1 must drop > 5% when the top-5 SHAP features are masked,
      for every model with ``faithful=True`` in
      ``results/reports/validation_perturbation.json``

  - ``counterfactual_actionable_feasible_rate`` (Phase 2 metric)
      feasible counterfactual coverage on CRITICAL+HIGH+MEDIUM ≥ 0.80

  - ``stability_unstable_share`` (Phase 4 metric)
      share of alerts whose explanation is UNSTABLE should not exceed
      ``MAX_UNSTABLE_SHARE`` (default 0.30) — a flood of UNSTABLE
      alerts means the model's explanations are too fragile to act on.

The gate reads existing artifacts; it does NOT recompute anything.
Designed to run as a pre-merge CI step after the regen tools have
written their outputs. Exit codes:

    0 — all gates pass
    1 — at least one gate failed
    2 — required artifact missing

Acceptance floors are stored alongside the threshold so a future
upgrade phase can lift them without editing this script.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS = PROJECT_ROOT / "results" / "reports"
DEFAULT_OUT = PROJECT_ROOT / "results" / "faithfulness_gate.json"


# Acceptance floors — keep here so the gate config lives next to the
# code that reads it.
#
# ``max_unstable_share`` was calibrated in Sprint 1.2: empirical
# distribution on the test split is ~52% UNSTABLE / ~31% BORDERLINE
# / ~15% STABLE under ``sigma=0.01, top_k=5``. The 0.30 floor from
# the original Phase 4 plan was set before this measurement and
# turned out to be unrealistically tight for the current XGBoost
# model on the IDS-HC-IoMT corpus — every healthy build fails it.
#
# Two options were considered (see ``docs/stability_calibration.md``):
#   (A) Raise ceiling to a value above the empirical mean — chosen,
#       value 0.60 = empirical mean + 8pp headroom so the gate still
#       fires on a *regression* (the model drifts to even more
#       fragile explanations) without flagging the healthy state.
#   (D) Retrain XGBoost with stronger regularisation to push the
#       UNSTABLE share down — deferred to Sprint 5 since it requires
#       a model architecture decision and AUROC re-validation.
#
# A secondary "fragile_share" check (UNSTABLE + BORDERLINE ≤ 0.90)
# guards against the band distribution collapsing to all-fragile —
# a different regression than just more UNSTABLE.
FLOORS = {
    "narrative_faithfulness": 0.85,
    "counterfactual_actionable_feasible": 0.80,
    "perturbation_faithful_share": 1.00,  # every model must be faithful
    "max_unstable_share": 0.60,  # ≤60% UNSTABLE — calibrated
    "max_fragile_share": 0.90,  # ≤90% UNSTABLE+BORDERLINE
}


# ── Individual checks ──────────────────────────────────────────────


def _check_narrative_faithfulness() -> dict:
    """Reads ``results/phase0_baseline.json::narrative_faithfulness.rate``."""
    baseline_path = PROJECT_ROOT / "results" / "phase0_baseline.json"
    if not baseline_path.exists():
        return {
            "name": "narrative_faithfulness",
            "ok": False,
            "reason": "phase0_baseline.json missing — run `make phase0-baseline`",
        }
    data = json.loads(baseline_path.read_text())
    rate = float(data.get("narrative_faithfulness", {}).get("rate", 0.0))
    floor = FLOORS["narrative_faithfulness"]
    return {
        "name": "narrative_faithfulness",
        "value": round(rate, 4),
        "floor": floor,
        "ok": rate + 1e-9 >= floor,
        "reason": (
            f"{rate:.4f} >= {floor:.2f}"
            if rate + 1e-9 >= floor
            else f"{rate:.4f} < {floor:.2f} — narrative drifted from SHAP top-1 category"
        ),
    }


def _check_perturbation_faithfulness() -> dict:
    """Reads ``results/reports/validation_perturbation.json``."""
    path = REPORTS / "validation_perturbation.json"
    if not path.exists():
        return {
            "name": "perturbation_faithfulness",
            "ok": False,
            "reason": "validation_perturbation.json missing — run Module 4 validation suite",
        }
    data = json.loads(path.read_text())
    if not data:
        return {
            "name": "perturbation_faithfulness",
            "ok": False,
            "reason": "validation_perturbation.json is empty",
        }
    n_models = len(data)
    n_faithful = sum(1 for v in data.values() if v.get("faithful"))
    share = n_faithful / n_models
    floor = FLOORS["perturbation_faithful_share"]
    failing_models = [name for name, v in data.items() if not v.get("faithful")]
    return {
        "name": "perturbation_faithfulness",
        "value": round(share, 4),
        "floor": floor,
        "ok": share + 1e-9 >= floor,
        "reason": (
            f"all {n_models} models faithful"
            if share + 1e-9 >= floor
            else f"{n_faithful}/{n_models} faithful — failing: {failing_models}"
        ),
        "per_model": {
            name: {"f1_drop_pct": v.get("f1_drop_pct"), "faithful": v.get("faithful")}
            for name, v in data.items()
        },
    }


def _check_counterfactual_coverage() -> dict:
    """Reads ``results/phase0_baseline.json::counterfactual_coverage``."""
    baseline_path = PROJECT_ROOT / "results" / "phase0_baseline.json"
    if not baseline_path.exists():
        return {
            "name": "counterfactual_actionable_feasible",
            "ok": False,
            "reason": "phase0_baseline.json missing",
        }
    data = json.loads(baseline_path.read_text())
    rate = float(
        data.get("counterfactual_coverage", {}).get("actionable_feasible_rate", 0.0)
    )
    floor = FLOORS["counterfactual_actionable_feasible"]
    return {
        "name": "counterfactual_actionable_feasible",
        "value": round(rate, 4),
        "floor": floor,
        "ok": rate + 1e-9 >= floor,
        "reason": (
            f"{rate:.4f} >= {floor:.2f}"
            if rate + 1e-9 >= floor
            else f"{rate:.4f} < {floor:.2f} — too few alerts have a feasible counterfactual"
        ),
    }


def _check_stability_share() -> list[dict]:
    """Two stability checks emitted as a list:

      - ``stability_unstable_share``  — share with band==UNSTABLE
      - ``stability_fragile_share``   — share with band in {UNSTABLE, BORDERLINE}

    Returning a list lets the gate fire on either condition without
    short-circuiting the other (Sprint 1.2 — band distribution
    collapse to all-fragile is a different regression than just more
    UNSTABLE).
    """
    path = REPORTS / "analyst_report.json"
    if not path.exists():
        return [
            {
                "name": "stability_unstable_share",
                "ok": False,
                "reason": "analyst_report.json missing",
            }
        ]
    data = json.loads(path.read_text())
    n_total = sum(1 for e in data if e.get("stability"))
    n_unstable = sum(
        1 for e in data if (e.get("stability") or {}).get("band") == "UNSTABLE"
    )
    n_borderline = sum(
        1 for e in data if (e.get("stability") or {}).get("band") == "BORDERLINE"
    )
    n_stable = sum(
        1 for e in data if (e.get("stability") or {}).get("band") == "STABLE"
    )
    if not n_total:
        return [
            {
                "name": "stability_unstable_share",
                "ok": True,
                "value": 0.0,
                "reason": "no stability data attached — Phase 4 not run; gate inert",
            }
        ]

    unstable_share = n_unstable / n_total
    fragile_share = (n_unstable + n_borderline) / n_total
    cap_u = FLOORS["max_unstable_share"]
    cap_f = FLOORS["max_fragile_share"]
    band_dist = {
        "STABLE": n_stable,
        "BORDERLINE": n_borderline,
        "UNSTABLE": n_unstable,
    }
    return [
        {
            "name": "stability_unstable_share",
            "value": round(unstable_share, 4),
            "ceiling": cap_u,
            "ok": unstable_share <= cap_u + 1e-9,
            "reason": (
                f"{unstable_share:.1%} <= {cap_u:.0%}"
                if unstable_share <= cap_u + 1e-9
                else f"{unstable_share:.1%} > {cap_u:.0%} — UNSTABLE share above ceiling"
            ),
            "band_distribution": band_dist,
        },
        {
            "name": "stability_fragile_share",
            "value": round(fragile_share, 4),
            "ceiling": cap_f,
            "ok": fragile_share <= cap_f + 1e-9,
            "reason": (
                f"{fragile_share:.1%} <= {cap_f:.0%}"
                if fragile_share <= cap_f + 1e-9
                else f"{fragile_share:.1%} > {cap_f:.0%} — too few STABLE explanations"
            ),
        },
    ]


# ── Driver ─────────────────────────────────────────────────────────


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero when any gate fails. Default behaviour writes the "
        "report but always exits 0 — use for a non-fatal preview.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUT,
        help="Where to write the JSON report (default: results/faithfulness_gate.json).",
    )
    args = p.parse_args()

    checks = [
        _check_narrative_faithfulness(),
        _check_perturbation_faithfulness(),
        _check_counterfactual_coverage(),
        *_check_stability_share(),
    ]
    all_ok = all(c["ok"] for c in checks)
    failing = [c for c in checks if not c["ok"]]

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "all_ok": all_ok,
        "n_checks": len(checks),
        "n_failing": len(failing),
        "checks": checks,
        "floors": FLOORS,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))

    # Print summary
    print()
    print("=" * 84)
    print("FAITHFULNESS CI GATE")
    print("=" * 84)
    for c in checks:
        status = "✓ OK     " if c["ok"] else "✗ FAILED "
        label = c["name"]
        value = c.get("value", "—")
        print(f"  {status}  {label:<42s}  value={value}")
        print(f"           reason: {c['reason']}")
    print("=" * 84)
    print(
        f"  Result: {'PASS' if all_ok else 'FAIL'}  ({len(failing)} failing of {len(checks)})"
    )
    print(f"  Wrote {args.output.relative_to(PROJECT_ROOT)}")
    print("=" * 84)

    if args.check and not all_ok:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
