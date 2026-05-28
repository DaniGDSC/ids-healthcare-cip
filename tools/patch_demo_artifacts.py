#!/usr/bin/env python3
"""Patch demo-split analyst/clinician artifacts to reflect the upgraded
Module 3 risk-level assignment.

The demo split was last generated under the pre-formula-fix Module 4 +
pre-Phase-2 codebase. Re-running the full Module 4 chain for demo would
require regenerating Track-A SHAP values, which we can't do offline
because the signed-pickle integrity check refuses the stale model
pickles. Instead, this tool patches the existing artifacts in-place:

  - drop clinician summaries whose new risk_level is NORMAL (the
    formula-fix detection gate now demotes context-driven false alerts
    that the pre-fix pipeline surfaced as LOW)
  - re-stamp ``severity`` and ``risk_level`` fields on every retained
    entry to the canonical value from ``demo_scores.npz``

The output is internally consistent with the new ``demo_scores.npz``
and the subsequent ``phase1_regen_module5.py demo`` run. It does NOT
add Phase 1.1 observation phrases or Phase 2 counterfactuals to demo —
those require a full Module 4 + counterfactual recompute that's only
worthwhile when SHAP values for the demo split are produced (a Module
2 / Module 4 re-train concern, not a Module 3 concern).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS = PROJECT_ROOT / "results" / "reports"


def main() -> int:
    demo_npz = REPORTS / "demo_scores.npz"
    if not demo_npz.exists():
        print(f"ERROR: {demo_npz} not found — run regen_risk_scores_offline first",
              file=sys.stderr)
        return 2

    risk_levels = np.load(demo_npz, allow_pickle=True)["risk_levels"]
    print(f"[patch-demo] loaded new risk_levels from {demo_npz.name} "
          f"({len(risk_levels)} samples)")

    # ── analyst_report_demo.json ──
    analyst_p = REPORTS / "analyst_report_demo.json"
    analyst = json.loads(analyst_p.read_text())
    n_an = len(analyst)
    re_stamped_an = 0
    for entry in analyst:
        idx = entry["sample_index"]
        new_level = str(risk_levels[idx])
        if entry.get("risk_level") != new_level or entry.get("severity") != new_level:
            re_stamped_an += 1
        entry["risk_level"] = new_level
        entry["severity"]   = new_level
    analyst_p.write_text(json.dumps(analyst, indent=2))
    print(f"[patch-demo] analyst_report: {n_an} entries kept, "
          f"{re_stamped_an} severity fields re-stamped")

    # ── clinician_summaries_demo.json ──
    clinician_p = REPORTS / "clinician_summaries_demo.json"
    clinician = json.loads(clinician_p.read_text())
    n_before = len(clinician)
    kept = []
    n_dropped_normal = 0
    n_re_stamped = 0
    for entry in clinician:
        idx = entry["sample_index"]
        new_level = str(risk_levels[idx])
        if new_level == "NORMAL":
            n_dropped_normal += 1
            continue
        if entry.get("severity") != new_level:
            n_re_stamped += 1
        entry["severity"]   = new_level
        entry["risk_level"] = new_level
        kept.append(entry)
    clinician_p.write_text(json.dumps(kept, indent=2))
    print(f"[patch-demo] clinician_summaries: {n_before} → {len(kept)} "
          f"({n_dropped_normal} dropped as NORMAL, {n_re_stamped} re-stamped)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
