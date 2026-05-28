#!/usr/bin/env python3
"""Offline regeneration of ``risk_scores`` artifacts using cached components.

The full Module 3 entry-point ``module3_risk_scoring.module3_risk_scores``
reloads the Track-A classifiers via the signed-pickle verifier, which
refuses the stale-sha pickles currently on disk (the models were
re-trained without re-signing — see Phase 2's offline regen note).

This tool sidesteps that by reading the *previously computed* risk
components straight out of the split-matched npz and re-running only
the final composition + tier assignment step. The components
themselves are not invalidated by the formula fix — only the way they
get combined into a final tier changed.

Usage:
  python -m tools.regen_risk_scores_offline test    # writes risk_scores.npz
  python -m tools.regen_risk_scores_offline demo    # writes demo_scores.npz

Inputs (read-only):
  - the split's risk-components npz (c_detect, d_crit, s_data,
    d_clinical_tier, y_true, plus c_track_a / c_track_b passthrough)

Output:
  - overwrites the same npz with R + tier labels recomputed under the
    upgraded formula
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from common import split_paths as sp  # noqa: E402
from module3_risk_scoring.composition import (  # noqa: E402
    assign_risk_levels,
    compute_composite_risk,
)


def main(split: str = "test") -> int:
    npz_path = sp.risk_scores(split)
    if not npz_path.exists():
        print(f"ERROR: {npz_path} not found", file=sys.stderr)
        return 2

    d = dict(np.load(npz_path, allow_pickle=True))
    required = ("c_detect", "d_crit", "s_data", "d_clinical_tier", "y_true")
    missing = [k for k in required if k not in d]
    if missing:
        print(f"ERROR: missing component arrays {missing} in {npz_path.name}",
              file=sys.stderr)
        return 2

    c_detect = d["c_detect"]
    R = compute_composite_risk(c_detect, d["d_crit"], d["s_data"], d["d_clinical_tier"])
    levels = assign_risk_levels(R, c_detect=c_detect)

    # Preserve other arrays (c_track_a, c_track_b, y_true) and overwrite R + levels.
    d["R"] = R
    d["risk_levels"] = levels

    np.savez(npz_path, **d)

    print(f"[regen-risk] split={split}  wrote {npz_path.relative_to(PROJECT_ROOT)}")
    print(f"            R range: [{R.min():.4f}, {R.max():.4f}]  mean={R.mean():.4f}")
    print(f"            tier distribution:")
    for tier in ("CRITICAL", "HIGH", "MEDIUM", "LOW", "NORMAL"):
        n = int((levels == tier).sum())
        pct = 100 * n / len(levels)
        print(f"              {tier:<10s} {n:>5d}  ({pct:>5.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "test"))
