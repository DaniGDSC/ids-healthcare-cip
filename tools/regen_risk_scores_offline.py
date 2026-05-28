#!/usr/bin/env python3
"""Offline regeneration of ``risk_scores`` artifacts using cached components.

The full Module 3 entry-point ``module3_risk_scoring.module3_risk_scores``
reloads the Track-A classifiers via the signed-pickle verifier. This
tool sidesteps that by reading the *previously computed* risk
components straight out of the split-matched npz and re-running only
the final composition + tier assignment step.

Sprint 4 — ``--formula-version`` exposes the v1/v2 switch so the
offline regen can land either version into the npz, and the
``formula_version`` field is recorded so downstream consumers know
which interpretation to apply.

Usage:
  python -m tools.regen_risk_scores_offline test                       # default v1
  python -m tools.regen_risk_scores_offline test --formula-version v2  # 2-layer
  python -m tools.regen_risk_scores_offline demo --formula-version v2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from common import split_paths as sp  # noqa: E402
from module3_risk_scoring.composition import (  # noqa: E402
    SUPPORTED_FORMULA_VERSIONS,
    assign_risk_levels,
    compute_composite_risk,
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("split", nargs="?", default="test", choices=("test", "demo"))
    p.add_argument("--formula-version", choices=SUPPORTED_FORMULA_VERSIONS, default="v1")
    args = p.parse_args(argv)

    split = args.split
    formula_version = args.formula_version

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
    R = compute_composite_risk(
        c_detect, d["d_crit"], d["s_data"], d["d_clinical_tier"],
        formula_version=formula_version,
    )
    levels = assign_risk_levels(
        R, c_detect=c_detect, formula_version=formula_version,
    )

    # Preserve other arrays (c_track_a, c_track_b, y_true) and overwrite R + levels.
    d["R"] = R
    d["risk_levels"] = levels
    d["formula_version"] = np.array(formula_version, dtype=str)
    # Sprint 6 / Tầng 3.5 — embed schema_version so the version gate
    # can verify post-regen artifact freshness.
    from common.artifact_versioning import version_kwarg_for
    d.update(version_kwarg_for(npz_path.name))

    np.savez(npz_path, **d)

    print(f"[regen-risk] split={split}  formula_version={formula_version}  "
          f"wrote {npz_path.relative_to(PROJECT_ROOT)}")
    print(f"            R range: [{R.min():.4f}, {R.max():.4f}]  mean={R.mean():.4f}")
    print(f"            tier distribution:")
    for tier in ("CRITICAL", "HIGH", "MEDIUM", "LOW", "NORMAL"):
        n = int((levels == tier).sum())
        pct = 100 * n / len(levels)
        print(f"              {tier:<10s} {n:>5d}  ({pct:>5.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
