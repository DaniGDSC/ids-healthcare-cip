#!/usr/bin/env python3
"""Compute and cache Track-A SHAP values for a chosen split.

Originally a demo-only helper (Sprint 2 1.3). Sprint 5 generalised it
to accept either ``test`` or ``demo`` so the test SHAP cache can be
refreshed offline when it goes stale (e.g. when Module 4 demo regen
overwrites the test file).

Output: ``results/reports/shap_values_<model>{_demo}.npz`` with
``shap_values`` (attack-class slice), ``expected_value`` (base) and
``feature_names``. Test runs use no suffix to stay byte-compatible
with the original RQ1 paper artifact name.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from common import split_paths as sp  # noqa: E402
from module4_explanations.compute import compute_tree_shap  # noqa: E402
from module4_explanations.config import SHAP_MODELS, TRACK_A_MODELS  # noqa: E402
from module4_explanations.io import load_test_data  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("split", nargs="?", default="demo", choices=("test", "demo"))
    args = p.parse_args()
    split = args.split

    parquet = sp.parquet(split)
    if not parquet.exists():
        print(f"ERROR: {parquet} not found", file=sys.stderr)
        return 2
    suffix = "_demo" if split == "demo" else ""

    X, _y, _attack_cats, feat_names = load_test_data(parquet)
    print(f"[regen-shap] split={split}  loaded {len(X)} samples × {len(feat_names)} features")

    for name in SHAP_MODELS:
        cfg = TRACK_A_MODELS[name]
        t0 = time.perf_counter()
        sv, expected = compute_tree_shap(
            name, PROJECT_ROOT / cfg["pipeline"], X, list(feat_names),
        )
        elapsed = time.perf_counter() - t0
        out = PROJECT_ROOT / f"results/reports/shap_values_{name}{suffix}.npz"
        from common.artifact_versioning import version_kwarg_for
        np.savez(
            out,
            shap_values=sv.astype(np.float64),
            expected_value=np.float64(expected),
            feature_names=np.array(feat_names),
            **version_kwarg_for(out.name),
        )
        print(f"  ✓ {name:<15s} shape={sv.shape} → {out.relative_to(PROJECT_ROOT)}  ({elapsed:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
