#!/usr/bin/env python3
"""Compute and cache Track-A SHAP values for the demo split.

The demo split has no cached SHAP values on disk because Module 4 was
previously run in ``--explanations-only`` (thin) mode for demo, which
skips SHAP persistence. ``tools/phase1_regen_module4.py`` reads SHAP
from the cache so it can run offline (no signed-pickle load on the
hot path); without a demo SHAP cache, the Phase 2 (counterfactual) +
Phase 4 (stability) enrichments can't be applied to demo records.

This tool fixes that: load the (Sprint 1.1 re-signed) classifier,
compute TreeSHAP for the demo split, and write the cache file in the
same schema the test split uses::

    results/reports/shap_values_<model>_demo.npz
      ├── shap_values:    float64[N, F]   (attack-class slice)
      ├── expected_value: float64         (base value)
      └── feature_names:  str[F]

The naming uses an ``_demo`` suffix so callers can pick the right file
per split.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from module4_explanations.compute import compute_tree_shap  # noqa: E402
from module4_explanations.config import SHAP_MODELS, TRACK_A_MODELS  # noqa: E402
from module4_explanations.io import load_test_data  # noqa: E402

DEMO_PARQUET = PROJECT_ROOT / "data/processed/demo_phase1.parquet"


def main() -> int:
    if not DEMO_PARQUET.exists():
        print(f"ERROR: {DEMO_PARQUET} not found", file=sys.stderr)
        return 2

    X, _y, _attack_cats, feat_names = load_test_data(DEMO_PARQUET)
    print(f"[regen-demo-shap] loaded {len(X)} demo samples × {len(feat_names)} features")

    for name in SHAP_MODELS:
        cfg = TRACK_A_MODELS[name]
        t0 = time.perf_counter()
        sv, expected = compute_tree_shap(
            name, PROJECT_ROOT / cfg["pipeline"], X, list(feat_names),
        )
        elapsed = time.perf_counter() - t0
        out = PROJECT_ROOT / f"results/reports/shap_values_{name}_demo.npz"
        np.savez(
            out,
            shap_values=sv.astype(np.float64),
            expected_value=np.float64(expected),
            feature_names=np.array(feat_names),
        )
        print(f"  ✓ {name:<15s} shape={sv.shape} → {out.relative_to(PROJECT_ROOT)}  ({elapsed:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
