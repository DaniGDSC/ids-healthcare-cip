#!/usr/bin/env python3
"""Compute Track-A predictions for the demo split.

The demo split's prediction npz files were never generated — only the
DAE produced demo outputs. Module 4 demo regen requires y_pred/y_proba
for XGBoost / RandomForest / DecisionTree so SHAP + counterfactual +
stability can be computed.

Loads the (Sprint 1.1 re-signed) Track-A pickles via ``loads_signed``,
applies them to the demo parquet, and writes one npz per model with
the same schema as the test predictions::

    {y_true: int64[N], y_pred: int64[N], y_proba: float64[N]}

The threshold used for ``y_pred`` is the canonical Track-A threshold
from ``common.model_registry.get_track_a_thresholds()``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from common import loads_signed  # noqa: E402
from common.model_registry import get_track_a_thresholds  # noqa: E402
from module4_explanations.io import load_test_data  # noqa: E402


MODELS = {
    "xgboost":       PROJECT_ROOT / "results/models/xgboost_final_pipeline.pkl",
    "random_forest": PROJECT_ROOT / "results/models/random_forest_final_pipeline.pkl",
    "decision_tree": PROJECT_ROOT / "results/models/decision_tree_final_pipeline.pkl",
}

DEMO_PARQUET = PROJECT_ROOT / "data/processed/demo_phase1.parquet"


def main() -> int:
    if not DEMO_PARQUET.exists():
        print(f"ERROR: {DEMO_PARQUET} not found", file=sys.stderr)
        return 2

    X, y_true, _, feat_names = load_test_data(DEMO_PARQUET)
    print(f"[regen-demo-preds] loaded {len(X)} demo samples × {len(feat_names)} features")

    thresholds = get_track_a_thresholds()

    for name, pkl_path in MODELS.items():
        if not pkl_path.exists():
            print(f"  SKIP {name}: pickle not found at {pkl_path}", file=sys.stderr)
            continue
        obj = loads_signed(pkl_path)
        clf = obj.named_steps["classifier"] if hasattr(obj, "named_steps") else obj
        y_proba = clf.predict_proba(X)[:, 1]
        thr = float(thresholds.get(name, 0.5))
        y_pred = (y_proba >= thr).astype(np.int64)
        out = PROJECT_ROOT / f"results/models/{name}_demo_predictions.npz"
        np.savez(out, y_true=y_true.astype(np.int64),
                 y_pred=y_pred, y_proba=y_proba.astype(np.float64))
        n_flagged = int(y_pred.sum())
        print(f"  ✓ {name:<15s} → {out.relative_to(PROJECT_ROOT)}  "
              f"(threshold={thr:.4f}, flagged={n_flagged}/{len(X)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
