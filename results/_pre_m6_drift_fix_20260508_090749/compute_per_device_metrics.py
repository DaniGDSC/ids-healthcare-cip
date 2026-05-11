"""Per-device-class FNR/FPR/recall with Wilson 95% CIs.

Closes acceptance criterion AC-5 of results/reports/track_a_performance.yaml
(per-device-class metrics) by joining:

  - data/processed/test_phase1.parquet            (row features + Label)
  - results/models/xgboost_test_predictions.npz   (y_pred, y_proba per row)
  - module6_evaluation._derive_device_class()     (biometric-feature → class)

When the test parquet carries a `device_class` column directly (full
GAP-PB-1 closure), that column wins over the biometric heuristic. Output
schema:

  {
    "<device_class>": {
      "n_total": int, "n_attacks": int, "n_benign": int,
      "tp": int, "fn": int, "fp": int, "tn": int,
      "fnr": float, "fpr": float, "recall": float,
      "fnr_95_ci_wilson": [float, float],
      "fpr_95_ci_wilson": [float, float],
    },
    ...
  }
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from module6_evaluation.module6_evaluation import _derive_device_class  # noqa: E402

logger = logging.getLogger(__name__)


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval. Returns (low, high) in [0, 1]."""
    if n == 0:
        return (0.0, 0.0)
    p_hat = k / n
    denom = 1 + z * z / n
    centre = (p_hat + z * z / (2 * n)) / denom
    half = z * math.sqrt((p_hat * (1 - p_hat) + z * z / (4 * n)) / n) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def attach_device_class(df: pd.DataFrame) -> pd.DataFrame:
    """Add a `device_class` column to df.

    Prefers the parquet's authoritative column (GAP-A7 closure path —
    Module 1 writes it). Falls back to the shared biometric heuristic in
    common.device_class when the parquet predates the schema upgrade.
    """
    if "device_class" in df.columns:
        logger.info("Using device_class column from parquet (GAP-A7 closure path).")
        return df

    from common.device_class import derive_device_class_array
    logger.info("device_class absent from parquet — deriving from biometric "
                "features (GAP-A7 fallback; re-run Module 1 for authoritative).")
    df = df.copy()
    feat_names = [c for c in df.columns
                  if c not in ("Label", "Attack Category", "row_id")]
    df["device_class"] = derive_device_class_array(df[feat_names].values, feat_names)
    return df


def compute_per_device(df: pd.DataFrame, y_pred: np.ndarray) -> dict:
    """Per-device-class confusion matrix + Wilson CIs."""
    assert len(df) == len(y_pred), "Row count mismatch between parquet and predictions"

    out = {}
    y_true = df["Label"].values
    classes = sorted(df["device_class"].unique())

    for dc in classes:
        mask = (df["device_class"] == dc).values
        yt = y_true[mask]
        yp = y_pred[mask]

        tp = int(((yt == 1) & (yp == 1)).sum())
        fn = int(((yt == 1) & (yp == 0)).sum())
        fp = int(((yt == 0) & (yp == 1)).sum())
        tn = int(((yt == 0) & (yp == 0)).sum())

        n_attacks = tp + fn
        n_benign = fp + tn

        fnr = (fn / n_attacks) if n_attacks else 0.0
        fpr = (fp / n_benign) if n_benign else 0.0
        recall = (tp / n_attacks) if n_attacks else 0.0

        fnr_lo, fnr_hi = wilson_ci(fn, n_attacks)
        fpr_lo, fpr_hi = wilson_ci(fp, n_benign)

        out[dc] = {
            "n_total": int(mask.sum()),
            "n_attacks": n_attacks,
            "n_benign": n_benign,
            "tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "fnr": round(fnr, 4),
            "fpr": round(fpr, 4),
            "recall": round(recall, 4),
            "fnr_95_ci_wilson": [round(fnr_lo, 4), round(fnr_hi, 4)],
            "fpr_95_ci_wilson": [round(fpr_lo, 4), round(fpr_hi, 4)],
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute per-device-class metrics with Wilson 95% CIs.",
    )
    parser.add_argument(
        "--predictions",
        default="results/models/xgboost_test_predictions.npz",
        help="Path to model predictions NPZ (default: XGBoost).",
    )
    parser.add_argument(
        "--out", default="results/reports/per_device_metrics.json",
        help="Output JSON path.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    pred_path = PROJECT_ROOT / args.predictions
    test_path = PROJECT_ROOT / "data/processed/test_phase1.parquet"
    out_path = PROJECT_ROOT / args.out

    if not pred_path.exists():
        raise FileNotFoundError(
            f"{pred_path} not found. Re-run "
            "`python module2_detection/module2_train_models.py` first."
        )

    preds = dict(np.load(pred_path, allow_pickle=True))
    df = pd.read_parquet(test_path)

    if "row_id" in preds and "row_id" in df.columns:
        df = df.set_index("row_id").loc[preds["row_id"]].reset_index()
        logger.info("Joined on row_id (real GAP-PB-1 closure).")
    else:
        if len(df) != len(preds["y_pred"]):
            raise ValueError(
                f"Row count mismatch: parquet={len(df)}, "
                f"predictions={len(preds['y_pred'])}. "
                "Re-run Module 1 + Module 2 to align."
            )
        logger.info("Joining on positional index (GAP-PB-1 fallback; "
                    "row_id missing from parquet or predictions).")

    df = attach_device_class(df)
    metrics = compute_per_device(df, preds["y_pred"])

    payload = {
        "model": pred_path.stem.replace("_test_predictions", ""),
        "n_rows": int(len(df)),
        "device_class_source": (
            "parquet column" if "device_class" in df.columns
            and "device_class" in pd.read_parquet(test_path).columns
            else "biometric-heuristic fallback (module6_evaluation._derive_device_class)"
        ),
        "ci_method": "Wilson score interval, z=1.96",
        "per_device_class": metrics,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    logger.info("Wrote %s", out_path.relative_to(PROJECT_ROOT))

    logger.info("")
    logger.info("Per-device-class summary:")
    logger.info("%-18s %8s %8s %8s %8s   %s", "device_class", "n_tot", "attacks",
                "FNR", "FPR", "FNR 95% CI")
    for dc, m in sorted(metrics.items(), key=lambda kv: -kv[1]["n_attacks"]):
        logger.info(
            "%-18s %8d %8d %8.4f %8.4f   [%.4f, %.4f]",
            dc, m["n_total"], m["n_attacks"], m["fnr"], m["fpr"],
            *m["fnr_95_ci_wilson"],
        )


if __name__ == "__main__":
    main()
