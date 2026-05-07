"""Track A calibration (Enhancement 1).

Wraps each fitted Track A model (XGBoost surrogate, RandomForest,
DecisionTree) with a post-hoc calibrator fitted on the held-out
validation set, then re-emits calibrated val/test probabilities.

Why post-hoc calibration matters in this cascade
------------------------------------------------
Trees produce **uncalibrated** probabilities by default:
  - GradientBoostingClassifier shrinks probabilities toward 0.5 due to
    the additive logit space.
  - RandomForest produces hard fractions of votes — so e.g. a 4-tree
    majority of 4-of-50 trees gives exactly 0.08, regardless of how
    confidently each tree voted.
  - DecisionTree leaves are nearly one-hot — confidences are 0/1 with
    almost no middle ground.

These artefacts make threshold choices fragile and corrupt the fusion
logic: a row with ``P_xgb = 0.06`` from a poorly-calibrated tree may
actually mean "20% probability the row is an attack" or "1% probability"
depending on the calibration error. Calibration via isotonic regression
or Platt scaling fits a monotone mapping ``raw → calibrated`` on the val
set so downstream thresholds operate on a well-defined probability scale.

Method
------
Default = isotonic (non-parametric, fits any monotone shape, no
distributional assumption). Falls back to Platt (sigmoid) automatically
when the val set has fewer than ~1000 rows because isotonic overfits
on small samples.

Calibrators are persisted as ``*_{name}_calibrator.pkl`` (joblib).
Calibrated probas are persisted as ``*_{name}_val_proba_calibrated.npy``
and ``*_{name}_test_proba_calibrated.npy`` so the binary cascade and
the multi-class cascade can both opt in without modifying their
training scripts.

Run
---
    python module2_detection/calibrate.py            # binary + multi-class
    python module2_detection/calibrate.py --binary   # binary only
    python module2_detection/calibrate.py --multiclass  # multi-class only
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

# Isotonic regression needs ~1000 samples to avoid overfitting; below
# that we fall back to Platt scaling (sigmoid). Threshold from the
# scikit-learn user guide.
_ISOTONIC_MIN_SAMPLES = 1000


from module2_detection._features import drop_non_feature_cols


def _load_split(name: str, label_col: str = "Label") -> tuple:
    df = pd.read_parquet(PROJECT_ROOT / "data/processed" / f"{name}.parquet")
    y = df[label_col].values
    X = drop_non_feature_cols(df).values.astype(np.float32)
    return X, y


def _load_signed_pickle(path: Path):
    """Load the binary signed pickle written by module2_train_models."""
    sys.path.insert(0, str(PROJECT_ROOT))
    from common import loads_signed
    return loads_signed(path)


def _calibrator(method: str, n_val_rows: int) -> str:
    if method == "auto":
        return "isotonic" if n_val_rows >= _ISOTONIC_MIN_SAMPLES else "sigmoid"
    return method


def _calibrate_binary_models(method: str) -> None:
    """Calibrate each binary tree on val; emit *_calibrated artefacts."""
    output_dir = PROJECT_ROOT / "results/models"
    X_val, y_val = _load_split("val_phase1", label_col="Label")
    X_test, y_test = _load_split("test_phase1", label_col="Label")
    cal_method = _calibrator(method, len(X_val))
    logger.info("Binary calibration: method=%s, n_val=%d, n_test=%d",
                cal_method, len(X_val), len(X_test))

    for name in ("xgboost", "random_forest", "decision_tree"):
        pkl = output_dir / f"{name}_final_pipeline.pkl"
        if not pkl.exists():
            logger.warning("  %s: no fitted model at %s — skipping", name, pkl)
            continue
        clf = _load_signed_pickle(pkl)

        # Prefit calibration: wrap the already-fitted classifier with the
        # `prefit` cv option. CalibratedClassifierCV fits ONLY the
        # calibrator on val; the underlying clf is untouched.
        cal = CalibratedClassifierCV(estimator=clf, method=cal_method, cv="prefit")
        cal.fit(X_val, y_val)

        # Raw vs calibrated diagnostics
        raw_val = clf.predict_proba(X_val)[:, 1]
        cal_val = cal.predict_proba(X_val)[:, 1]
        raw_test = clf.predict_proba(X_test)[:, 1]
        cal_test = cal.predict_proba(X_test)[:, 1]

        diagnostics = {
            "method": cal_method,
            "n_val": int(len(X_val)),
            "raw": {
                "val_brier": float(brier_score_loss(y_val, raw_val)),
                "val_log_loss": float(log_loss(y_val, np.clip(raw_val, 1e-6, 1 - 1e-6))),
                "test_brier": float(brier_score_loss(y_test, raw_test)),
                "test_auc_roc": float(roc_auc_score(y_test, raw_test)),
                "test_auprc": float(average_precision_score(y_test, raw_test)),
            },
            "calibrated": {
                "val_brier": float(brier_score_loss(y_val, cal_val)),
                "val_log_loss": float(log_loss(y_val, np.clip(cal_val, 1e-6, 1 - 1e-6))),
                "test_brier": float(brier_score_loss(y_test, cal_test)),
                "test_auc_roc": float(roc_auc_score(y_test, cal_test)),
                "test_auprc": float(average_precision_score(y_test, cal_test)),
            },
            "improvements": {
                "delta_val_brier": float(brier_score_loss(y_val, raw_val)
                                          - brier_score_loss(y_val, cal_val)),
                "delta_test_brier": float(brier_score_loss(y_test, raw_test)
                                           - brier_score_loss(y_test, cal_test)),
            },
        }

        np.save(output_dir / f"{name}_val_proba_calibrated.npy", cal_val)
        np.save(output_dir / f"{name}_test_proba_calibrated.npy", cal_test)
        joblib.dump(cal, output_dir / f"{name}_calibrator.pkl")
        (output_dir / f"{name}_calibration_report.json").write_text(
            json.dumps(diagnostics, indent=2), encoding="utf-8"
        )

        logger.info(
            "  %-15s val Brier %.4f→%.4f (Δ%+.4f)  test Brier %.4f→%.4f (Δ%+.4f)  test AUC %.4f→%.4f",
            name,
            diagnostics["raw"]["val_brier"],
            diagnostics["calibrated"]["val_brier"],
            diagnostics["improvements"]["delta_val_brier"],
            diagnostics["raw"]["test_brier"],
            diagnostics["calibrated"]["test_brier"],
            diagnostics["improvements"]["delta_test_brier"],
            diagnostics["raw"]["test_auc_roc"],
            diagnostics["calibrated"]["test_auc_roc"],
        )


def _load_multiclass_split(
    name: str, label_col: str, label_order: tuple[str, ...],
) -> tuple:
    df = pd.read_parquet(PROJECT_ROOT / "data/processed" / f"{name}.parquet")
    label_to_id = {s: i for i, s in enumerate(label_order)}
    y = np.array([label_to_id[s] for s in df[label_col].astype(str).values],
                 dtype=np.int64)
    X = drop_non_feature_cols(df).values.astype(np.float32)
    return X, y


def _calibrate_multiclass_models(method: str) -> None:
    """Calibrate each multi-class tree on val; emit *_multiclass_*_calibrated artefacts."""
    from src.data_models import MULTICLASS_LABEL_ORDER_EHMS

    output_dir = PROJECT_ROOT / "results/models"
    label_order = MULTICLASS_LABEL_ORDER_EHMS

    X_val, y_val = _load_multiclass_split(
        "val_phase1", "Attack Category", label_order,
    )
    X_test, y_test = _load_multiclass_split(
        "test_phase1", "Attack Category", label_order,
    )
    cal_method = _calibrator(method, len(X_val))
    logger.info("Multi-class calibration: method=%s, n_val=%d, classes=%s",
                cal_method, len(X_val), label_order)

    for name in ("xgboost", "random_forest", "decision_tree"):
        pkl = output_dir / f"{name}_multiclass_final_pipeline.pkl"
        if not pkl.exists():
            logger.warning("  %s: no fitted multiclass model at %s — skipping",
                           name, pkl)
            continue
        clf = joblib.load(pkl)
        cal = CalibratedClassifierCV(estimator=clf, method=cal_method, cv="prefit")
        cal.fit(X_val, y_val)

        raw_val = clf.predict_proba(X_val)
        cal_val = cal.predict_proba(X_val)
        raw_test = clf.predict_proba(X_test)
        cal_test = cal.predict_proba(X_test)

        # Multi-class Brier (one-hot encoded)
        n_classes = len(label_order)
        y_val_oh = np.eye(n_classes)[y_val]
        y_test_oh = np.eye(n_classes)[y_test]
        raw_val_brier = float(np.mean((raw_val - y_val_oh) ** 2))
        cal_val_brier = float(np.mean((cal_val - y_val_oh) ** 2))
        raw_test_brier = float(np.mean((raw_test - y_test_oh) ** 2))
        cal_test_brier = float(np.mean((cal_test - y_test_oh) ** 2))

        # P(attack) = 1 - P(normal); compare to binary y_val/y_test
        normal_idx = label_order.index("normal") if "normal" in label_order else 0
        raw_val_pa = 1.0 - raw_val[:, normal_idx]
        cal_val_pa = 1.0 - cal_val[:, normal_idx]
        raw_test_pa = 1.0 - raw_test[:, normal_idx]
        cal_test_pa = 1.0 - cal_test[:, normal_idx]
        bin_y_val = (y_val != normal_idx).astype(int)
        bin_y_test = (y_test != normal_idx).astype(int)

        diagnostics = {
            "method": cal_method,
            "label_order": list(label_order),
            "n_val": int(len(X_val)),
            "raw": {
                "val_multiclass_brier": raw_val_brier,
                "test_multiclass_brier": raw_test_brier,
                "test_p_attack_auc_roc": float(roc_auc_score(bin_y_test, raw_test_pa)),
                "test_p_attack_brier": float(brier_score_loss(bin_y_test, raw_test_pa)),
            },
            "calibrated": {
                "val_multiclass_brier": cal_val_brier,
                "test_multiclass_brier": cal_test_brier,
                "test_p_attack_auc_roc": float(roc_auc_score(bin_y_test, cal_test_pa)),
                "test_p_attack_brier": float(brier_score_loss(bin_y_test, cal_test_pa)),
            },
            "improvements": {
                "delta_val_multiclass_brier": raw_val_brier - cal_val_brier,
                "delta_test_multiclass_brier": raw_test_brier - cal_test_brier,
            },
        }

        np.save(output_dir / f"{name}_multiclass_val_proba_calibrated.npy", cal_val)
        np.save(output_dir / f"{name}_multiclass_test_proba_calibrated.npy", cal_test)
        joblib.dump(cal, output_dir / f"{name}_multiclass_calibrator.pkl")
        (output_dir / f"{name}_multiclass_calibration_report.json").write_text(
            json.dumps(diagnostics, indent=2), encoding="utf-8"
        )

        logger.info(
            "  %-15s mc-Brier val %.4f→%.4f (Δ%+.4f)  test %.4f→%.4f (Δ%+.4f)  test P(atk) AUC %.4f→%.4f",
            name,
            raw_val_brier, cal_val_brier,
            raw_val_brier - cal_val_brier,
            raw_test_brier, cal_test_brier,
            raw_test_brier - cal_test_brier,
            diagnostics["raw"]["test_p_attack_auc_roc"],
            diagnostics["calibrated"]["test_p_attack_auc_roc"],
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Post-hoc calibration of Track A trees (Enhancement 1)",
    )
    parser.add_argument("--binary", action="store_true",
                        help="Calibrate binary models only")
    parser.add_argument("--multiclass", action="store_true",
                        help="Calibrate multi-class models only")
    parser.add_argument("--method", default="auto",
                        choices=("auto", "isotonic", "sigmoid"),
                        help="Calibration method (default: auto — isotonic "
                             "if n_val>=1000, else sigmoid)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    sep = "=" * 72

    do_binary = args.binary or not (args.binary or args.multiclass)
    do_mc = args.multiclass or not (args.binary or args.multiclass)

    if do_binary:
        logger.info(sep)
        logger.info("ENHANCEMENT 1 — BINARY TRACK A CALIBRATION")
        logger.info(sep)
        _calibrate_binary_models(args.method)

    if do_mc:
        logger.info(sep)
        logger.info("ENHANCEMENT 1 — MULTI-CLASS TRACK A CALIBRATION")
        logger.info(sep)
        _calibrate_multiclass_models(args.method)

    return 0


if __name__ == "__main__":
    sys.exit(main())
