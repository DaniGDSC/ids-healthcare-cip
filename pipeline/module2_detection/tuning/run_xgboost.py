#!/usr/bin/env python3
"""Run XGBoost fine-tuning pipeline.

Loads Phase 1 preprocessed data, runs RandomizedSearchCV with SMOTE-in-CV,
optimizes the decision threshold on attack-class F2, evaluates on the
held-out test set, and persists all artifacts.

Usage:
    python src/phase2_5_fine_tuning/run_xgboost.py \
        --train-parquet data/processed/train_phase1.parquet \
        --test-parquet data/processed/test_phase1.parquet \
        --output-dir data/phase2/xgboost \
        --n-iter 50 \
        --cv-folds 5 \
        --scoring f1_weighted \
        --random-state 42
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

# Direct execution fix: add project root to sys.path so the absolute
# import below resolves when this script is invoked directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from pipeline.common import dumps_signed
from pipeline.module2_detection.models.XGBoost import XGBoostDetector
from pipeline.module2_detection.tuning._data import load_data

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


# ── Main ─────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="XGBoost fine-tuning for IoMT intrusion detection",
    )
    parser.add_argument(
        "--train-parquet",
        default="data/processed/train_phase1.parquet",
        help="Path to training parquet (relative to project root)",
    )
    parser.add_argument(
        "--test-parquet",
        default="data/processed/test_phase1.parquet",
        help="Path to test parquet (relative to project root)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/phase2/xgboost",
        help="Output directory for artifacts (relative to project root)",
    )
    parser.add_argument(
        "--n-iter", type=int, default=50,
        help="Number of random HP samples (default: 50)",
    )
    parser.add_argument(
        "--cv-folds", type=int, default=5,
        help="CV folds for RandomizedSearchCV (default: 5)",
    )
    parser.add_argument(
        "--scoring", default="f1_weighted",
        help="Scoring metric for CV (default: f1_weighted)",
    )
    parser.add_argument(
        "--random-state", type=int, default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    t0 = time.perf_counter()
    sep = "=" * 72

    logger.info(sep)
    logger.info("PHASE 2 — XGBOOST FINE-TUNING")
    logger.info(sep)

    # ── Load data ──
    train_path = PROJECT_ROOT / args.train_parquet
    test_path = PROJECT_ROOT / args.test_parquet
    X_train, X_test, y_train, y_test, feat_names = load_data(
        train_path, test_path,
    )

    # ── Train ──
    logger.info("")
    logger.info("── RandomizedSearchCV (n_iter=%d, cv=%d, scoring=%s) ──",
                args.n_iter, args.cv_folds, args.scoring)

    detector = XGBoostDetector(
        n_iter=args.n_iter,
        cv_folds=args.cv_folds,
        scoring=args.scoring,
        random_state=args.random_state,
    )
    detector.fit(X_train, y_train)

    # ── Evaluate ──
    logger.info("")
    logger.info("── Test Set Evaluation ──")
    test_metrics = detector.evaluate(X_test, y_test)

    # ── Save artifacts ──
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Best classifier — bare estimator, NOT the SMOTE wrapper.
    # SMOTE is training-only and bloats the deserialiser surface
    # (finding #15). The persisted artefact is signed via the Module 5
    # ECDSA key so downstream consumers refuse to deserialise tampered
    # files (finding #3a).
    pipeline_path = output_dir / "best_pipeline.pkl"
    classifier_only = detector.pipeline.named_steps["classifier"]
    dumps_signed(classifier_only, pipeline_path)
    logger.info("Saved signed classifier: %s", pipeline_path)

    # 2. Full report
    report = detector.get_report()
    report["data"] = {
        "train_parquet": str(train_path),
        "test_parquet": str(test_path),
        "n_features": len(feat_names),
        "feature_names": feat_names,
        "train_samples": int(len(y_train)),
        "test_samples": int(len(y_test)),
        "train_attack_rate": round(float(y_train.mean()), 4),
        "test_attack_rate": round(float(y_test.mean()), 4),
    }
    report["elapsed_seconds"] = round(time.perf_counter() - t0, 1)

    report_path = output_dir / "xgboost_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8",
    )
    logger.info("Saved report: %s", report_path)

    # 3. Best hyperparameters (standalone for easy loading)
    params_path = output_dir / "best_params.json"
    params_path.write_text(
        json.dumps(detector.best_params, indent=2), encoding="utf-8",
    )
    logger.info("Saved best params: %s", params_path)

    # 4. Test predictions — single forward pass (T-2)
    y_proba = detector.predict_proba(X_test)
    y_pred = (y_proba >= detector.optimal_threshold).astype(int)
    preds_path = output_dir / "test_predictions.npz"
    np.savez(
        preds_path,
        y_true=y_test,
        y_pred=y_pred,
        y_proba=y_proba,
    )
    logger.info("Saved predictions: %s", preds_path)

    # ── Summary ──
    logger.info("")
    logger.info(sep)
    logger.info("XGBOOST FINE-TUNING SUMMARY")
    logger.info(sep)
    logger.info("  Features       : %d", len(feat_names))
    logger.info("  HP candidates  : %d (CV=%d folds)", args.n_iter, args.cv_folds)
    logger.info("  Best CV score  : %.4f (%s)", report["cv_results"]["best_score"], args.scoring)
    logger.info("  Threshold      : %.3f (F2-optimized)", detector.optimal_threshold)
    logger.info("  Test attack F1 : %.4f", test_metrics["attack_f1"])
    logger.info("  Test attack F2 : %.4f", test_metrics["attack_f2"])
    logger.info("  Test AUC-ROC   : %.4f", test_metrics["auc_roc"])
    logger.info("  Test macro F1  : %.4f", test_metrics["macro_f1"])
    logger.info("  Elapsed        : %.1f s", report["elapsed_seconds"])
    logger.info("  Artifacts      : %s", output_dir)
    logger.info(sep)


if __name__ == "__main__":
    main()
