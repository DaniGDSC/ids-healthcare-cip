"""Shared fine-tuning runner for Track A detectors.

Eliminates the triplicated ``main()`` previously in ``run_xgboost.py``,
``run_random_forest.py`` and ``run_decision_tree.py``. Each runner now
just calls :func:`run_tuning` with its model class and a few defaults.

The signed-artefact contract (bare classifier saved via
``common.dumps_signed`` — SMOTE wrapper stripped) is preserved bit-identical
to the pre-refactor scripts. See findings #3a, #15.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Type

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from common import dumps_signed
from module2_detection.models.base import BaseDetector
from module2_detection.tuning._data import load_data

logger = logging.getLogger(__name__)


def run_tuning(
    *,
    detector_cls: Type[BaseDetector],
    display_name: str,
    output_subdir: str,
    report_filename: str,
    default_n_iter: int,
) -> None:
    """Generic fine-tuning entry point.

    Args:
        detector_cls: Track A detector subclass (XGBoost / RandomForest / DecisionTree).
        display_name: Banner / summary label, e.g. "XGBOOST".
        output_subdir: ``data/phase2/<output_subdir>`` for artefacts.
        report_filename: Filename for the JSON report inside ``output_subdir``.
        default_n_iter: Default ``--n-iter`` value (was 25/40/50 per detector).
    """
    parser = argparse.ArgumentParser(
        description=f"{display_name.title()} fine-tuning for IoMT intrusion detection",
    )
    parser.add_argument("--train-parquet", default="data/processed/train_phase1.parquet")
    parser.add_argument("--test-parquet", default="data/processed/test_phase1.parquet")
    parser.add_argument("--output-dir", default=f"data/phase2/{output_subdir}")
    parser.add_argument("--n-iter", type=int, default=default_n_iter)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--scoring", default="f1_weighted")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    t0 = time.perf_counter()
    sep = "=" * 72

    logger.info(sep)
    logger.info("PHASE 2 — %s FINE-TUNING", display_name.upper())
    logger.info(sep)

    train_path = PROJECT_ROOT / args.train_parquet
    test_path = PROJECT_ROOT / args.test_parquet
    X_train, X_test, y_train, y_test, feat_names = load_data(train_path, test_path)

    logger.info("")
    logger.info(
        "── RandomizedSearchCV (n_iter=%d, cv=%d, scoring=%s) ──",
        args.n_iter, args.cv_folds, args.scoring,
    )
    detector = detector_cls(
        n_iter=args.n_iter,
        cv_folds=args.cv_folds,
        scoring=args.scoring,
        random_state=args.random_state,
    )
    detector.fit(X_train, y_train)

    logger.info("")
    logger.info("── Test Set Evaluation ──")
    test_metrics = detector.evaluate(X_test, y_test)

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Bare classifier signed via the Module 5 ECDSA key. SMOTE wrapper is
    # intentionally stripped (training-only). See findings #3a, #15.
    pipeline_path = output_dir / "best_pipeline.pkl"
    if detector.pipeline is None:
        raise RuntimeError("detector.pipeline is None after fit() — search failed")
    classifier_only = detector.pipeline.named_steps["classifier"]
    dumps_signed(classifier_only, pipeline_path)
    logger.info("Saved signed classifier: %s", pipeline_path)

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

    report_path = output_dir / report_filename
    report_path.write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8",
    )
    logger.info("Saved report: %s", report_path)

    params_path = output_dir / "best_params.json"
    params_path.write_text(
        json.dumps(detector.best_params, indent=2), encoding="utf-8",
    )
    logger.info("Saved best params: %s", params_path)

    # Single forward pass (T-2)
    y_proba = detector.predict_proba(X_test)
    y_pred = (y_proba >= detector.optimal_threshold).astype(int)
    preds_path = output_dir / "test_predictions.npz"
    np.savez(preds_path, y_true=y_test, y_pred=y_pred, y_proba=y_proba)
    logger.info("Saved predictions: %s", preds_path)

    logger.info("")
    logger.info(sep)
    logger.info("%s FINE-TUNING SUMMARY", display_name.upper())
    logger.info(sep)
    logger.info("  Features       : %d", len(feat_names))
    logger.info("  HP candidates  : %d (CV=%d folds)", args.n_iter, args.cv_folds)
    logger.info(
        "  Best CV score  : %.4f (%s)",
        report["cv_results"]["best_score"], args.scoring,
    )
    logger.info("  Threshold      : %.3f (F2-optimized)", detector.optimal_threshold)
    logger.info("  Test attack F1 : %.4f", test_metrics["attack_f1"])
    logger.info("  Test attack F2 : %.4f", test_metrics["attack_f2"])
    logger.info("  Test AUC-ROC   : %.4f", test_metrics["auc_roc"])
    logger.info("  Test macro F1  : %.4f", test_metrics["macro_f1"])
    logger.info("  Elapsed        : %.1f s", report["elapsed_seconds"])
    logger.info("  Artifacts      : %s", output_dir)
    logger.info(sep)
