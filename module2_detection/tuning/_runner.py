"""Shared CLI runner for Track A tuning scripts.

The three Track A run scripts (``run_xgboost.py``, ``run_random_forest.py``,
``run_decision_tree.py``) used to duplicate ~150 LOC of argparse, fit,
evaluate, save-artefacts, log-summary boilerplate each. This module
collapses all of that into one ``run_track_a_tuning()`` function — each
concrete script now only declares its detector class, default n_iter,
and output directory name.

Artefact contract (preserved across the refactor):
  - ``<output_dir>/best_pipeline.pkl`` — signed bare classifier
  - ``<output_dir>/<report_filename>.json`` — full report dict
  - ``<output_dir>/best_params.json`` — hyperparameters only
  - ``<output_dir>/test_predictions.npz`` — y_true / y_pred / y_proba
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

# Make project root importable when invoked directly.
_PROJECT_ROOT_FOR_IMPORT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT_FOR_IMPORT))

from common import dumps_signed  # noqa: E402
from module2_detection.models._base_detector import BaseTrackADetector  # noqa: E402
from module2_detection.tuning._data import load_data  # noqa: E402

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

_SEP = "=" * 72


def _build_parser(
    *,
    description: str,
    default_output_subdir: str,
    default_n_iter: int,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--train-parquet",
        default="data/processed/train_phase1.parquet",
    )
    parser.add_argument(
        "--test-parquet",
        default="data/processed/test_phase1.parquet",
    )
    parser.add_argument(
        "--output-dir",
        default=f"data/phase2/{default_output_subdir}",
    )
    parser.add_argument("--n-iter", type=int, default=default_n_iter)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--scoring", default="f1_weighted")
    parser.add_argument("--random-state", type=int, default=42)
    return parser


def _save_artefacts(
    *,
    detector: BaseTrackADetector,
    test_metrics: dict,
    output_dir: Path,
    train_path: Path,
    test_path: Path,
    feat_names: list,
    y_train: np.ndarray,
    y_test: np.ndarray,
    X_test: np.ndarray,
    report_filename: str,
    elapsed_seconds: float,
    random_state: int,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Best classifier — bare estimator, NOT the SMOTE wrapper.
    # SMOTE is training-only and bloats the deserialiser surface
    # (finding #15). Signed via the Module 5 ECDSA key so downstream
    # consumers refuse to deserialise tampered files (finding #3a).
    pipeline_path = output_dir / "best_pipeline.pkl"
    classifier_only = detector.pipeline.named_steps["classifier"]
    dumps_signed(classifier_only, pipeline_path)
    logger.info("Saved signed classifier: %s", pipeline_path)

    # 2. Full report — strict JSON serialisation (no default=str coercion).
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
        "random_state": int(random_state),
    }
    report["elapsed_seconds"] = round(elapsed_seconds, 1)

    report_path = output_dir / report_filename
    try:
        report_payload = json.dumps(report, indent=2)
    except TypeError as exc:
        # Same strictness as PreprocessingExporter — non-JSON-serialisable
        # values are a producer bug, not something to coerce silently.
        raise TypeError(
            f"{report_filename} contains a non-JSON-serialisable value "
            f"(detail: {exc}). Fix the producer."
        ) from exc
    report_path.write_text(report_payload, encoding="utf-8")
    logger.info("Saved report: %s", report_path)

    # 3. Best hyperparameters (standalone, easy to load downstream).
    params_path = output_dir / "best_params.json"
    params_path.write_text(
        json.dumps(detector.best_params, indent=2), encoding="utf-8",
    )
    logger.info("Saved best params: %s", params_path)

    # 4. Test predictions — single forward pass (T-2).
    y_proba = detector.predict_proba(X_test)
    y_pred = (y_proba >= detector.optimal_threshold).astype(int)
    preds_path = output_dir / "test_predictions.npz"
    np.savez(preds_path, y_true=y_test, y_pred=y_pred, y_proba=y_proba)
    logger.info("Saved predictions: %s", preds_path)

    return report


def _log_summary(
    *,
    log_name: str,
    feat_names: list,
    n_iter: int,
    cv_folds: int,
    scoring: str,
    report: dict,
    detector: BaseTrackADetector,
    test_metrics: dict,
    output_dir: Path,
) -> None:
    logger.info("")
    logger.info(_SEP)
    logger.info("%s FINE-TUNING SUMMARY", log_name.upper())
    logger.info(_SEP)
    logger.info("  Features       : %d", len(feat_names))
    logger.info("  HP candidates  : %d (CV=%d folds)", n_iter, cv_folds)
    logger.info(
        "  Best CV score  : %.4f (%s)",
        report["cv_results"]["best_score"],
        scoring,
    )
    logger.info(
        "  Threshold      : %.3f (F2-optimized)", detector.optimal_threshold,
    )
    logger.info("  Test attack F1 : %.4f", test_metrics["attack_f1"])
    logger.info("  Test attack F2 : %.4f", test_metrics["attack_f2"])
    logger.info("  Test AUC-ROC   : %.4f", test_metrics["auc_roc"])
    logger.info("  Test macro F1  : %.4f", test_metrics["macro_f1"])
    logger.info("  Elapsed        : %.1f s", report["elapsed_seconds"])
    logger.info("  Artifacts      : %s", output_dir)
    logger.info(_SEP)


def run_track_a_tuning(
    *,
    detector_class: Type[BaseTrackADetector],
    output_subdir: str,
    report_filename: str,
    description: str,
    default_n_iter: int,
    argv: list[str] | None = None,
) -> None:
    """Shared end-to-end tuning runner.

    Args:
        detector_class:  Concrete ``BaseTrackADetector`` subclass to tune.
        output_subdir:   Name under ``data/phase2/`` for artefacts.
        report_filename: Filename of the JSON report
            (e.g. ``"xgboost_report.json"``).
        description:     argparse description string.
        default_n_iter:  Default n_iter for RandomizedSearchCV.
        argv:            Optional argv for testing.
    """
    parser = _build_parser(
        description=description,
        default_output_subdir=output_subdir,
        default_n_iter=default_n_iter,
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    t0 = time.perf_counter()

    logger.info(_SEP)
    logger.info("PHASE 2 — %s FINE-TUNING", detector_class.LOG_NAME.upper())
    logger.info(_SEP)

    # ── Load data ──
    train_path = PROJECT_ROOT / args.train_parquet
    test_path = PROJECT_ROOT / args.test_parquet
    X_train, X_test, y_train, y_test, feat_names = load_data(
        train_path, test_path,
    )

    # ── Train ──
    logger.info("")
    logger.info(
        "── RandomizedSearchCV (n_iter=%d, cv=%d, scoring=%s) ──",
        args.n_iter, args.cv_folds, args.scoring,
    )

    detector = detector_class(
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

    # ── Save artefacts ──
    output_dir = PROJECT_ROOT / args.output_dir
    elapsed = time.perf_counter() - t0
    report = _save_artefacts(
        detector=detector,
        test_metrics=test_metrics,
        output_dir=output_dir,
        train_path=train_path,
        test_path=test_path,
        feat_names=feat_names,
        y_train=y_train,
        y_test=y_test,
        X_test=X_test,
        report_filename=report_filename,
        elapsed_seconds=elapsed,
        random_state=args.random_state,
    )

    _log_summary(
        log_name=detector_class.LOG_NAME,
        feat_names=feat_names,
        n_iter=args.n_iter,
        cv_folds=args.cv_folds,
        scoring=args.scoring,
        report=report,
        detector=detector,
        test_metrics=test_metrics,
        output_dir=output_dir,
    )
