
#!/usr/bin/env python3
"""Train final models with best hyperparameters — no more tuning.

Retrains each model on the full training set using the best
hyperparameters found during CV tuning (Phase 2.5):
  - Track A (XGBoost, RF, DT): SMOTE-balanced full training set
  - Track B (DAE): full benign-only training set

Artifacts are saved to data/phase2/{model}/final/.

Usage:
    python train_final_models.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    fbeta_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.tree import DecisionTreeClassifier

# Project root needs to be on sys.path so the absolute import below
# resolves when this script is invoked directly (not via -m).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import dumps_signed
from module2_detection.models._threshold import find_optimal_threshold

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RANDOM_STATE = 42


# ── Data loading ────────────────────────────────────────────────────────

def load_data(label_col: str = "Label") -> tuple:
    """Load Phase 1 parquet files, return X/y arrays and feature names."""
    train_path = PROJECT_ROOT / "data/processed/train_phase1.parquet"
    test_path = PROJECT_ROOT / "data/processed/test_phase1.parquet"

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    drop_cols = [c for c in [label_col, "Attack Category", "row_id", "device_class"] if c in train_df.columns]

    y_train = train_df[label_col].values
    y_test = test_df[label_col].values
    X_train = train_df.drop(columns=drop_cols).values.astype(np.float32)
    X_test = test_df.drop(columns=drop_cols).values.astype(np.float32)
    feat_names = [c for c in train_df.columns if c not in drop_cols]

    logger.info(
        "Data: train=%d (benign=%d, attack=%d), test=%d, features=%d",
        len(y_train), (y_train == 0).sum(), (y_train == 1).sum(),
        len(y_test), len(feat_names),
    )
    return X_train, X_test, y_train, y_test, feat_names


# ── Threshold optimization ──────────────────────────────────────────────

# find_optimal_threshold is imported from models._threshold (shared utility).
# Opt-1: precision_recall_curve replaces the O(T×N) Python loop — see _threshold.py.


# ── Evaluate and log ────────────────────────────────────────────────────

def evaluate(
    name: str,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
) -> dict:
    """Compute metrics and log classification report."""
    metrics = {
        "attack_f1": float(f1_score(y_test, y_pred, pos_label=1)),
        "attack_f2": float(fbeta_score(y_test, y_pred, beta=2, pos_label=1)),
        "weighted_f1": float(f1_score(y_test, y_pred, average="weighted")),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro")),
        "auc_roc": float(roc_auc_score(y_test, y_proba)),
        "optimal_threshold": threshold,
    }
    logger.info(
        "%s: attack_f1=%.4f  attack_f2=%.4f  AUC=%.4f  threshold=%.3f",
        name, metrics["attack_f1"], metrics["attack_f2"],
        metrics["auc_roc"], threshold,
    )
    logger.info("\n%s", classification_report(
        y_test, y_pred, target_names=["Normal", "Attack"], digits=4,
    ))
    return metrics


# ── Track A: train one supervised model ─────────────────────────────────

TRACK_A_MODELS = {
    "xgboost": {
        "cls": GradientBoostingClassifier,
        "params_file": "results/models/xgboost_best_params.json",
        "output_dir": "results/models",
        "cls_kwargs": {"random_state": RANDOM_STATE},
    },
    "random_forest": {
        "cls": RandomForestClassifier,
        "params_file": "results/models/random_forest_best_params.json",
        "output_dir": "results/models",
        "cls_kwargs": {"random_state": RANDOM_STATE, "n_jobs": -1, "bootstrap": True},
    },
    "decision_tree": {
        "cls": DecisionTreeClassifier,
        "params_file": "results/models/decision_tree_best_params.json",
        "output_dir": "results/models",
        "cls_kwargs": {"random_state": RANDOM_STATE},
    },
}


def strip_prefix(params: dict, prefix: str = "classifier__") -> dict:
    """Remove 'classifier__' prefix from param keys."""
    return {k.replace(prefix, ""): v for k, v in params.items()}


def train_track_a(
    name: str,
    cfg: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feat_names: list,
) -> dict:
    """Train a Track A model with fixed best params on full SMOTE-balanced data."""
    t0 = time.perf_counter()
    sep = "-" * 60

    logger.info(sep)
    logger.info("FINAL TRAINING: %s", name.upper())
    logger.info(sep)

    # Load best params
    params_path = PROJECT_ROOT / cfg["params_file"]
    with open(params_path) as f:
        raw_params = json.load(f)
    clf_params = strip_prefix(raw_params)
    logger.info("Best params: %s", clf_params)

    # Build pipeline: SMOTE + classifier with fixed params
    def _fresh_pipeline() -> ImbPipeline:
        return ImbPipeline([
            ("smote", SMOTE(
                sampling_strategy="auto",
                k_neighbors=5,
                random_state=RANDOM_STATE,
            )),
            ("classifier", cfg["cls"](**cfg["cls_kwargs"], **clf_params)),
        ])

    pipeline = _fresh_pipeline()

    # Fit on full training set
    logger.info("Fitting on full training set (%d samples)...", len(y_train))
    pipeline.fit(X_train, y_train)

    # ── Threshold optimization via OUT-OF-FOLD probabilities ──
    # The previous implementation called
    #     pipeline.predict_proba(X_train)
    # which gives back resubstitution probabilities the model has
    # already memorised — boosting trees and bagged trees on the
    # WUSTL-EHMS data return probas pinned to ~0/~1 on training rows,
    # and the F2-optimal threshold derived from that is meaningless.
    # cross_val_predict on a fresh copy of the same pipeline gives us
    # the probability distribution an unseen sample would actually
    # receive, which is the distribution we should optimise against.
    # See finding #2 in the Phase 2 security review.
    logger.info("Computing out-of-fold probabilities for threshold fit...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof_proba = cross_val_predict(
        _fresh_pipeline(),
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]
    threshold = find_optimal_threshold(y_train, oof_proba)

    # Test evaluation
    y_proba_test = pipeline.predict_proba(X_test)[:, 1]
    y_pred_test = (y_proba_test >= threshold).astype(int)
    metrics = evaluate(name, y_test, y_pred_test, y_proba_test, threshold)

    elapsed = round(time.perf_counter() - t0, 1)

    # Save artifacts
    output_dir = PROJECT_ROOT / cfg["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Persist the FITTED CLASSIFIER ONLY (not the SMOTE wrapper) ──
    # SMOTE is a training-time only transform; serialising it bloats
    # the artefact and adds an unnecessary deserialiser surface
    # (SMOTE carries its own internal NearestNeighbors fit). The
    # downstream consumers (Module 3/4) only call ``predict_proba`` —
    # which lives on the classifier — so they don't need SMOTE.
    # See finding #15 in the Phase 2 security review.
    classifier_only = pipeline.named_steps["classifier"]
    pipeline_path = output_dir / f"{name}_final_pipeline.pkl"
    # ECDSA-signed; verifier in common.signed_pickle refuses
    # to deserialise without a valid signature against the Module 5
    # public key. Closes the pickle-RCE sink in finding #3a.
    dumps_signed(classifier_only, pipeline_path)

    # Report
    report = {
        "model_type": name,
        "stage": "final_training",
        "best_params": clf_params,
        "optimal_threshold": threshold,
        "test_metrics": metrics,
        "data": {
            "n_features": len(feat_names),
            "feature_names": feat_names,
            "train_samples": int(len(y_train)),
            "test_samples": int(len(y_test)),
            "train_attack_rate": round(float(y_train.mean()), 4),
        },
        "elapsed_seconds": elapsed,
    }
    report_path = output_dir / f"{name}_final_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    # Test predictions
    np.savez(
        output_dir / f"{name}_test_predictions.npz",
        y_true=y_test, y_pred=y_pred_test, y_proba=y_proba_test,
    )

    # OOF probabilities for cascaded DAE input
    oof_path = output_dir / f"{name}_oof_proba.npy"
    np.save(oof_path, oof_proba)
    logger.info("Saved OOF probas: %s", oof_path)

    logger.info("Saved: %s (%.1fs)", output_dir, elapsed)
    return metrics


# ── Track B (DAE) ──────────────────────────────────────────────────────
#
# DAE training lives in module2_detection.dae_training so the benign-only
# training step is decoupled from inference-time cascaded scoring (which
# now lives in detection_engine.DetectionEngine). main() composes the
# two below.


def evaluate_dae(npz_path: Path, threshold: float) -> dict:
    """Compute DAE test metrics from the engine-emitted prediction npz.

    Matches the Track A `evaluate()` schema so the summary table and
    downstream consumers can treat all four models uniformly. AUC-ROC is
    computed against `reconstruction_error` (the underlying scalar), F1/F2
    against the engine's threshold decision in `y_pred`.
    """
    data = np.load(npz_path)
    y_test = data["y_true"]
    y_pred = data["y_pred"].astype(int)
    score = data["reconstruction_error"]

    metrics = {
        "attack_f1": float(f1_score(y_test, y_pred, pos_label=1)),
        "attack_f2": float(fbeta_score(y_test, y_pred, beta=2, pos_label=1)),
        "weighted_f1": float(f1_score(y_test, y_pred, average="weighted")),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro")),
        "auc_roc": float(roc_auc_score(y_test, score)),
        "optimal_threshold": float(threshold),
    }
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    metrics["confusion_matrix"] = {
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
    logger.info(
        "dae: attack_f1=%.4f  attack_f2=%.4f  AUC=%.4f  threshold=%.3g",
        metrics["attack_f1"], metrics["attack_f2"],
        metrics["auc_roc"], threshold,
    )
    logger.info("\n%s", classification_report(
        y_test, y_pred, target_names=["Normal", "Attack"], digits=4,
    ))
    return metrics


# ── Main ────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    sep = "=" * 72
    logger.info(sep)
    logger.info("FINAL MODEL TRAINING — FIXED BEST HYPERPARAMETERS")
    logger.info(sep)

    t0 = time.perf_counter()
    X_train, X_test, y_train, y_test, feat_names = load_data()

    all_metrics = {}

    # Track A: supervised classifiers (saves *_oof_proba.npy for the DAE)
    for name, cfg in TRACK_A_MODELS.items():
        metrics = train_track_a(name, cfg, X_train, y_train, X_test, y_test, feat_names)
        all_metrics[name] = metrics

    # Track B: DAE training (benign-only) + test-set scoring via engine.
    # Training only writes the DAE artifact; engine.write_test_predictions
    # produces dae_test_predictions.npz with the cascaded-fusion scores
    # the rest of the pipeline expects.
    from module2_detection.dae_training import train_dae
    from detection_engine import DetectionEngine
    from common.model_registry import invalidate_cache

    dae_summary = train_dae()
    invalidate_cache()  # force re-load of the freshly written DAE artifact
    dae_npz = DetectionEngine().write_test_predictions()

    # Patch the report and merge metrics so the summary table reflects
    # real DAE performance instead of the all-zeros placeholder.
    dae_report_path = PROJECT_ROOT / "results/models/dae_final_report.json"
    dae_report = json.loads(dae_report_path.read_text(encoding="utf-8"))
    dae_metrics = evaluate_dae(dae_npz, dae_report.get("threshold", 0.0))
    dae_report["test_metrics"] = dae_metrics
    dae_report_path.write_text(
        json.dumps(dae_report, indent=2, default=str), encoding="utf-8",
    )
    all_metrics["dae"] = {**dae_summary, **dae_metrics}

    # Final summary
    total = round(time.perf_counter() - t0, 1)
    logger.info("")
    logger.info(sep)
    logger.info("FINAL TRAINING COMPLETE — %.1fs total", total)
    logger.info(sep)
    logger.info("%-16s %10s %10s %10s", "Model", "Attack F1", "Attack F2", "AUC-ROC")
    logger.info("-" * 50)
    for name, m in all_metrics.items():
        f1 = m.get("attack_f1", 0)
        f2 = m.get("attack_f2", 0)
        auc = m.get("auc_roc", 0)
        logger.info("%-16s %10.4f %10.4f %10.4f", name, f1, f2, auc)
    logger.info(sep)


if __name__ == "__main__":
    main()
