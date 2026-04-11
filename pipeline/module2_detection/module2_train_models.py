
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
    f1_score,
    fbeta_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.tree import DecisionTreeClassifier

# Project root needs to be on sys.path so the absolute import below
# resolves when this script is invoked directly (not via -m).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from pipeline.common import dumps_signed
from pipeline.module2_detection.models.DAE import DAEDetector

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

RANDOM_STATE = 42


# ── Data loading ────────────────────────────────────────────────────────

def load_data(label_col: str = "Label") -> tuple:
    """Load Phase 1 parquet files, return X/y arrays and feature names."""
    train_path = PROJECT_ROOT / "data/processed/train_phase1.parquet"
    test_path = PROJECT_ROOT / "data/processed/test_phase1.parquet"

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    drop_cols = [c for c in [label_col, "Attack Category"] if c in train_df.columns]

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

def find_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    beta: float = 2.0,
    n_thresholds: int = 200,
) -> float:
    """Find threshold that maximizes F-beta on attack class."""
    thresholds = np.linspace(0.05, 0.95, n_thresholds)
    best_score, best_t = 0.0, 0.5
    for t in thresholds:
        score = fbeta_score(y_true, (y_proba >= t).astype(int), beta=beta, pos_label=1)
        if score > best_score:
            best_score, best_t = score, float(t)
    return best_t


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
    # ECDSA-signed; verifier in pipeline.common.signed_pickle refuses
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

    logger.info("Saved: %s (%.1fs)", output_dir, elapsed)
    return metrics


# ── Track B: DAE ────────────────────────────────────────────────────────

def train_track_b_dae(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feat_names: list,
) -> dict:
    """Train DAE on full benign-only training set with best params."""
    t0 = time.perf_counter()
    sep = "-" * 60

    logger.info(sep)
    logger.info("FINAL TRAINING: DAE (TRACK B)")
    logger.info(sep)

    # Load best params
    params_path = PROJECT_ROOT / "results/models/dae_best_params.json"
    with open(params_path) as f:
        best_hp = json.load(f)
    logger.info("Best params: %s", best_hp)

    # Benign-only subset
    X_benign = X_train[y_train == 0]
    logger.info("Fitting on full benign training set (%d samples)...", len(X_benign))

    # Train with NO validation split — use all benign data for final model
    det = DAEDetector(
        **best_hp,
        epochs=100,
        batch_size=256,
        random_state=RANDOM_STATE,
    )
    det.fit(X_benign, validation_split=0.0)

    # Evaluate
    test_metrics = det.evaluate(X_test, y_test)

    elapsed = round(time.perf_counter() - t0, 1)

    # Save artifacts
    output_dir = PROJECT_ROOT / "results/models"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Native (pickle-free) artefacts ──
    # Canonical persistence: JSON sidecar + Keras weights file. Loading
    # via DAEDetector.from_artefacts() never executes Python. This is
    # the format Module 3/4 will eventually load exclusively. See
    # Phase 2 finding #3b.
    det.save_artefacts(
        json_path=output_dir / "dae_detector.json",
        weights_path=output_dir / "dae_model.weights.h5",
    )

    # Report
    report = det.get_report()
    report["stage"] = "final_training"
    report["best_hyperparameters"] = best_hp
    report["data"] = {
        "n_features": len(feat_names),
        "feature_names": feat_names,
        "benign_train_samples": int((y_train == 0).sum()),
        "test_samples": int(len(y_test)),
    }
    report["elapsed_seconds"] = elapsed

    report_path = output_dir / "dae_final_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    # Test predictions
    y_pred = det.predict(X_test)
    errors = det.reconstruction_error(X_test)
    np.savez(
        output_dir / "dae_test_predictions.npz",
        y_true=y_test, y_pred=y_pred, reconstruction_error=errors,
    )

    # NOTE: there is no longer a dae_detector.pkl on the load path —
    # the canonical artefact pair is dae_detector.json + dae_model.weights.h5
    # written above by det.save_artefacts(). All Module 3/4 callers
    # were updated to load via DAEDetector.from_artefacts(). Removing
    # the legacy pickle eliminates the highest-blast-radius pickle RCE
    # sink in the production inference path. See Phase 2 finding #3b.

    logger.info("Saved: %s (%.1fs)", output_dir, elapsed)
    return test_metrics


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

    # Track A models
    for name, cfg in TRACK_A_MODELS.items():
        metrics = train_track_a(name, cfg, X_train, y_train, X_test, y_test, feat_names)
        all_metrics[name] = metrics

    # Track B: DAE
    dae_metrics = train_track_b_dae(X_train, y_train, X_test, y_test, feat_names)
    all_metrics["dae"] = dae_metrics

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
