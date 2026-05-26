#!/usr/bin/env python3
"""Train final detectors with best hyperparameters — no more tuning.

Retrains each model on the full training set using the best
hyperparameters found during CV tuning (Phase 2.5):
  - Track A (GradientBoosting/XGBoost-equivalent, RandomForest,
    DecisionTree): SMOTE-balanced full training set
  - Track B (DAE): full benign-only training set

Artifacts are saved to ``results/models/`` (Track A) and via the
DAE training module (Track B).

Usage:
    # Train end-to-end (Track A + Track B + cascaded test eval):
    python -m module2_detection.module2_train_models

    # Re-emit predictions on a frozen split without re-training:
    python -m module2_detection.module2_train_models --predict-only --split test
    python -m module2_detection.module2_train_models --predict-only --split demo
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

# Default seed when no tuning report is found. Real runs propagate the
# random_state from the tuning report's metadata via _resolve_random_state
# so the final-training stage uses the same seed the hyperparameters were
# chosen under — see finding #17 lineage.
DEFAULT_RANDOM_STATE = 42
RANDOM_STATE = DEFAULT_RANDOM_STATE  # legacy alias for any external consumer


def _resolve_random_state(params_file: Path) -> int:
    """Read random_state from the tuning report next to ``params_file``.

    The runner writes ``data.random_state`` into the tuning report. We
    look up that value so the final-fit seed matches the seed under
    which the hyperparameters were selected; otherwise random_state
    drift between tuning and final training breaks reproducibility.

    Falls back to DEFAULT_RANDOM_STATE if the report is missing or
    doesn't carry the seed (e.g. legacy tuning artefacts pre-Y6).
    """
    report_path = (
        params_file.parent
        / f"{params_file.stem.replace('_best_params', '')}_report.json"
    )
    # Also try the canonical report filenames produced by the runner
    candidate_reports = [
        report_path,
        params_file.parent / "xgboost_report.json",
        params_file.parent / "random_forest_report.json",
        params_file.parent / "decision_tree_report.json",
    ]
    for cand in candidate_reports:
        if cand.exists():
            try:
                data = json.loads(cand.read_text())
                seed = data.get("data", {}).get("random_state")
                if isinstance(seed, int):
                    return seed
            except (json.JSONDecodeError, OSError):
                continue
    return DEFAULT_RANDOM_STATE


# ── Data loading ────────────────────────────────────────────────────────
# The leakage guard + canonical load_data implementation live in
# `module2_detection.tuning._data`; we re-export them here so all
# training-side code paths share one source of truth AND so tests that
# import from `module2_train_models` for backward-compat keep working.
from module2_detection.tuning._data import (  # noqa: E402,F401
    _FORBIDDEN_TRAINING_PARQUETS,
    _assert_no_demo_leakage,
    load_data as _load_data_shared,
)


def load_data(label_col: str = "Label") -> tuple:
    """Load Phase 1 train + test parquets at default paths.

    Thin wrapper around ``tuning._data.load_data`` that pins the paths
    to the canonical Phase 1 outputs. Use the shared loader directly if
    you need to specify alternative paths (e.g. CI fixtures).
    """
    return _load_data_shared(
        PROJECT_ROOT / "data/processed/train_phase1.parquet",
        PROJECT_ROOT / "data/processed/test_phase1.parquet",
        label_col=label_col,
    )


def load_split_data(split: str, label_col: str = "Label") -> tuple:
    """Load a single labelled split parquet → (X, y, feat_names).

    Used by ``--predict-only`` and by detection_engine.write_predictions
    to score a frozen split (test = paper-clean; demo = operator-clean).
    Inference-side: demo is explicitly allowed here, unlike load_data().
    """
    if split not in ("test", "demo"):
        raise ValueError(f"unknown split: {split!r} (expected 'test' or 'demo')")

    path = PROJECT_ROOT / f"data/processed/{split}_phase1.parquet"
    df = pd.read_parquet(path)
    drop_cols = [
        c
        for c in [label_col, "Attack Category", "row_id", "device_class"]
        if c in df.columns
    ]
    y = df[label_col].values
    X = df.drop(columns=drop_cols).values.astype(np.float32)
    feat_names = [c for c in df.columns if c not in drop_cols]
    logger.info(
        "Split %s: %d samples (benign=%d, attack=%d), %d features",
        split,
        len(y),
        (y == 0).sum(),
        (y == 1).sum(),
        len(feat_names),
    )
    return X, y, feat_names


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
        name,
        metrics["attack_f1"],
        metrics["attack_f2"],
        metrics["auc_roc"],
        threshold,
    )
    logger.info(
        "\n%s",
        classification_report(
            y_test,
            y_pred,
            target_names=["Normal", "Attack"],
            digits=4,
        ),
    )
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

    # Load best params + resolve the random_state that was used to pick them.
    params_path = PROJECT_ROOT / cfg["params_file"]
    with open(params_path) as f:
        raw_params = json.load(f)
    clf_params = strip_prefix(raw_params)
    logger.info("Best params: %s", clf_params)

    # Y6 fix: use the same seed the tuning report was produced under, not
    # a hardcoded RANDOM_STATE. Keeps SMOTE + classifier deterministic
    # w.r.t. the hyperparameter selection that produced clf_params.
    run_seed = _resolve_random_state(params_path)
    logger.info("Final-fit random_state=%d (resolved from tuning report)", run_seed)

    # Build pipeline: SMOTE + classifier with fixed params
    def _fresh_pipeline() -> ImbPipeline:
        # cls_kwargs already carries random_state for those classifiers
        # whose default is hardcoded; override it with the resolved seed
        # so the run is consistent end-to-end.
        cls_kwargs = {**cfg["cls_kwargs"], "random_state": run_seed}
        return ImbPipeline(
            [
                (
                    "smote",
                    SMOTE(
                        sampling_strategy="auto",
                        k_neighbors=5,
                        random_state=run_seed,
                    ),
                ),
                ("classifier", cfg["cls"](**cls_kwargs, **clf_params)),
            ]
        )

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
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=run_seed)
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
    # Strict JSON serialisation: a non-JSON value here is a producer
    # bug, not something to silently coerce with default=str (which
    # would render numpy arrays as their repr string and look plausible).
    try:
        payload = json.dumps(report, indent=2)
    except TypeError as exc:
        raise TypeError(
            f"{name}_final_report.json contains a non-JSON-serialisable "
            f"value (detail: {exc}). Fix the producer."
        ) from exc
    report_path.write_text(payload, encoding="utf-8")

    # Test predictions
    np.savez(
        output_dir / f"{name}_test_predictions.npz",
        y_true=y_test,
        y_pred=y_pred_test,
        y_proba=y_proba_test,
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
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    logger.info(
        "dae: attack_f1=%.4f  attack_f2=%.4f  AUC=%.4f  threshold=%.3g",
        metrics["attack_f1"],
        metrics["attack_f2"],
        metrics["auc_roc"],
        threshold,
    )
    logger.info(
        "\n%s",
        classification_report(
            y_test,
            y_pred,
            target_names=["Normal", "Attack"],
            digits=4,
        ),
    )
    return metrics


# ── Predict-only path (re-emit predictions on a frozen split) ───────────


def predict_split(split: str) -> dict:
    """Score a frozen labelled split with the already-trained models.

    Skips all training: loads the four signed pipeline artefacts via
    ``common.model_registry`` and emits one npz per model:
        results/models/{xgboost,random_forest,decision_tree}_{split}_predictions.npz
        results/models/dae_{split}_predictions.npz

    Returns the same metrics dict shape as ``train_track_a`` so callers
    can log a summary table for whichever split was scored.
    """
    from common.model_registry import (
        get_track_a_classifiers,
        get_track_a_thresholds,
    )
    from detection_engine import DetectionEngine

    sep = "-" * 60
    logger.info(sep)
    logger.info("PREDICT-ONLY: split=%s", split)
    logger.info(sep)

    X, y, _feat_names = load_split_data(split)
    classifiers = get_track_a_classifiers()
    thresholds = get_track_a_thresholds()

    output_dir = PROJECT_ROOT / "results/models"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_metrics: dict = {}

    for name, clf in classifiers.items():
        threshold = thresholds[name]
        y_proba = clf.predict_proba(X)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)

        npz_path = output_dir / f"{name}_{split}_predictions.npz"
        np.savez(npz_path, y_true=y, y_pred=y_pred, y_proba=y_proba)
        logger.info("Saved: %s", npz_path)

        all_metrics[name] = evaluate(name, y, y_pred, y_proba, threshold)

    # DAE — split-aware via the engine; models are cached by model_registry.
    dae_npz = DetectionEngine().write_predictions(split=split)

    dae_report_path = PROJECT_ROOT / "results/models/dae_final_report.json"
    if dae_report_path.exists():
        dae_report = json.loads(dae_report_path.read_text(encoding="utf-8"))
        dae_metrics = evaluate_dae(dae_npz, dae_report.get("threshold", 0.0))
        all_metrics["dae"] = dae_metrics

    return all_metrics


# ── Main ────────────────────────────────────────────────────────────────


def main() -> None:
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description=(
            "Train detection models on the training split, then score the "
            "frozen test split. Use --predict-only to skip training and only "
            "re-emit predictions on a chosen frozen split (test/demo)."
        )
    )
    parser.add_argument(
        "--predict-only",
        action="store_true",
        help=(
            "Skip training. Load the existing signed pipeline artefacts and "
            "write {model}_<split>_predictions.npz for each split selected "
            "by --split."
        ),
    )
    parser.add_argument(
        "--split",
        choices=("test", "demo", "both"),
        default="test",
        help=(
            "Frozen split(s) to score in --predict-only mode "
            "(test=paper-clean, demo=operator-clean). Ignored when "
            "training. Default: test."
        ),
    )
    args = parser.parse_args()

    sep = "=" * 72

    # ── Predict-only path: no training, just re-score frozen splits ──
    if args.predict_only:
        logger.info(sep)
        logger.info("PREDICT-ONLY MODE — split(s)=%s", args.split)
        logger.info(sep)

        t0 = time.perf_counter()
        splits_to_run = ["test", "demo"] if args.split == "both" else [args.split]
        for split in splits_to_run:
            predict_split(split)

        total = round(time.perf_counter() - t0, 1)
        logger.info(sep)
        logger.info("PREDICT-ONLY COMPLETE — %.1fs (splits=%s)", total, splits_to_run)
        logger.info(sep)
        return

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
    # Training only writes the DAE artifact; engine.write_predictions
    # produces dae_test_predictions.npz with the cascaded-fusion scores
    # the rest of the pipeline expects.
    from module2_detection.dae_training import train_dae
    from detection_engine import DetectionEngine
    from common.model_registry import invalidate_cache

    dae_summary = train_dae()
    invalidate_cache()  # force re-load of the freshly written DAE artifact
    dae_npz = DetectionEngine().write_predictions()

    # Patch the report and merge metrics so the summary table reflects
    # real DAE performance instead of the all-zeros placeholder.
    dae_report_path = PROJECT_ROOT / "results/models/dae_final_report.json"
    dae_report = json.loads(dae_report_path.read_text(encoding="utf-8"))
    dae_metrics = evaluate_dae(dae_npz, dae_report.get("threshold", 0.0))
    dae_report["test_metrics"] = dae_metrics
    try:
        dae_payload = json.dumps(dae_report, indent=2)
    except TypeError as exc:
        raise TypeError(
            f"dae_final_report.json contains a non-JSON-serialisable "
            f"value (detail: {exc}). Fix the producer."
        ) from exc
    dae_report_path.write_text(dae_payload, encoding="utf-8")
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
