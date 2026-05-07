
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

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor
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
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import dumps_signed
from module2_detection.models.DAE import DAEDetector
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

    # Non-feature columns introduced by GAP-PB-1 (row_id, device_class,
    # attack_category) must be dropped from the feature matrix but preserved
    # for downstream join. We drop them here; the predictions writer reads
    # row_id back from the test parquet directly.
    drop_cols = [
        c for c in [label_col, "Attack Category", "row_id",
                    "device_class", "attack_category"]
        if c in train_df.columns
    ]

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
            "random_seed": int(RANDOM_STATE),
        },
        "elapsed_seconds": elapsed,
    }
    report_path = output_dir / f"{name}_final_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    # Test predictions — include row_id when available so downstream
    # joins (per-device-class metrics, stratified-split materialiser) can
    # match predictions back to test_phase1.parquet rows. row_id falls back
    # to the positional index if the parquet schema doesn't yet carry it
    # (graceful degradation; closes GAP-PB-1 once Module 1 emits row_id).
    pred_kwargs = dict(y_true=y_test, y_pred=y_pred_test, y_proba=y_proba_test)
    test_parquet_path = PROJECT_ROOT / "data/processed/test_phase1.parquet"
    if test_parquet_path.exists():
        try:
            test_df = pd.read_parquet(test_parquet_path, columns=["row_id"])
            pred_kwargs["row_id"] = test_df["row_id"].values
        except (KeyError, ValueError):
            pred_kwargs["row_id"] = np.arange(len(y_test), dtype=np.int64)
    np.savez(output_dir / f"{name}_test_predictions.npz", **pred_kwargs)

    # OOF probabilities — kept as a fallback for the cascaded DAE input
    # when no held-out validation parquet is available (legacy path).
    oof_path = output_dir / f"{name}_oof_proba.npy"
    np.save(oof_path, oof_proba)
    logger.info("Saved OOF probas: %s", oof_path)

    # ── GAP-L1-1: validation-set probas for cascaded DAE input ──
    # When a held-out validation parquet exists (emitted by Module 1
    # when val_ratio>0), generate val-set probas and persist them. The
    # DAE cascade in train_track_b_dae prefers these over OOF probas
    # because they avoid leaking CV-fold structure into the joint
    # feature-prediction space the DAE learns.
    val_parquet_path = PROJECT_ROOT / "data/processed/val_phase1.parquet"
    if val_parquet_path.exists():
        val_df = pd.read_parquet(val_parquet_path)
        drop_cols = [
            c for c in ["Label", "Attack Category", "row_id",
                        "device_class", "attack_category"]
            if c in val_df.columns
        ]
        X_val = val_df.drop(columns=drop_cols).values.astype(np.float32)
        val_proba = pipeline.predict_proba(X_val)[:, 1]
        val_path = output_dir / f"{name}_val_proba.npy"
        np.save(val_path, val_proba)
        logger.info("Saved validation-set probas (n=%d): %s",
                    len(val_proba), val_path)

    logger.info("Saved: %s (%.1fs)", output_dir, elapsed)
    return metrics


# ── Track B: DAE ────────────────────────────────────────────────────────

_TRACK_A_MODELS = ("xgboost", "random_forest", "decision_tree")


def _load_track_a_probas(
    output_dir: Path,
    benign_mask: np.ndarray,
    *,
    suffix: str,
) -> np.ndarray:
    """Load Track A probabilities for all 3 models and stack benign rows.

    Args:
        output_dir: Directory containing ``{model}_{suffix}.npy`` files.
        benign_mask: Boolean mask selecting benign rows.
        suffix: Either ``"oof_proba"`` (out-of-fold, used as fallback) or
            ``"val_proba"`` (held-out validation set — closes GAP-L1-1).

    Returns:
        Array of shape ``(n_benign, 3)`` — one column per Track A model.

    Opt-5: three .npy files are loaded concurrently via ThreadPoolExecutor
    (I/O bound, GIL released for numpy file reads).
    """
    def _load_one(name: str) -> np.ndarray:
        return np.load(output_dir / f"{name}_{suffix}.npy")[benign_mask]

    with ThreadPoolExecutor(max_workers=3) as pool:
        cols = list(pool.map(_load_one, _TRACK_A_MODELS))
    return np.column_stack(cols)


def _val_probas_available(output_dir: Path) -> bool:
    """True iff all 3 Track A val-proba files exist."""
    return all(
        (output_dir / f"{n}_val_proba.npy").exists()
        for n in _TRACK_A_MODELS
    )


def _track_a_test_probas(X_test: np.ndarray, output_dir: Path) -> np.ndarray:
    """Run Track A models on test set, return stacked probabilities.

    Returns:
        Array of shape (n_test, 3).
    """
    from common import loads_signed

    cols = []
    for name in ("xgboost", "random_forest", "decision_tree"):
        clf = loads_signed(output_dir / f"{name}_final_pipeline.pkl")
        cols.append(clf.predict_proba(X_test)[:, 1])
    return np.column_stack(cols)


def train_track_b_dae(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feat_names: list,
) -> dict:
    """Train cascaded DAE: input = [raw features || Track A probas].

    Track A must be trained first. The DAE learns to reconstruct benign
    samples in the joint (features + Track-A-prediction) space. Spoofing
    attacks that look normal in raw features but trigger Track A become
    visible as high reconstruction error.

    GAP-L1-1: when held-out validation artifacts exist (``val_phase1.parquet``
    + ``*_val_proba.npy`` for all three Track A models), train on val benign
    samples augmented with val-set probas. Otherwise fall back to the legacy
    OOF path (train benign + OOF probas).
    """
    t0 = time.perf_counter()
    sep = "-" * 60

    logger.info(sep)
    logger.info("FINAL TRAINING: DAE (TRACK B — CASCADED)")
    logger.info(sep)

    output_dir = PROJECT_ROOT / "results/models"

    # Load best params
    params_path = output_dir / "dae_best_params.json"
    with open(params_path) as f:
        best_hp = json.load(f)
    logger.info("Best params: %s", best_hp)

    val_parquet_path = PROJECT_ROOT / "data/processed/val_phase1.parquet"
    use_val_path = val_parquet_path.exists() and _val_probas_available(output_dir)

    if use_val_path:
        # ── GAP-L1-1: held-out validation-set probas ──
        # The DAE trains on benign val samples + val-set Track A probas
        # (predict_proba on rows the supervised models never saw during
        # CV). This eliminates the train-inference skew that arose from
        # using OOF probas, which encode CV-fold structure into the
        # joint (features, Track-A-prediction) space.
        val_df = pd.read_parquet(val_parquet_path)
        drop_cols = [
            c for c in ["Label", "Attack Category", "row_id",
                        "device_class", "attack_category"]
            if c in val_df.columns
        ]
        y_val = val_df["Label"].values.astype(int)
        X_val = val_df.drop(columns=drop_cols).values.astype(np.float32)
        benign_mask = y_val == 0
        X_benign = X_val[benign_mask]
        probas = _load_track_a_probas(output_dir, benign_mask, suffix="val_proba")
        proba_source = "val"
        logger.info(
            "GAP-L1-1: using held-out val set for DAE training "
            "(n_val=%d, n_benign=%d)",
            len(y_val), int(benign_mask.sum()),
        )
    else:
        # Legacy OOF path (no val artifacts available)
        benign_mask = y_train == 0
        X_benign = X_train[benign_mask]
        probas = _load_track_a_probas(output_dir, benign_mask, suffix="oof_proba")
        proba_source = "oof"
        logger.info(
            "Val artifacts unavailable; falling back to OOF probas "
            "(legacy path; benign train n=%d)",
            int(benign_mask.sum()),
        )

    logger.info(
        "Track A %s probas (benign): shape=%s, means=%s",
        proba_source, probas.shape,
        np.round(probas.mean(axis=0), 4),
    )

    # Augmented input: [25 raw features || 3 Track A probas] = 28 features
    X_benign_aug = np.column_stack([X_benign, probas])
    aug_feat_names = feat_names + ["track_a_xgb", "track_a_rf", "track_a_dt"]
    logger.info(
        "Cascaded DAE input: %d features (%d raw + %d Track A)",
        X_benign_aug.shape[1], len(feat_names), probas.shape[1],
    )

    # Adjust architecture for 28 features
    # Bottleneck must be < n_features; scale encoder/decoder proportionally
    n_feat = X_benign_aug.shape[1]
    enc_dim = max(best_hp.get("encoding_dims", [20, 12, 20])[0], n_feat - 4)
    bot_dim = min(best_hp.get("encoding_dims", [20, 12, 20])[1], n_feat - 2)
    dec_dim = enc_dim
    adjusted_dims = [enc_dim, bot_dim, dec_dim]
    logger.info("Adjusted architecture: %s (for %d features)", adjusted_dims, n_feat)

    det = DAEDetector(
        encoding_dims=adjusted_dims,
        noise_rate=best_hp.get("noise_rate", 0.2),
        learning_rate=best_hp.get("learning_rate", 0.0001),
        threshold_percentile=best_hp.get("threshold_percentile", 95.0),
        clip_percentile=best_hp.get("clip_percentile", 1.0),
        epochs=100,
        batch_size=256,
        random_state=RANDOM_STATE,
    )
    det.fit(X_benign_aug, validation_split=0.0)

    # Augmented test set
    test_probas = _track_a_test_probas(X_test, output_dir)
    X_test_aug = np.column_stack([X_test, test_probas])

    # Evaluate
    test_metrics = det.evaluate(X_test_aug, y_test)

    elapsed = round(time.perf_counter() - t0, 1)

    # Save artifacts
    output_dir.mkdir(parents=True, exist_ok=True)

    det.save_artefacts(
        json_path=output_dir / "dae_detector.json",
        weights_path=output_dir / "dae_model.weights.h5",
    )

    # Report
    report = det.get_report()
    report["stage"] = "final_training"
    report["architecture"] = "cascaded"
    report["best_hyperparameters"] = best_hp
    report["adjusted_encoding_dims"] = adjusted_dims
    report["data"] = {
        "n_raw_features": len(feat_names),
        "n_track_a_features": 3,
        "n_total_features": n_feat,
        "feature_names": aug_feat_names,
        "benign_train_samples": int(benign_mask.sum()),
        "test_samples": int(len(y_test)),
        "random_seed": int(RANDOM_STATE),
        "track_a_proba_source": proba_source,  # "val" closes GAP-L1-1; "oof" = legacy
    }
    report["elapsed_seconds"] = elapsed

    report_path = output_dir / "dae_final_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    # Test predictions (on augmented input)
    y_pred = det.predict(X_test_aug)
    errors = det.reconstruction_error(X_test_aug)
    np.savez(
        output_dir / "dae_test_predictions.npz",
        y_true=y_test, y_pred=y_pred, reconstruction_error=errors,
    )

    logger.info("Saved: %s (%.1fs)", output_dir, elapsed)
    return test_metrics


# ── Demo-pool prediction (post-training) ────────────────────────────────


def predict_demo() -> None:
    """Emit ``{model}_demo_predictions.npz`` for all 3 Track A models + DAE.

    Loads the frozen pipelines from ``results/models/`` and runs them on
    ``data/processed/demo_phase1.parquet`` (the 10% frozen demo pool).
    Outputs match the test-prediction schema so M3 can score demo rows
    via the same path. Per ARCHITECTURE.md: demo never touches the
    paper-metrics path; M6's ``evaluation_alerts.json`` is sourced from
    these demo predictions, NOT from test predictions.
    """
    from common import loads_signed
    from module2_detection.models.DAE import DAEDetector

    output_dir = PROJECT_ROOT / "results/models"
    demo_path = PROJECT_ROOT / "data/processed/demo_phase1.parquet"
    if not demo_path.exists():
        logger.warning(
            "demo_phase1.parquet not found — skipping demo-pool predictions. "
            "Re-run module1_preprocessing.phase1 to materialise the demo split."
        )
        return

    demo_df = pd.read_parquet(demo_path)
    drop_cols = [
        c for c in ["Label", "Attack Category", "row_id",
                    "device_class", "attack_category"]
        if c in demo_df.columns
    ]
    y_demo = demo_df["Label"].values.astype(int)
    X_demo = demo_df.drop(columns=drop_cols).values.astype(np.float32)
    row_ids = (
        demo_df["row_id"].values
        if "row_id" in demo_df.columns
        else np.arange(len(demo_df), dtype=np.int64)
    )

    logger.info(
        "Demo-pool predictions: n=%d, attack_rate=%.4f",
        len(y_demo), float(y_demo.mean()),
    )

    # Track A: 3 supervised pipelines
    for name in ("xgboost", "random_forest", "decision_tree"):
        clf = loads_signed(output_dir / f"{name}_final_pipeline.pkl")
        with open(output_dir / f"{name}_final_report.json") as f:
            threshold = json.load(f)["optimal_threshold"]
        y_proba = clf.predict_proba(X_demo)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)
        np.savez(
            output_dir / f"{name}_demo_predictions.npz",
            y_true=y_demo, y_pred=y_pred, y_proba=y_proba, row_id=row_ids,
        )
        logger.info("  %s_demo_predictions.npz written (n=%d)", name, len(y_demo))

    # Track B: DAE on cascaded input [raw 25 || P_xgb, P_rf, P_dt] = 28 dims
    track_a_demo = np.column_stack([
        np.load(output_dir / f"{n}_demo_predictions.npz")["y_proba"]
        for n in ("xgboost", "random_forest", "decision_tree")
    ])
    X_demo_aug = np.column_stack([X_demo, track_a_demo])

    det = DAEDetector.from_artefacts(
        output_dir / "dae_detector.json",
        output_dir / "dae_model.weights.h5",
    )
    y_pred_dae = det.predict(X_demo_aug)
    errors = det.reconstruction_error(X_demo_aug)
    np.savez(
        output_dir / "dae_demo_predictions.npz",
        y_true=y_demo, y_pred=y_pred_dae, reconstruction_error=errors,
        row_id=row_ids,
    )
    logger.info("  dae_demo_predictions.npz written (n=%d)", len(y_demo))


# ── Main ────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Train final Track A + Track B models with fixed best hyperparameters."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for SMOTE, KFold, classifiers, and DAE init "
             "(default: 42; persisted into every *_final_report.json)",
    )
    args = parser.parse_args()

    global RANDOM_STATE
    RANDOM_STATE = args.seed

    sep = "=" * 72
    logger.info(sep)
    logger.info("FINAL MODEL TRAINING — FIXED BEST HYPERPARAMETERS  (seed=%d)",
                RANDOM_STATE)
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

    # ── Demo-pool predictions (Strategy 1 — frozen demo split) ──
    # Loads the just-trained models + demo_phase1.parquet and emits a
    # parallel set of {model}_demo_predictions.npz files. M3 then scores
    # the demo rows separately to produce demo_scores.npz; M6 sources
    # evaluation_alerts.json from those (NEVER from test).
    predict_demo()

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
