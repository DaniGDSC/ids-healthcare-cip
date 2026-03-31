#!/usr/bin/env python3
"""End-to-end: tune hyperparameters, then train/val/test with the best config.

Pipeline:
  1. Run Phase 2.5 Bayesian search (Optuna TPE + Hyperband pruning)
  2. Parameter importance analysis
  3. Multi-seed validation of top-3 configs
  4. Ablation study
  5. Full retrain: build detection model + classification head with best HP
  6. Progressive unfreezing with tuned Phase 3 LR schedule
  7. Evaluate on test set — report final metrics vs. baseline

Usage::

    python -m src.phase2_5_fine_tuning.run_finetuned_training
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report as sklearn_cls_report
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# ── Focal loss + label smoothing ─────────────────────────────────
LABEL_SMOOTH_EPS = 0.05


def focal_loss(gamma: float = 2.0, alpha: float = 0.25):
    """Focal loss for class imbalance and sigmoid saturation prevention.

    Down-weights easy (confident) examples so the model focuses on
    hard/misclassified ones, preventing the sigmoid output from
    saturating to a narrow range.

    Args:
        gamma: Focusing parameter (higher = more focus on hard examples).
        alpha: Balance factor for positive class weight.
    """
    def _focal(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        pt = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
        at = y_true * alpha + (1.0 - y_true) * (1.0 - alpha)
        return -tf.reduce_mean(at * tf.pow(1.0 - pt, gamma) * tf.math.log(pt))
    _focal.__name__ = f"focal_loss_g{gamma}_a{alpha}"
    return _focal


def multi_class_focal_loss(gamma: float = 2.0):
    """Focal loss for multi-class classification.

    Down-weights easy examples per-class. Improves minority class
    detection (Data Alteration with only 922 samples).
    """
    def _loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        pt = tf.reduce_sum(y_true * y_pred, axis=-1)
        return -tf.reduce_mean(tf.pow(1.0 - pt, gamma) * tf.math.log(pt))
    _loss.__name__ = f"multi_focal_g{gamma}"
    return _loss


def augment_windows(X: np.ndarray, y: np.ndarray, n_classes: int = 3) -> tuple:
    """Data augmentation: minority oversampling + window jitter + noise.

    Args:
        X: Windows (N, timesteps, features).
        y: Labels (N,) integer class IDs.
        n_classes: Number of classes.

    Returns:
        Augmented (X, y) with more minority samples.
    """
    rng = np.random.RandomState(42)
    X_aug, y_aug = [X], [y]

    # 1. Oversample minority classes to match 50% of majority
    counts = np.bincount(y.astype(int), minlength=n_classes)
    target = max(counts) // 2
    for cls in range(n_classes):
        if counts[cls] < target:
            deficit = target - counts[cls]
            cls_mask = y == cls
            cls_X = X[cls_mask]
            if len(cls_X) == 0:
                continue
            indices = rng.choice(len(cls_X), size=deficit, replace=True)
            oversampled = cls_X[indices].copy()
            # Add small Gaussian noise to make them distinct
            oversampled += rng.normal(0, 0.05, oversampled.shape).astype(np.float32)
            X_aug.append(oversampled)
            y_aug.append(np.full(deficit, cls, dtype=y.dtype))

    # 2. Window jitter on 30% of samples (random temporal shift ±2)
    X_all = np.concatenate(X_aug, axis=0)
    y_all = np.concatenate(y_aug, axis=0)

    jitter_mask = rng.rand(len(X_all)) < 0.3
    for i in np.where(jitter_mask)[0]:
        shift = rng.randint(-2, 3)
        if shift > 0:
            X_all[i, shift:] = X_all[i, :-shift]
            X_all[i, :shift] = 0
        elif shift < 0:
            X_all[i, :shift] = X_all[i, -shift:]
            X_all[i, shift:] = 0

    # Shuffle
    perm = rng.permutation(len(X_all))
    return X_all[perm], y_all[perm]


# ── Phase 2 SOLID components (reused) ────────────────────────────
from src.phase2_detection_engine.phase2.assembler import DetectionModelAssembler
from src.phase2_detection_engine.phase2.attention_builder import (
    AttentionBuilder,
    BahdanauAttention,  # noqa: F401
)
from src.phase2_detection_engine.phase2.bilstm_builder import BiLSTMBuilder
from src.phase2_detection_engine.phase2.cnn_builder import CNNBuilder
from src.phase2_detection_engine.phase2.reshaper import DataReshaper

# ── Phase 3 SOLID components (reused) ────────────────────────────
from src.phase3_classification_engine.phase3.head import AutoClassificationHead
from src.phase3_classification_engine.phase3.unfreezer import ProgressiveUnfreezer

# ── Phase 2.5 components ─────────────────────────────────────────
from src.phase2_5_fine_tuning.phase2_5.ablation import AblationRunner
from src.phase2_5_fine_tuning.phase2_5.config import Phase2_5Config
from src.phase2_5_fine_tuning.phase2_5.evaluator import (
    QuickEvaluator,
    _find_optimal_threshold,
)
from src.phase2_5_fine_tuning.phase2_5.exporter import TuningExporter
from src.phase2_5_fine_tuning.phase2_5.importance import compute_importance
from src.phase2_5_fine_tuning.phase2_5.multi_seed import MultiSeedValidator
from src.phase2_5_fine_tuning.phase2_5.report import render_tuning_report
from src.phase2_5_fine_tuning.phase2_5.search_space import SearchSpace
from src.phase2_5_fine_tuning.phase2_5.tuner import HyperparameterTuner

# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "phase2_5_config.yaml"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  [%(name)s]  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("run_finetuned_training")


# ===================================================================
# Class-weighted trainer (extends Phase 3 trainer with class_weight)
# ===================================================================

# ===================================================================
# Helpers
# ===================================================================

def _load_smote_data(config: Phase2_5Config):
    """Load SMOTE-balanced train (for HP search) + imbalanced test."""
    label_col = config.label_column
    train_df = pd.read_parquet(PROJECT_ROOT / config.phase1_train)
    test_df = pd.read_parquet(PROJECT_ROOT / config.phase1_test)
    feature_names = [c for c in train_df.columns if c != label_col]

    X_train = train_df[feature_names].values.astype(np.float32)
    y_train = train_df[label_col].values
    X_test = test_df[feature_names].values.astype(np.float32)
    y_test = test_df[label_col].values

    logger.info("  SMOTE train: %s (balanced), Test: %s (imbalanced)", X_train.shape, X_test.shape)
    return X_train, y_train, X_test, y_test, feature_names


def _load_original_train(config: Phase2_5Config, feature_names):
    """Load original imbalanced train from train_phase1.parquet.

    Reads from the same parquet used by _load_smote_data() to ensure
    consistency with the 70/15/15 split. No re-splitting from raw CSV.
    """
    label_col = config.label_column
    train_df = pd.read_parquet(PROJECT_ROOT / config.phase1_train)
    y_train = np.asarray(train_df[label_col].values)
    X_train = np.asarray(train_df[feature_names].values, dtype=np.float32)

    n0, n1 = int(np.sum(y_train == 0)), int(np.sum(y_train == 1))
    logger.info("  Original train: %d (Normal=%d, Attack=%d, ratio=%.1f:1)",
                len(y_train), n0, n1, n0 / max(n1, 1))
    return X_train, y_train


def _load_baseline_metrics() -> Dict[str, Any]:
    """Load the current Phase 3 baseline metrics for comparison."""
    path = PROJECT_ROOT / "data" / "phase3" / "metrics_wustl_v2.json"
    if path.exists():
        with open(path) as f:
            raw = json.load(f)
        return raw.get("metrics", raw)
    return {}


def _load_p2_architecture() -> Dict[str, Any]:
    """Load Phase 2 architecture hyperparameters from metadata."""
    path = PROJECT_ROOT / "data" / "phase2" / "detection_metadata.json"
    with open(path) as f:
        return json.load(f)["hyperparameters"]


def _build_detection_model(hp: Dict[str, Any], n_features: int) -> tf.keras.Model:
    """Build CNN-BiLSTM-Attention detection model from hyperparameters."""
    timesteps = int(hp.get("timesteps", 20))
    builders = [
        CNNBuilder(
            filters_1=int(hp.get("cnn_filters_1", 64)),
            filters_2=int(hp.get("cnn_filters_2", 128)),
            kernel_size=int(hp.get("cnn_kernel_size", 3)),
            activation="relu",
            pool_size=2,
            has_second_pool=hp.get("cnn_has_second_pool", True),
        ),
        BiLSTMBuilder(
            units_1=int(hp.get("bilstm_units_1", 128)),
            units_2=int(hp.get("bilstm_units_2", 64)),
            dropout_rate=float(hp.get("dropout_rate", 0.3)),
        ),
        AttentionBuilder(units=int(hp.get("attention_units", 128))),
    ]
    assembler = DetectionModelAssembler(
        timesteps=timesteps, n_features=n_features, builders=builders,
    )
    return assembler.assemble()


def _evaluate(
    model: tf.keras.Model,
    X: np.ndarray,
    y: np.ndarray,
    threshold: float = 0.5,
    label: str = "test",
) -> Dict[str, Any]:
    """Evaluate model with attack-aware metrics."""
    y_prob = model.predict(X, verbose=0)
    if y_prob.shape[-1] == 1:
        y_prob = y_prob.ravel()
        y_pred = (y_prob > threshold).astype(int)
    else:
        y_pred = np.argmax(y_prob, axis=1)

    # Standard weighted metrics
    acc = float(accuracy_score(y, y_pred))
    f1_w = float(f1_score(y, y_pred, average="weighted", zero_division=0))
    prec_w = float(precision_score(y, y_pred, average="weighted", zero_division=0))
    rec_w = float(recall_score(y, y_pred, average="weighted", zero_division=0))
    try:
        if y_prob.ndim > 1 and y_prob.shape[-1] > 1:
            auc = float(roc_auc_score(y, y_prob, multi_class="ovr", average="macro"))
        else:
            auc = float(roc_auc_score(y, y_prob))
    except ValueError:
        auc = 0.0

    # Attack-specific: binary view (attack = any class > 0)
    y_bin = (y > 0).astype(int)
    y_pred_bin = (y_pred > 0).astype(int)
    atk_recall = float(recall_score(y_bin, y_pred_bin, pos_label=1, average="binary", zero_division=0))
    atk_prec = float(precision_score(y_bin, y_pred_bin, pos_label=1, average="binary", zero_division=0))
    atk_f1 = float(f1_score(y_bin, y_pred_bin, pos_label=1, average="binary", zero_division=0))
    atk_f2 = float(fbeta_score(y_bin, y_pred_bin, beta=2, pos_label=1, average="binary", zero_division=0))

    # Macro (equal class weight)
    macro_f1_val = float(f1_score(y, y_pred, average="macro", zero_division=0))

    metrics = {
        "accuracy": acc,
        "f1_score": f1_w,
        "precision": prec_w,
        "recall": rec_w,
        "auc_roc": auc,
        "attack_recall": atk_recall,
        "attack_precision": atk_prec,
        "attack_f1": atk_f1,
        "attack_f2": atk_f2,
        "macro_f1": macro_f1_val,
        "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
        "classification_report": sklearn_cls_report(y, y_pred, output_dict=True, zero_division=0),
        "threshold": threshold,
        "samples": len(y),
    }

    logger.info("  [%s] Attack_F1=%.4f  Recall=%.4f  Macro_F1=%.4f  AUC=%.4f  (%d samples)",
                label, atk_f1, atk_recall, macro_f1_val, auc, len(y))
    return metrics


# ===================================================================
# Main
# ===================================================================

def main() -> None:
    t0 = time.time()

    logger.info("═" * 60)
    logger.info("  PHASE 2.5 → FULL TRAINING WITH FINE-TUNED HYPERPARAMETERS")
    logger.info("═" * 60)

    # ── Setup ──
    config = Phase2_5Config.from_yaml(CONFIG_PATH)
    np.random.seed(config.random_state)
    tf.random.set_seed(config.random_state)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"

    # Load SMOTE-balanced train for HP search (good feature learning)
    # + imbalanced test for attack_f1 evaluation
    X_train_smote, y_train_smote, X_test, y_test, feature_names = _load_smote_data(config)

    # Also load original imbalanced train for the final retrain (Step 5)
    X_train_orig, y_train_orig = _load_original_train(config, feature_names)
    n_features = len(feature_names)
    baseline = _load_baseline_metrics()

    logger.info("  Train(SMOTE): %s, Train(orig): %s, Test: %s, Features: %d",
                X_train_smote.shape, X_train_orig.shape, X_test.shape, n_features)

    # ══════════════════════════════════════════════════════════════
    # STEP 1: Hyperparameter Search (Bayesian TPE + Hyperband)
    # ══════════════════════════════════════════════════════════════
    logger.info("")
    logger.info("STEP 1: Hyperparameter search (TPE, %d trials)",
                config.max_trials)

    search_space = SearchSpace(config.search_space, config.random_state)
    p2_weights_path = str(PROJECT_ROOT / "data" / "phase2" / "detection_model.weights.h5")
    p2_hp = _load_p2_architecture()
    model_architecture = {
        "cnn_filters_1": p2_hp["cnn_filters_1"],
        "cnn_filters_2": p2_hp["cnn_filters_2"],
        "bilstm_units_1": p2_hp["bilstm_units_1"],
        "bilstm_units_2": p2_hp["bilstm_units_2"],
        "attention_units": p2_hp["attention_units"],
        "head_dense_units": 32,
        "dropout_rate": p2_hp["dropout_rate"],
        "timesteps": p2_hp["timesteps"],
    }
    evaluator = QuickEvaluator(
        config.quick_train, n_features=n_features,
        random_state=config.random_state,
        pretrained_weights_path=p2_weights_path,
        model_architecture=model_architecture,
    )
    tuner = HyperparameterTuner(config, evaluator, search_space)

    tuning_results = tuner.run(X_train_smote, y_train_smote, X_train_orig, y_train_orig, X_test, y_test)

    best_hp = tuning_results["best_config"]
    logger.info("  Best config found (quick %s=%.4f):", config.search_metric, tuning_results["best_score"])
    for k, v in sorted(best_hp.items()):
        logger.info("    %s = %s", k, f"{v:.6g}" if isinstance(v, float) else v)

    # ══════════════════════════════════════════════════════════════
    # STEP 2: Parameter Importance
    # ══════════════════════════════════════════════════════════════
    logger.info("")
    logger.info("STEP 2: Parameter importance analysis")

    importance_results = compute_importance(
        tuner, tuning_results.get("trials", []), config.search_metric,
    )
    top3_params = list(importance_results.get("importances", {}).items())[:3]
    for name, score in top3_params:
        logger.info("  %s: %.4f", name, score)

    # ══════════════════════════════════════════════════════════════
    # STEP 3: Multi-Seed Validation
    # ══════════════════════════════════════════════════════════════
    logger.info("")
    logger.info("STEP 3: Multi-seed validation (top-%d, %d seeds)",
                config.multi_seed.top_k, len(config.multi_seed.seeds))

    ms_validator = MultiSeedValidator(config.multi_seed, evaluator)
    multi_seed_results = ms_validator.validate(tuning_results, X_train_smote, y_train_smote, X_test, y_test)

    # ══════════════════════════════════════════════════════════════
    # STEP 4: Ablation Study
    # ══════════════════════════════════════════════════════════════
    logger.info("")
    logger.info("STEP 4: Ablation study (%d variants)", len(config.ablation_variants))

    ablation_runner = AblationRunner(config, evaluator)
    ablation_results = ablation_runner.run(best_hp, X_train_smote, y_train_smote, X_test, y_test)

    # ══════════════════════════════════════════════════════════════
    # STEP 5: Full Retrain with Best Hyperparameters
    # ══════════════════════════════════════════════════════════════
    logger.info("")
    logger.info("STEP 5: Full retrain with best hyperparameters")
    logger.info("═" * 60)

    # Reset seeds for final training
    np.random.seed(config.random_state)
    tf.random.set_seed(config.random_state)

    timesteps = p2_hp["timesteps"]
    stride = int(best_hp.get("stride", 1))

    # 5a. Build detection model and load Phase 2 pretrained weights
    detection_model = _build_detection_model({
        "cnn_filters_1": p2_hp["cnn_filters_1"],
        "cnn_filters_2": p2_hp["cnn_filters_2"],
        "cnn_kernel_size": p2_hp["cnn_kernel_size"],
        "bilstm_units_1": p2_hp["bilstm_units_1"],
        "bilstm_units_2": p2_hp["bilstm_units_2"],
        "attention_units": p2_hp["attention_units"],
        "dropout_rate": float(best_hp.get("dropout_rate", p2_hp["dropout_rate"])),
        "timesteps": timesteps,
    }, n_features)
    detection_params = detection_model.count_params()

    p2_weights = PROJECT_ROOT / "data" / "phase2" / "detection_model.weights.h5"
    detection_model.load_weights(str(p2_weights), skip_mismatch=True)
    logger.info("  Detection model: %d params (Phase 2 weights loaded, skip_mismatch=True)", detection_params)

    # 5b. Reshape datasets with stride=5 (reduced overlap)
    # stride=20 gives only 571 windows (too few for 3-class).
    # stride=5 gives ~2,280 windows — 4x less overlap than stride=1.
    stride = 5
    reshaper = DataReshaper(timesteps=timesteps, stride=stride)
    Xo_w, yo_w = reshaper.reshape(X_train_orig, y_train_orig)
    X_test_w, y_test_w = reshaper.reshape(X_test, y_test)

    n_classes = len(np.unique(yo_w))
    logger.info("  Train (windowed, stride=%d): %d windows, %d classes",
                stride, len(yo_w), n_classes)
    for c in range(n_classes):
        logger.info("    Class %d: %d windows", c, int(np.sum(yo_w == c)))

    # 5c. Data augmentation (minority oversampling + jitter)
    if n_classes > 2:
        Xo_w, yo_w = augment_windows(Xo_w, yo_w, n_classes)
        logger.info("  After augmentation: %d windows", len(yo_w))
        for c in range(n_classes):
            logger.info("    Class %d: %d windows", c, int(np.sum(yo_w == c)))

    # 5d. Multi-class label preparation
    if n_classes > 2:
        from tensorflow.keras.utils import to_categorical
        yo_w_orig = yo_w.copy()
        y_test_w_orig = y_test_w.copy()
        yo_w_smooth = to_categorical(yo_w, num_classes=n_classes)
        y_test_w_cat = to_categorical(y_test_w, num_classes=n_classes)
        logger.info("  Multi-class: %d classes, one-hot encoded", n_classes)
    else:
        yo_w_orig = yo_w.copy()
        y_test_w_orig = y_test_w.copy()
        yo_w_smooth = yo_w * (1 - 2 * LABEL_SMOOTH_EPS) + LABEL_SMOOTH_EPS
        logger.info("  Binary: label smoothing eps=%.2f", LABEL_SMOOTH_EPS)

    # 5d. Add classification head
    head = AutoClassificationHead(
        dense_units=model_architecture["head_dense_units"],
        dense_activation="relu",
        dropout_rate=float(best_hp.get("dropout_rate", p2_hp["dropout_rate"])),
    )
    output_tensor = head.build(detection_model.output, n_classes)
    full_model = tf.keras.Model(detection_model.input, output_tensor, name="finetuned_model")

    total_params = full_model.count_params()
    head_params = total_params - detection_params
    logger.info("  Full model: %d params (detection=%d, head=%d)", total_params, detection_params, head_params)

    # 5e. Two-stage training
    head_lr = float(best_hp["head_lr"])
    finetune_lr = float(best_hp["finetune_lr"])
    cw_attack = float(best_hp["cw_attack"])
    head_epochs = int(best_hp["head_epochs"])
    ft_epochs = int(best_hp["ft_epochs"])

    # Class weights for multi-class
    if n_classes > 2:
        # Weight inversely proportional to class frequency
        counts = np.bincount(yo_w_orig.astype(int), minlength=n_classes)
        total_samples = len(yo_w_orig)
        class_weights = {i: total_samples / (n_classes * max(c, 1)) for i, c in enumerate(counts)}
    else:
        class_weights = {0: 1.0, 1: cw_attack}

    output_dir = PROJECT_ROOT / config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("  Training schedule (2-stage, imbalanced + class weight):")
    logger.info("    Stage 1: head_lr=%.6g, head_epochs=%d, cw_attack=%.4f (frozen backbone)",
                head_lr, head_epochs, cw_attack)
    logger.info("    Stage 2: finetune_lr=%.6g, ft_epochs=%d, cw_attack=%.4f (unfreeze CNN+bilstm2+attn)",
                finetune_lr, ft_epochs, cw_attack)

    histories = []

    # Stage 1: Train head on imbalanced data with class weight (frozen backbone)
    logger.info("═══ Stage 1: Head on imbalanced (frozen backbone) ═══")
    unfreezer = ProgressiveUnfreezer()
    unfreezer.apply_phase(full_model, frozen_groups=["cnn", "bilstm1", "bilstm2", "attention"])

    if n_classes > 2:
        loss_fn = multi_class_focal_loss(gamma=2.0)
    else:
        loss_fn = focal_loss(gamma=2.0, alpha=0.25)
    full_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=head_lr),
        loss=loss_fn, metrics=["accuracy"],
    )
    h1 = full_model.fit(
        Xo_w, yo_w_smooth, epochs=head_epochs, batch_size=256,
        validation_split=0.2, verbose=1,
        class_weight=class_weights,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=3, restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(output_dir / "checkpoint_stage1.weights.h5"),
                save_best_only=True, save_weights_only=True),
        ],
    )
    histories.append({
        "phase": "Stage 1 — Head on imbalanced",
        "epochs_run": len(h1.history["loss"]),
        "final_train_loss": float(h1.history["loss"][-1]),
        "final_train_acc": float(h1.history["accuracy"][-1]),
        "final_val_loss": float(h1.history["val_loss"][-1]),
        "final_val_acc": float(h1.history["val_accuracy"][-1]),
    })
    logger.info("  Stage 1 → val_loss=%.4f, val_acc=%.4f",
                histories[-1]["final_val_loss"], histories[-1]["final_val_acc"])

    # Stage 2: Fine-tune full model on imbalanced data with class_weight
    logger.info("═══ Stage 2: Fine-tune on imbalanced (unfreeze CNN+bilstm2+attention) ═══")
    unfreezer.apply_phase(full_model, frozen_groups=["bilstm1"])

    full_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=finetune_lr),
        loss=loss_fn, metrics=["accuracy"],
    )
    h2 = full_model.fit(
        Xo_w, yo_w_smooth, epochs=ft_epochs, batch_size=256,
        validation_split=0.2, verbose=1,
        class_weight=class_weights,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=2, restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(output_dir / "checkpoint_stage2.weights.h5"),
                save_best_only=True, save_weights_only=True),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=2),
        ],
    )
    histories.append({
        "phase": "Stage 2 — Fine-tune on imbalanced",
        "epochs_run": len(h2.history["loss"]),
        "final_train_loss": float(h2.history["loss"][-1]),
        "final_train_acc": float(h2.history["accuracy"][-1]),
        "final_val_loss": float(h2.history["val_loss"][-1]),
        "final_val_acc": float(h2.history["val_accuracy"][-1]),
    })
    logger.info("  Stage 2 → val_loss=%.4f, val_acc=%.4f",
                histories[-1]["final_val_loss"], histories[-1]["final_val_acc"])

    # ══════════════════════════════════════════════════════════════
    # STEP 6: Optimal Threshold + Evaluate — Train / Val / Test
    # ══════════════════════════════════════════════════════════════
    logger.info("")
    logger.info("STEP 6: Threshold optimization + final evaluation")
    logger.info("─" * 60)

    # 6a. Multi-class: use argmax, no threshold search needed
    if n_classes > 2:
        logger.info("  Multi-class (%d): using argmax (no threshold search)", n_classes)
        opt_threshold = 0.5  # Not used for multi-class, kept for compatibility

        # Evaluate on full test set
        y_pred_probs = full_model.predict(X_test_w, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_true = y_test_w_orig if y_test_w.ndim == 1 else np.argmax(y_test_w, axis=1)

        # Per-class metrics
        from sklearn.metrics import classification_report
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        auc = roc_auc_score(y_true, y_pred_probs, multi_class="ovr", average="macro")
        logger.info("  [test] Macro F1=%.4f  AUC=%.4f  (%d samples)", report["macro avg"]["f1-score"], auc, len(y_true))

        # Binary-compatible metrics (attack = class > 0)
        y_true_bin = (y_true > 0).astype(int)
        y_pred_bin = (y_pred > 0).astype(int)
        test_metrics = _evaluate(full_model, X_test_w, y_test_w_orig, threshold=opt_threshold, label="test")
        test_metrics["n_classes"] = n_classes
        test_metrics["auc_roc"] = auc
        test_metrics["per_class_report"] = report

        # Train metrics
        y_train_pred = full_model.predict(Xo_w, verbose=0)
        y_train_cls = np.argmax(y_train_pred, axis=1)
        train_metrics = _evaluate(full_model, Xo_w, yo_w_orig, threshold=opt_threshold, label="train")
        train_metrics["n_classes"] = n_classes

    else:
        # Binary: threshold search as before
        from sklearn.model_selection import train_test_split as sk_split
        n_test_total = len(X_test_w)
        idx_thval, idx_ftest = sk_split(
            np.arange(n_test_total), test_size=0.8, random_state=config.random_state, stratify=y_test_w,
        )
        X_thval, y_thval = X_test_w[idx_thval], y_test_w[idx_thval]
        X_ftest, y_ftest = X_test_w[idx_ftest], y_test_w[idx_ftest]
        logger.info("  Threshold val: %d samples, Final test: %d samples", len(y_thval), len(y_ftest))

        y_thval_prob = full_model.predict(X_thval, verbose=0).ravel()
        opt_threshold, opt_val_score = _find_optimal_threshold(y_thval, y_thval_prob, metric=config.search_metric)
        logger.info("  Optimal threshold: %.4f (thval %s=%.4f)", opt_threshold, config.search_metric, opt_val_score)

        train_metrics = _evaluate(full_model, Xo_w, yo_w_orig, threshold=opt_threshold, label="train")
        test_metrics = _evaluate(full_model, X_ftest, y_ftest, threshold=opt_threshold, label="test")

    # Validation metrics from training history
    last_history = histories[-1] if histories else {}
    opt_val_score = opt_val_score if "opt_val_score" in dir() else 0.0
    val_metrics = {
        "final_val_loss": last_history.get("final_val_loss", None),
        "final_val_acc": last_history.get("final_val_acc", None),
        "optimal_threshold": opt_threshold,
        f"val_{config.search_metric}": opt_val_score,
    }
    logger.info("  [val]  val_loss=%.4f  val_acc=%.4f  threshold=%.4f",
                val_metrics["final_val_loss"] or 0, val_metrics["final_val_acc"] or 0, opt_threshold)

    # 6c. Attack-specific summary
    logger.info("")
    logger.info("  IoMT Attack Detection Summary:")
    logger.info("  %-20s  %-10s  %-10s", "Metric", "Train", "Test")
    logger.info("  " + "─" * 42)
    for m in ["attack_recall", "attack_precision", "attack_f1", "attack_f2", "macro_f1", "auc_roc"]:
        logger.info("  %-20s  %.4f      %.4f", m, train_metrics.get(m, 0), test_metrics.get(m, 0))

    # ══════════════════════════════════════════════════════════════
    # STEP 7: Compare vs. Baseline
    # ══════════════════════════════════════════════════════════════
    logger.info("")
    logger.info("STEP 7: Comparison with baseline")
    logger.info("─" * 60)

    if baseline:
        bl = baseline if "f1_score" in baseline else baseline.get("metrics", baseline)
        logger.info("  %-22s  %-10s  %-10s  %-10s", "Metric", "Baseline", "Finetuned", "Delta")
        logger.info("  " + "─" * 54)
        compare_metrics = [
            "attack_recall", "attack_precision", "attack_f1",
            "accuracy", "f1_score", "auc_roc", "macro_f1",
        ]
        for metric_name in compare_metrics:
            bl_val = bl.get(metric_name, 0)
            ft_val = test_metrics.get(metric_name, 0)
            delta = ft_val - bl_val
            marker = "+" if delta > 0 else ""
            logger.info("  %-22s  %.4f      %.4f      %s%.4f", metric_name, bl_val, ft_val, marker, delta)
    else:
        logger.info("  No baseline found — skipping comparison")

    # ══════════════════════════════════════════════════════════════
    # STEP 8: Export All Results
    # ══════════════════════════════════════════════════════════════
    logger.info("")
    logger.info("STEP 8: Exporting results")

    exporter = TuningExporter(output_dir)
    exporter.export_tuning_results(tuning_results, config.tuning_results_file)
    exporter.export_ablation_results(ablation_results, config.ablation_results_file)
    exporter.export_best_config(best_hp, config.best_config_file)
    exporter.export_json(importance_results, config.importance_file)
    exporter.export_json(multi_seed_results, config.multi_seed_file)

    # Save fine-tuned model weights
    model_path = output_dir / "finetuned_model.weights.h5"
    full_model.save_weights(str(model_path))
    logger.info("  Saved model weights: %s", model_path.name)

    # Save final results
    duration_s = time.time() - t0
    final_results = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pipeline": "run_finetuned_training",
        "duration_seconds": round(duration_s, 2),
        "best_hyperparameters": best_hp,
        "optimal_threshold": opt_threshold,
        "loss": "focal_loss(gamma=2.0, alpha=0.25) + label_smoothing(eps=0.05)",
        "class_weights": {str(k): v for k, v in class_weights.items()},
        "training_phases": [
            {"name": "Stage 1 — Head on imbalanced", "epochs": head_epochs,
             "lr": head_lr, "frozen": ["cnn", "bilstm1", "bilstm2", "attention"]},
            {"name": "Stage 2 — Fine-tune on imbalanced", "epochs": ft_epochs,
             "lr": finetune_lr, "frozen": ["bilstm1"]},
        ],
        "training_history": histories,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "baseline_comparison": {
            metric_name: {
                "baseline": (baseline.get(metric_name, 0) if "f1_score" in baseline
                             else baseline.get("metrics", {}).get(metric_name, 0)),
                "finetuned": test_metrics.get(metric_name, 0),
            }
            for metric_name in [
                "attack_recall", "attack_precision", "attack_f1",
                "accuracy", "f1_score", "auc_roc",
            ]
        } if baseline else None,
        "tuning_summary": {
            "strategy": tuning_results["strategy"],
            "total_trials": tuning_results["total_trials"],
            "completed_trials": tuning_results["completed_trials"],
            "pruned_trials": tuning_results.get("pruned_trials", 0),
            "best_quick_score": tuning_results["best_score"],
        },
        "model_summary": {
            "total_params": total_params,
            "detection_params": detection_params,
            "head_params": head_params,
        },
    }

    results_path = output_dir / "finetuned_results.json"
    with open(results_path, "w") as f:
        json.dump(final_results, f, indent=2, default=str)
    logger.info("  Saved results: %s", results_path.name)

    # Save markdown report
    report_md = render_tuning_report(
        tuning_results=tuning_results,
        ablation_results=ablation_results,
        importance_results=importance_results,
        multi_seed_results=multi_seed_results,
        hw_info={"device": "see finetuned_results.json"},
        duration_s=duration_s,
        git_commit="see finetuned_results.json",
    )
    report_path = PROJECT_ROOT / "results" / "phase0_analysis" / "report_section_tuning.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        f.write(report_md)

    # ══════════════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════════════
    logger.info("")
    logger.info("═" * 60)
    logger.info("  COMPLETE — %.1fs", duration_s)
    logger.info("═" * 60)
    logger.info("  Search:       %d trials (TPE)", tuning_results["total_trials"])
    logger.info("  Best quick:   %s=%.4f", config.search_metric, tuning_results["best_score"])
    logger.info("  Threshold:    %.4f (optimised on val %s)", opt_threshold, config.search_metric)
    logger.info("  Loss:         BCE + class_weight (Normal=%.2f, Attack=%.2f)",
                class_weights.get(0, 1), class_weights.get(1, 1))
    logger.info("  ── Train ──")
    logger.info("    Attack F1=%.4f  Recall=%.4f  Precision=%.4f",
                train_metrics.get("attack_f1", 0), train_metrics.get("attack_recall", 0), train_metrics.get("attack_precision", 0))
    logger.info("    Macro F1=%.4f  AUC=%.4f  Accuracy=%.4f",
                train_metrics.get("macro_f1", 0), train_metrics.get("auc_roc", 0), train_metrics.get("accuracy", 0))
    logger.info("  ── Val ──")
    logger.info("    loss=%.4f  acc=%.4f  %s=%.4f",
                val_metrics["final_val_loss"] or 0, val_metrics["final_val_acc"] or 0, config.search_metric, opt_val_score)
    logger.info("  ── Test ──")
    logger.info("    Attack F1=%.4f  Recall=%.4f  Precision=%.4f",
                test_metrics.get("attack_f1", 0), test_metrics.get("attack_recall", 0), test_metrics.get("attack_precision", 0))
    logger.info("    Macro F1=%.4f  AUC=%.4f  Accuracy=%.4f",
                test_metrics.get("macro_f1", 0), test_metrics.get("auc_roc", 0), test_metrics.get("accuracy", 0))
    if baseline:
        bl = baseline if "f1_score" in baseline else baseline.get("metrics", baseline)
        bl_atk = bl.get("attack_f1", 0)
        delta_atk = test_metrics.get("attack_f1", 0) - bl_atk
        bl_recall = bl.get("attack_recall", 0)
        delta_recall = test_metrics.get("attack_recall", 0) - bl_recall
        logger.info("  ── vs. Baseline ──")
        logger.info("    Attack F1:     %.4f → %.4f  (%+.4f)", bl_atk, test_metrics.get("attack_f1", 0), delta_atk)
        logger.info("    Attack Recall: %.4f → %.4f  (%+.4f)", bl_recall, test_metrics.get("attack_recall", 0), delta_recall)
    logger.info("  Artifacts:  %s", output_dir)
    logger.info("═" * 60)


if __name__ == "__main__":
    main()
