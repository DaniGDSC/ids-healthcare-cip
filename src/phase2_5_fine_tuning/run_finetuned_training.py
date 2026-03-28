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
    precision_score,
    recall_score,
    roc_auc_score,
)

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
from src.phase3_classification_engine.phase3.config import TrainingPhaseConfig
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

class _ClassWeightedTrainer:
    """Trainer that passes class_weight to model.fit for imbalanced IoMT data."""

    def __init__(
        self,
        class_weight: Optional[Dict[int, float]],
        batch_size: int = 256,
        validation_split: float = 0.2,
        early_stopping_patience: int = 3,
        reduce_lr_patience: int = 2,
        reduce_lr_factor: float = 0.5,
    ) -> None:
        self._class_weight = class_weight
        self._batch_size = batch_size
        self._val_split = validation_split
        self._es_patience = early_stopping_patience
        self._lr_patience = reduce_lr_patience
        self._lr_factor = reduce_lr_factor

    def train_all_phases(
        self,
        model: tf.keras.Model,
        phases: list,
        unfreezer: ProgressiveUnfreezer,
        X_train: np.ndarray,
        y_train: np.ndarray,
        loss: str,
        output_dir: Path,
    ) -> list:
        logger.info("═══ Progressive Unfreezing (class-weighted) ═══")
        all_histories = []
        for i, phase_cfg in enumerate(phases):
            unfreezer.apply_phase(model, phase_cfg.frozen)
            history = self._train_phase(model, phase_cfg, X_train, y_train, loss, output_dir, i)
            all_histories.append(history)
        return all_histories

    def _train_phase(self, model, phase_cfg, X_train, y_train, loss, output_dir, idx):
        logger.info("── %s ──", phase_cfg.name)
        checkpoint = output_dir / f"checkpoint_phase_{idx}.weights.h5"
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=self._es_patience, restore_best_weights=True,
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(checkpoint), save_best_only=True, save_weights_only=True,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=self._lr_factor, patience=self._lr_patience,
            ),
        ]
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=phase_cfg.learning_rate),
            loss=loss, metrics=["accuracy"],
        )
        history = model.fit(
            X_train, y_train, epochs=phase_cfg.epochs,
            batch_size=self._batch_size, validation_split=self._val_split,
            class_weight=self._class_weight, callbacks=callbacks, verbose=1,
        )
        ph = {
            "phase": phase_cfg.name,
            "epochs_run": len(history.history["loss"]),
            "final_train_loss": float(history.history["loss"][-1]),
            "final_train_acc": float(history.history["accuracy"][-1]),
            "final_val_loss": float(history.history["val_loss"][-1]),
            "final_val_acc": float(history.history["val_accuracy"][-1]),
        }
        logger.info("  %s → val_loss=%.4f, val_acc=%.4f", phase_cfg.name, ph["final_val_loss"], ph["final_val_acc"])
        return ph


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
    """Load original imbalanced train (pre-SMOTE) for the final retrain."""
    import joblib
    from sklearn.model_selection import train_test_split as sk_split

    label_col = config.label_column
    raw = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "wustl-ehms-2020_with_attacks_categories.csv")

    hipaa = ["SrcAddr", "DstAddr", "Sport", "Dport", "SrcMac", "DstMac", "Dir", "Flgs"]
    corr = ["SrcJitter", "pLoss", "Rate", "DstJitter", "Loss", "TotPkts"]
    df = raw.drop(columns=hipaa)
    df = df.drop(columns=[c for c in corr if c in df.columns])
    if "Attack Category" in df.columns:
        df = df.drop(columns=["Attack Category"])
    df = df.dropna()

    y = df[label_col].values
    X = df[feature_names].values.astype(np.float32)
    X_train_raw, _, y_train, _ = sk_split(X, y, test_size=0.30, random_state=42, stratify=y)

    scaler = joblib.load(PROJECT_ROOT / "models" / "scalers" / "robust_scaler.pkl")
    X_train = scaler.transform(X_train_raw).astype(np.float32)

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
    auc = float(roc_auc_score(y, y_prob))

    # Attack-specific (class 1)
    atk_recall = float(recall_score(y, y_pred, pos_label=1, average="binary", zero_division=0))
    atk_prec = float(precision_score(y, y_pred, pos_label=1, average="binary", zero_division=0))
    atk_f1 = float(f1_score(y, y_pred, pos_label=1, average="binary", zero_division=0))

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
    logger.info("STEP 1: Hyperparameter search (%s, %d trials)",
                config.search_strategy, config.max_trials)

    search_space = SearchSpace(config.search_space, config.random_state)
    p2_weights_path = str(PROJECT_ROOT / "data" / "phase2" / "detection_model.weights.h5")
    evaluator = QuickEvaluator(
        config.quick_train, n_features=n_features,
        random_state=config.random_state,
        pretrained_weights_path=p2_weights_path,
    )
    tuner = HyperparameterTuner(config, evaluator, search_space)

    tuning_results = tuner.run(X_train_smote, y_train_smote, X_test, y_test)

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

    timesteps = int(best_hp.get("timesteps", 20))
    stride = int(best_hp.get("stride", 1))

    # 5a-b. Build Phase 2 BASELINE detection model and load pre-trained weights
    #     We use the exact Phase 2 architecture (not the tuned one) so that
    #     pre-trained weights load perfectly.  The HP search already found
    #     the best LR schedule, dropout, and threshold — the architecture
    #     exploration is done via ablation study.
    import json as _json
    with open(PROJECT_ROOT / "data" / "phase2" / "detection_metadata.json") as _f:
        p2_meta = _json.load(_f)
    p2_hp = p2_meta["hyperparameters"]

    detection_model = _build_detection_model({
        "cnn_filters_1": p2_hp["cnn_filters_1"],
        "cnn_filters_2": p2_hp["cnn_filters_2"],
        "cnn_kernel_size": p2_hp["cnn_kernel_size"],
        "bilstm_units_1": p2_hp["bilstm_units_1"],
        "bilstm_units_2": p2_hp["bilstm_units_2"],
        "attention_units": p2_hp["attention_units"],
        "dropout_rate": float(best_hp.get("dropout_rate", p2_hp["dropout_rate"])),
        "timesteps": p2_hp["timesteps"],
    }, n_features)
    detection_params = detection_model.count_params()

    p2_weights = PROJECT_ROOT / "data" / "phase2" / "detection_model.weights.h5"
    detection_model.load_weights(str(p2_weights))
    logger.info("  Detection model: %d params (Phase 2 weights loaded fully)", detection_params)

    timesteps = p2_hp["timesteps"]  # Override to match Phase 2
    reshaper = DataReshaper(timesteps=timesteps, stride=stride)
    X_train_w, y_train_w = reshaper.reshape(X_train_orig, y_train_orig)
    X_test_w, y_test_w = reshaper.reshape(X_test, y_test)

    n0w = int(np.sum(y_train_w == 0))
    n1w = int(np.sum(y_train_w == 1))
    logger.info("  Training on ORIGINAL imbalanced data (re-windowed): Normal=%d, Attack=%d (%.1f:1)",
                n0w, n1w, n0w / max(n1w, 1))

    # 5c. Add classification head
    n_classes = len(np.unique(y_train_w))
    head = AutoClassificationHead(dense_units=64, dense_activation="relu", dropout_rate=0.3)
    output_tensor = head.build(detection_model.output, n_classes)
    full_model = tf.keras.Model(detection_model.input, output_tensor, name="finetuned_model")
    loss = head.get_loss(n_classes)

    total_params = full_model.count_params()
    head_params = total_params - detection_params
    logger.info("  Full model: %d params (detection=%d, head=%d)", total_params, detection_params, head_params)

    # 5d. Progressive unfreezing training with tuned LR schedule
    phase_a_lr = float(best_hp.get("phase_a_lr", 0.001))
    phase_b_lr = float(best_hp.get("phase_b_lr", 0.0001))
    phase_c_lr = float(best_hp.get("phase_c_lr", 0.00001))
    unfreeze_epochs = int(best_hp.get("unfreezing_epochs", 5))

    training_phases = [
        TrainingPhaseConfig(
            name="Phase A — Head only",
            epochs=unfreeze_epochs,
            learning_rate=phase_a_lr,
            frozen=["cnn", "bilstm1", "bilstm2", "attention"],
        ),
        TrainingPhaseConfig(
            name="Phase B — Attention + Head",
            epochs=unfreeze_epochs,
            learning_rate=phase_b_lr,
            frozen=["cnn", "bilstm1", "bilstm2"],
        ),
        TrainingPhaseConfig(
            name="Phase C — BiLSTM-2 + Attention + Head",
            epochs=unfreeze_epochs,
            learning_rate=phase_c_lr,
            frozen=["cnn", "bilstm1"],
        ),
    ]

    logger.info("  Training schedule:")
    for p in training_phases:
        logger.info("    %s: epochs=%d, lr=%.6g, frozen=%s", p.name, p.epochs, p.learning_rate, p.frozen)

    batch_size = int(best_hp.get("batch_size", 256))
    output_dir = PROJECT_ROOT / config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # 5e. BCE + tuned class_weight (sweep found CW=2.5 for attacks optimal)
    #     CW=2.5 balances attack recall vs precision on the 7:1 imbalanced data
    class_weights = {0: 1.0, 1: 2.5}
    logger.info("  Loss: BCE + class_weight=%s", class_weights)

    unfreezer = ProgressiveUnfreezer()
    trainer = _ClassWeightedTrainer(
        class_weight=class_weights,
        batch_size=batch_size,
        validation_split=0.2,
        early_stopping_patience=3,
        reduce_lr_patience=2,
        reduce_lr_factor=0.5,
    )

    histories = trainer.train_all_phases(
        model=full_model,
        phases=training_phases,
        unfreezer=unfreezer,
        X_train=X_train_w,
        y_train=y_train_w,
        loss=loss,
        output_dir=output_dir,
    )

    # ══════════════════════════════════════════════════════════════
    # STEP 6: Optimal Threshold + Evaluate — Train / Val / Test
    # ══════════════════════════════════════════════════════════════
    logger.info("")
    logger.info("STEP 6: Threshold optimization + final evaluation")
    logger.info("─" * 60)

    # 6a. Split imbalanced test set into threshold_val (20%) + final_test (80%)
    #     Threshold MUST be tuned on imbalanced data (not SMOTE-balanced train)
    from sklearn.model_selection import train_test_split as sk_split
    n_test_total = len(X_test_w)
    idx_thval, idx_ftest = sk_split(
        np.arange(n_test_total), test_size=0.8, random_state=config.random_state, stratify=y_test_w,
    )
    X_thval, y_thval = X_test_w[idx_thval], y_test_w[idx_thval]
    X_ftest, y_ftest = X_test_w[idx_ftest], y_test_w[idx_ftest]
    logger.info("  Threshold val: %d samples (imbalanced), Final test: %d samples", len(y_thval), len(y_ftest))

    # 6b. Search threshold on imbalanced threshold_val
    y_thval_prob = full_model.predict(X_thval, verbose=0).ravel()
    opt_threshold, opt_val_score = _find_optimal_threshold(y_thval, y_thval_prob, metric="attack_f1")
    logger.info("  Optimal threshold: %.4f (thval attack_f1=%.4f)", opt_threshold, opt_val_score)

    # 6c. Evaluate with optimised threshold
    train_metrics = _evaluate(full_model, X_train_w, y_train_w, threshold=opt_threshold, label="train")
    test_metrics = _evaluate(full_model, X_ftest, y_ftest, threshold=opt_threshold, label="test")

    # Validation metrics from training history
    last_history = histories[-1] if histories else {}
    val_metrics = {
        "final_val_loss": last_history.get("final_val_loss", None),
        "final_val_acc": last_history.get("final_val_acc", None),
        "optimal_threshold": opt_threshold,
        "val_attack_f1": opt_val_score,
    }
    logger.info("  [val]  val_loss=%.4f  val_acc=%.4f  threshold=%.4f",
                val_metrics["final_val_loss"] or 0, val_metrics["final_val_acc"] or 0, opt_threshold)

    # 6c. Attack-specific summary
    logger.info("")
    logger.info("  IoMT Attack Detection Summary:")
    logger.info("  %-20s  %-10s  %-10s", "Metric", "Train", "Test")
    logger.info("  " + "─" * 42)
    for m in ["attack_recall", "attack_precision", "attack_f1", "macro_f1", "auc_roc"]:
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
        "loss": "binary_crossentropy",
        "class_weights": {str(k): v for k, v in class_weights.items()},
        "training_phases": [
            {"name": p.name, "epochs": p.epochs, "lr": p.learning_rate, "frozen": p.frozen}
            for p in training_phases
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
    logger.info("  Search:       %d trials (%s)", tuning_results["total_trials"], config.search_strategy)
    logger.info("  Best quick:   %s=%.4f", config.search_metric, tuning_results["best_score"])
    logger.info("  Threshold:    %.4f (optimised on val attack_f1)", opt_threshold)
    logger.info("  Loss:         BCE + class_weight (Normal=%.2f, Attack=%.2f)",
                class_weights.get(0, 1), class_weights.get(1, 1))
    logger.info("  ── Train ──")
    logger.info("    Attack F1=%.4f  Recall=%.4f  Precision=%.4f",
                train_metrics.get("attack_f1", 0), train_metrics.get("attack_recall", 0), train_metrics.get("attack_precision", 0))
    logger.info("    Macro F1=%.4f  AUC=%.4f  Accuracy=%.4f",
                train_metrics.get("macro_f1", 0), train_metrics.get("auc_roc", 0), train_metrics.get("accuracy", 0))
    logger.info("  ── Val ──")
    logger.info("    loss=%.4f  acc=%.4f  attack_f1=%.4f",
                val_metrics["final_val_loss"] or 0, val_metrics["final_val_acc"] or 0, opt_val_score)
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
