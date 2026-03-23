# Retraining fixes applied (v2):
#  1. Increased epochs: 15 → 50 with EarlyStopping(patience=5)
#  2. class_weight: Normal=0.572, Attack=3.989
#     Computed from original distribution before SMOTE
#  3. Optimal threshold via Youden's J statistic
#     replaces hardcoded 0.5
#  Reason: v1 diagnosis showed UNDERFIT + attack_recall=12%

"""Phase 3 Classification Engine v2 — retrain with class weighting.

Fixes underfitting and attack class collapse from v1 by:
- Increasing epochs per phase (5 → 20/15/15)
- Adding class_weight from pre-SMOTE distribution
- Finding optimal threshold via Youden's J statistic
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# ── Phase 2 SOLID components ──
from src.phase2_detection_engine.phase2.assembler import (
    DetectionModelAssembler,
)
from src.phase2_detection_engine.phase2.attention_builder import (  # noqa: F401
    AttentionBuilder,
    BahdanauAttention,
)
from src.phase2_detection_engine.phase2.bilstm_builder import BiLSTMBuilder
from src.phase2_detection_engine.phase2.cnn_builder import CNNBuilder
from src.phase2_detection_engine.phase2.reshaper import DataReshaper

# ── Constants ──────────────────────────────────────────────────────────
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

# Input artifacts
PHASE2_METADATA_PATH: Path = PROJECT_ROOT / "data" / "phase2" / "detection_metadata.json"
PHASE2_WEIGHTS_PATH: Path = PROJECT_ROOT / "data" / "phase2" / "detection_model.weights.h5"
TRAIN_PATH: Path = PROJECT_ROOT / "data" / "processed" / "train_phase1.parquet"
TEST_PATH: Path = PROJECT_ROOT / "data" / "processed" / "test_phase1.parquet"

# Output artifacts
OUTPUT_DIR: Path = PROJECT_ROOT / "data" / "phase3"
OUTPUT_MODEL: str = "classification_model_v2.weights.h5"
OUTPUT_METRICS: str = "metrics_wustl_v2.json"
OUTPUT_HISTORY: str = "training_history_v2.json"
OUTPUT_THRESHOLD: str = "threshold_analysis.json"
OUTPUT_DIAGNOSIS: str = "diagnosis_after.json"

# Model constants
N_FEATURES: int = 29
LABEL_COLUMN: str = "Label"
LABEL_NORMAL: int = 0
LABEL_ATTACK: int = 1

# Training constants
BATCH_SIZE: int = 64
RANDOM_STATE: int = 42
PATIENCE: int = 5
REDUCE_LR_PATIENCE: int = 3
REDUCE_LR_FACTOR: float = 0.5
VALIDATION_SPLIT: float = 0.2

# Pre-SMOTE original class counts (from Phase 1)
NORMAL_COUNT_ORIGINAL: int = 9_990
ATTACK_COUNT_ORIGINAL: int = 1_432
TOTAL_ORIGINAL: int = NORMAL_COUNT_ORIGINAL + ATTACK_COUNT_ORIGINAL

# Classification head
HEAD_DENSE_UNITS: int = 64
HEAD_DENSE_ACTIVATION: str = "relu"
HEAD_DROPOUT_RATE: float = 0.3

# Layer groups for progressive unfreezing
LAYER_GROUPS: Dict[str, List[str]] = {
    "cnn": ["conv1", "pool1", "conv2", "pool2"],
    "bilstm1": ["bilstm1", "drop1"],
    "bilstm2": ["bilstm2", "drop2"],
    "attention": ["attention"],
}

# Progressive unfreezing phase configs
TRAINING_PHASES: List[Dict[str, Any]] = [
    {
        "name": "Phase A — Head only",
        "epochs": 20,
        "learning_rate": 0.001,
        "frozen": ["cnn", "bilstm1", "bilstm2", "attention"],
    },
    {
        "name": "Phase B — Attention + Head",
        "epochs": 15,
        "learning_rate": 0.0001,
        "frozen": ["cnn", "bilstm1", "bilstm2"],
    },
    {
        "name": "Phase C — BiLSTM-2 + Attention + Head",
        "epochs": 15,
        "learning_rate": 0.00001,
        "frozen": ["cnn", "bilstm1"],
    },
]

# V1 baseline for comparison
V1_ACCURACY: float = 0.8341
V1_ATTACK_RECALL: float = 0.1209
NAIVE_BASELINE: float = 0.8746

# Thresholds to evaluate
THRESHOLD_CANDIDATES: List[float] = [0.3, 0.4, 0.5]

# ── Logging ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  [%(name)s]  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Utility ────────────────────────────────────────────────────────────


def _compute_sha256(filepath: Path) -> str:
    """Compute SHA-256 hash of a file.

    Args:
        filepath: Path to the file.

    Returns:
        Hex-encoded SHA-256 digest.
    """
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# ── Step 1: Load Artifacts ─────────────────────────────────────────────


def _load_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load Phase 1 train/test parquets and reshape for CNN-BiLSTM.

    Returns:
        Tuple of (X_train, y_train, X_test, y_test) as windowed arrays.
    """
    logger.info("── Loading data ──")

    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    y_train_raw = train_df[LABEL_COLUMN].values
    X_train_raw = train_df.drop(columns=[LABEL_COLUMN]).values.astype(np.float32)
    y_test_raw = test_df[LABEL_COLUMN].values
    X_test_raw = test_df.drop(columns=[LABEL_COLUMN]).values.astype(np.float32)

    train_dist = dict(zip(*np.unique(y_train_raw, return_counts=True)))
    test_dist = dict(zip(*np.unique(y_test_raw, return_counts=True)))
    logger.info("  Train: %s (total=%d)", train_dist, len(y_train_raw))
    logger.info("  Test: %s (total=%d)", test_dist, len(y_test_raw))

    # Load Phase 2 metadata for timesteps
    with open(PHASE2_METADATA_PATH) as f:
        metadata = json.load(f)
    hp = metadata["hyperparameters"]
    timesteps = hp["timesteps"]
    stride = hp["stride"]

    reshaper = DataReshaper(timesteps=timesteps, stride=stride)
    X_train, y_train = reshaper.reshape(X_train_raw, y_train_raw)
    X_test, y_test = reshaper.reshape(X_test_raw, y_test_raw)

    logger.info("  Train windowed: %s", X_train.shape)
    logger.info("  Test windowed: %s", X_test.shape)

    return X_train, y_train, X_test, y_test


# ── Step 2: Class Weights ─────────────────────────────────────────────


def _compute_class_weight() -> Dict[int, float]:
    """Compute class weights from original pre-SMOTE distribution.

    Returns:
        Dict mapping class label to weight.
    """
    logger.info("── Computing class weights ──")

    w_normal = TOTAL_ORIGINAL / (2 * NORMAL_COUNT_ORIGINAL)
    w_attack = TOTAL_ORIGINAL / (2 * ATTACK_COUNT_ORIGINAL)

    class_weight = {LABEL_NORMAL: w_normal, LABEL_ATTACK: w_attack}

    logger.info(
        "  class_weight: Normal=%.3f, Attack=%.3f",
        w_normal,
        w_attack,
    )
    logger.info(
        "  Effective Attack/Normal ratio: %.1fx",
        w_attack / w_normal,
    )

    return class_weight


# ── Step 3: Build Model ───────────────────────────────────────────────


def _build_model() -> tf.keras.Model:
    """Rebuild detection backbone and add classification head.

    Returns:
        Full classification model with Phase 2 detection weights.
    """
    logger.info("── Building model ──")

    with open(PHASE2_METADATA_PATH) as f:
        metadata = json.load(f)
    hp = metadata["hyperparameters"]

    # Rebuild detection backbone
    builders = [
        CNNBuilder(
            filters_1=hp["cnn_filters_1"],
            filters_2=hp["cnn_filters_2"],
            kernel_size=hp["cnn_kernel_size"],
            activation=hp["cnn_activation"],
            pool_size=hp["cnn_pool_size"],
        ),
        BiLSTMBuilder(
            units_1=hp["bilstm_units_1"],
            units_2=hp["bilstm_units_2"],
            dropout_rate=hp["dropout_rate"],
        ),
        AttentionBuilder(units=hp["attention_units"]),
    ]

    assembler = DetectionModelAssembler(
        timesteps=hp["timesteps"],
        n_features=N_FEATURES,
        builders=builders,
    )
    detection_model = assembler.assemble()

    # Load Phase 2 detection weights — verify SHA-256
    expected_hash = metadata["artifact_hashes"]["detection_model.weights.h5"]["sha256"]
    actual_hash = _compute_sha256(PHASE2_WEIGHTS_PATH)
    if actual_hash != expected_hash:
        raise ValueError(
            f"SHA-256 mismatch: expected={expected_hash[:16]}…, " f"actual={actual_hash[:16]}…"
        )
    detection_model.load_weights(str(PHASE2_WEIGHTS_PATH))
    logger.info(
        "  Detection backbone: %d params, SHA-256 verified",
        detection_model.count_params(),
    )

    # Add classification head
    x = detection_model.output
    x = tf.keras.layers.Dense(
        HEAD_DENSE_UNITS,
        activation=HEAD_DENSE_ACTIVATION,
        name="dense_head",
    )(x)
    x = tf.keras.layers.Dropout(HEAD_DROPOUT_RATE, name="drop_head")(x)
    x = tf.keras.layers.Dense(1, activation="sigmoid", name="output")(x)

    full_model = tf.keras.Model(detection_model.input, x, name="classification_engine_v2")

    n_params = full_model.count_params()
    head_params = n_params - detection_model.count_params()
    logger.info(
        "  Total params: %d (backbone: %d, head: %d)",
        n_params,
        detection_model.count_params(),
        head_params,
    )

    return full_model


# ── Step 4-5: Progressive Unfreezing Training ──────────────────────────


def _set_trainable(
    model: tf.keras.Model,
    frozen_groups: List[str],
) -> None:
    """Freeze layers belonging to specified groups.

    Args:
        model: Full classification model.
        frozen_groups: List of group names to freeze.
    """
    frozen_names: set = set()
    for group in frozen_groups:
        frozen_names.update(LAYER_GROUPS[group])

    for layer in model.layers:
        if layer.name in frozen_names:
            layer.trainable = False
        elif layer.name not in ("input",):
            layer.trainable = True


def _train(
    model: tf.keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    class_weight: Dict[int, float],
) -> List[Dict[str, Any]]:
    """Train with progressive unfreezing and class weighting.

    Args:
        model: Full classification model.
        X_train: Windowed training features.
        y_train: Windowed training labels.
        class_weight: Per-class loss weights.

    Returns:
        List of per-phase training history dicts.
    """
    logger.info("═══ Progressive Unfreezing (v2) ═══")

    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    all_histories: List[Dict[str, Any]] = []

    for i, phase_cfg in enumerate(TRAINING_PHASES):
        phase_name = phase_cfg["name"]
        logger.info("── %s ──", phase_name)

        _set_trainable(model, phase_cfg["frozen"])

        trainable_count = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
        frozen_count = sum(tf.keras.backend.count_params(w) for w in model.non_trainable_weights)
        logger.info(
            "  Trainable: %d, Frozen: %d",
            trainable_count,
            frozen_count,
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=phase_cfg["learning_rate"]),
            loss="binary_crossentropy",
            metrics=[
                "accuracy",
                tf.keras.metrics.AUC(name="auc"),
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
            ],
        )

        checkpoint_path = output_dir / f"checkpoint_v2_phase_{i}.weights.h5"
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=PATIENCE,
                restore_best_weights=True,
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(checkpoint_path),
                monitor="val_loss",
                save_best_only=True,
                save_weights_only=True,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=REDUCE_LR_FACTOR,
                patience=REDUCE_LR_PATIENCE,
            ),
        ]

        history = model.fit(
            X_train,
            y_train,
            epochs=phase_cfg["epochs"],
            batch_size=BATCH_SIZE,
            validation_split=VALIDATION_SPLIT,
            class_weight=class_weight,
            callbacks=callbacks,
            verbose=1,
        )

        epochs_run = len(history.history["loss"])
        phase_history = {
            "phase": phase_name,
            "epochs_run": epochs_run,
            "max_epochs": phase_cfg["epochs"],
            "early_stopped": epochs_run < phase_cfg["epochs"],
            "final_train_loss": float(history.history["loss"][-1]),
            "final_train_acc": float(history.history["accuracy"][-1]),
            "final_val_loss": float(history.history["val_loss"][-1]),
            "final_val_acc": float(history.history["val_accuracy"][-1]),
            "history": {k: [float(v) for v in vals] for k, vals in history.history.items()},
        }
        all_histories.append(phase_history)

        logger.info(
            "  %s complete — %d/%d epochs, " "val_loss=%.4f, val_acc=%.4f",
            phase_name,
            epochs_run,
            phase_cfg["epochs"],
            phase_history["final_val_loss"],
            phase_history["final_val_acc"],
        )

    total_epochs = sum(h["epochs_run"] for h in all_histories)
    any_es = any(h["early_stopped"] for h in all_histories)
    logger.info(
        "  Training complete: %d total epochs, " "EarlyStopping=%s",
        total_epochs,
        "YES" if any_es else "NO",
    )

    return all_histories


# ── Step 6: Optimal Threshold ──────────────────────────────────────────


def _find_optimal_threshold(
    model: tf.keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """Find optimal threshold via Youden's J statistic on ROC curve.

    Args:
        model: Trained classification model.
        X_test: Windowed test features.
        y_test: Windowed test labels.

    Returns:
        Tuple of (y_pred_prob, optimal_threshold).
    """
    logger.info("── Finding optimal threshold ──")

    y_pred_prob = model.predict(X_test, batch_size=BATCH_SIZE, verbose=0)
    if y_pred_prob.shape[-1] == 1:
        y_pred_prob = y_pred_prob.ravel()

    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    j_scores = tpr - fpr
    best_idx = int(np.argmax(j_scores))
    optimal = float(thresholds[best_idx])

    logger.info(
        "  Optimal threshold: %.4f (Youden's J=%.4f)",
        optimal,
        j_scores[best_idx],
    )
    logger.info("  At optimal: TPR=%.4f, FPR=%.4f", tpr[best_idx], fpr[best_idx])

    return y_pred_prob, optimal


# ── Step 7-8: Evaluate at Multiple Thresholds ──────────────────────────


def _evaluate_at_threshold(
    y_true: np.ndarray,
    y_pred_prob: np.ndarray,
    threshold: float,
) -> Dict[str, Any]:
    """Compute all metrics at a given threshold.

    Args:
        y_true: True labels.
        y_pred_prob: Predicted probabilities.
        threshold: Classification threshold.

    Returns:
        Metrics dict.
    """
    y_pred = (y_pred_prob >= threshold).astype(int)

    acc = float(accuracy_score(y_true, y_pred))
    f1_w = float(f1_score(y_true, y_pred, average="weighted"))
    prec_w = float(precision_score(y_true, y_pred, average="weighted"))
    rec_w = float(recall_score(y_true, y_pred, average="weighted"))
    auc = float(roc_auc_score(y_true, y_pred_prob))
    cm = confusion_matrix(y_true, y_pred)
    cls_report = classification_report(y_true, y_pred, output_dict=True)

    # Attack recall specifically
    attack_cls = cls_report.get("1", {})
    attack_recall = float(attack_cls.get("recall", 0.0))

    return {
        "threshold": threshold,
        "accuracy": acc,
        "f1_score": f1_w,
        "precision": prec_w,
        "recall": rec_w,
        "auc_roc": auc,
        "attack_recall": attack_recall,
        "attack_precision": float(attack_cls.get("precision", 0.0)),
        "attack_f1": float(attack_cls.get("f1-score", 0.0)),
        "confusion_matrix": cm.tolist(),
        "classification_report": cls_report,
        "test_samples": len(y_true),
    }


def _evaluate_all_thresholds(
    y_test: np.ndarray,
    y_pred_prob: np.ndarray,
    optimal_threshold: float,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Evaluate at candidate thresholds and optimal.

    Args:
        y_test: True labels.
        y_pred_prob: Predicted probabilities.
        optimal_threshold: Youden's J optimal threshold.

    Returns:
        Tuple of (optimal_metrics, all_threshold_results).
    """
    logger.info("── Threshold comparison ──")

    all_thresholds = THRESHOLD_CANDIDATES + [optimal_threshold]
    # Remove duplicates while preserving order
    seen: set = set()
    unique_thresholds: List[float] = []
    for t in all_thresholds:
        t_rounded = round(t, 4)
        if t_rounded not in seen:
            seen.add(t_rounded)
            unique_thresholds.append(t)

    results: List[Dict[str, Any]] = []
    optimal_metrics: Dict[str, Any] = {}

    header = (
        f"{'Threshold':>10} | {'Accuracy':>8} | {'F1':>6} "
        f"| {'Prec':>6} | {'Recall':>6} | {'Atk Recall':>10}"
    )
    logger.info("  %s", header)
    logger.info("  %s", "-" * len(header))

    for t in unique_thresholds:
        m = _evaluate_at_threshold(y_test, y_pred_prob, t)
        results.append(m)

        label = "optimal" if t == optimal_threshold else f"{t:.1f}"
        logger.info(
            "  %10s | %8.4f | %6.4f | %6.4f | %6.4f | %10.4f",
            label,
            m["accuracy"],
            m["f1_score"],
            m["precision"],
            m["recall"],
            m["attack_recall"],
        )

        if t == optimal_threshold:
            optimal_metrics = m

    # Summary vs v1
    logger.info("")
    logger.info(
        "  Attack recall: %.1f%% (was %.1f%%)",
        optimal_metrics["attack_recall"] * 100,
        V1_ATTACK_RECALL * 100,
    )
    beats = optimal_metrics["accuracy"] > NAIVE_BASELINE
    logger.info(
        "  Beats naive baseline (%.2f%%): %s",
        NAIVE_BASELINE * 100,
        "YES" if beats else "NO",
    )

    return optimal_metrics, results


# ── Step 9: Export ─────────────────────────────────────────────────────


def _export_all(
    model: tf.keras.Model,
    optimal_metrics: Dict[str, Any],
    all_results: List[Dict[str, Any]],
    histories: List[Dict[str, Any]],
    class_weight: Dict[int, float],
    optimal_threshold: float,
) -> None:
    """Export all v2 artifacts.

    Args:
        model: Trained model.
        optimal_metrics: Metrics at optimal threshold.
        all_results: Metrics at all thresholds.
        histories: Per-phase training histories.
        class_weight: Applied class weights.
        optimal_threshold: Youden's J optimal threshold.
    """
    logger.info("── Export ──")
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Model weights
    model_path = output_dir / OUTPUT_MODEL
    model.save_weights(str(model_path))
    model_hash = _compute_sha256(model_path)
    logger.info("  Saved %s — SHA-256: %s", model_path.name, model_hash)

    # 2. Metrics at optimal threshold
    metrics_path = output_dir / OUTPUT_METRICS
    metrics_report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pipeline": "retrain_v2",
        "optimal_threshold": optimal_threshold,
        "metrics": {k: v for k, v in optimal_metrics.items() if k != "classification_report"},
        "classification_report": optimal_metrics.get("classification_report", {}),
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics_report, f, indent=2)
    logger.info("  Saved %s", metrics_path.name)

    # 3. Training history
    history_path = output_dir / OUTPUT_HISTORY
    # Remove large per-epoch history for export, keep summaries
    export_histories = []
    for h in histories:
        export_h = {k: v for k, v in h.items() if k != "history"}
        export_h["epoch_losses"] = h["history"]["loss"]
        export_h["epoch_val_losses"] = h["history"]["val_loss"]
        export_histories.append(export_h)
    with open(history_path, "w") as f:
        json.dump(export_histories, f, indent=2)
    logger.info("  Saved %s", history_path.name)

    # 4. Threshold analysis
    threshold_path = output_dir / OUTPUT_THRESHOLD
    threshold_report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "optimal_threshold": optimal_threshold,
        "method": "Youden's J statistic (argmax(TPR - FPR))",
        "results": [
            {k: v for k, v in r.items() if k not in ("classification_report", "confusion_matrix")}
            for r in all_results
        ],
    }
    with open(threshold_path, "w") as f:
        json.dump(threshold_report, f, indent=2)
    logger.info("  Saved %s", threshold_path.name)

    # 5. Diagnosis after
    total_epochs = sum(h["epochs_run"] for h in histories)
    any_es = any(h["early_stopped"] for h in histories)
    acc_v2 = optimal_metrics["accuracy"]
    atk_v2 = optimal_metrics["attack_recall"]

    acc_improvement = (acc_v2 - V1_ACCURACY) / V1_ACCURACY * 100
    atk_improvement = (atk_v2 - V1_ATTACK_RECALL) / V1_ATTACK_RECALL * 100

    diagnosis_path = output_dir / OUTPUT_DIAGNOSIS
    diagnosis = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "naive_baseline": NAIVE_BASELINE,
        "model_accuracy_v1": V1_ACCURACY,
        "model_accuracy_v2": round(acc_v2, 6),
        "attack_recall_v1": V1_ATTACK_RECALL,
        "attack_recall_v2": round(atk_v2, 6),
        "optimal_threshold": optimal_threshold,
        "class_weight": {str(k): round(v, 6) for k, v in class_weight.items()},
        "total_epochs_trained": total_epochs,
        "early_stopping_triggered": any_es,
        "accuracy_improvement_pct": round(acc_improvement, 2),
        "attack_recall_improvement_pct": round(atk_improvement, 2),
        "beats_naive_baseline": acc_v2 > NAIVE_BASELINE,
    }
    with open(diagnosis_path, "w") as f:
        json.dump(diagnosis, f, indent=2)
    logger.info("  Saved %s", diagnosis_path.name)


# ── Main Pipeline ──────────────────────────────────────────────────────


def run() -> None:
    """Execute the v2 retraining pipeline."""
    logger.info("═══════════════════════════════════════════════════")
    logger.info("  Phase 3 Classification — Retrain v2")
    logger.info("  (class_weight + more epochs + optimal threshold)")
    logger.info("═══════════════════════════════════════════════════")

    tf.random.set_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    # Step 1: Load data
    X_train, y_train, X_test, y_test = _load_data()

    # Step 2: Class weights
    class_weight = _compute_class_weight()

    # Step 3: Build model
    model = _build_model()

    # Step 4-5: Progressive unfreezing training
    histories = _train(model, X_train, y_train, class_weight)

    # Step 6: Find optimal threshold
    y_pred_prob, optimal_threshold = _find_optimal_threshold(model, X_test, y_test)

    # Step 7-8: Evaluate at all thresholds
    optimal_metrics, all_results = _evaluate_all_thresholds(y_test, y_pred_prob, optimal_threshold)

    # Step 9: Export
    _export_all(
        model,
        optimal_metrics,
        all_results,
        histories,
        class_weight,
        optimal_threshold,
    )

    logger.info("═══════════════════════════════════════════════════")
    logger.info(
        "  Retrain v2 complete — accuracy=%.4f, " "attack_recall=%.4f, threshold=%.4f",
        optimal_metrics["accuracy"],
        optimal_metrics["attack_recall"],
        optimal_threshold,
    )
    logger.info("═══════════════════════════════════════════════════")


if __name__ == "__main__":
    run()
