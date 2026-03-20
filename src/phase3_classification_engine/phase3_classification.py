#!/usr/bin/env python3
"""Phase 3 Classification Engine — progressive unfreezing on detection backbone.

Loads Phase 2 detection model weights (474K params, no classification head),
adds a classification head (Dense→Dropout→Dense), and trains with 3-phase
progressive unfreezing:

    Phase A: Head only           (lr=1e-3, freeze all detection layers)
    Phase B: Attention + Head    (lr=1e-4, unfreeze attention)
    Phase C: BiLSTM-2 + Attn + Head (lr=1e-5, unfreeze bilstm2+drop2)

Usage::

    python -m src.phase3_classification_engine.phase3_classification
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# ── Phase 2 SOLID components (reused, NOT duplicated) ─────────────
from src.phase2_detection_engine.phase2.assembler import DetectionModelAssembler
from src.phase2_detection_engine.phase2.attention_builder import (  # noqa: F401
    AttentionBuilder,
    BahdanauAttention,
)
from src.phase2_detection_engine.phase2.bilstm_builder import BiLSTMBuilder
from src.phase2_detection_engine.phase2.cnn_builder import CNNBuilder
from src.phase2_detection_engine.phase2.reshaper import DataReshaper

# ── Phase 3 SOLID components (cross-dataset — reused, NOT duplicated) ──
from src.phase3_classification_engine.phase3.cross_dataset import CICIoMTLoader
from src.phase3_classification_engine.phase3.cross_dataset_report import (
    build_comparison_report,
    render_cross_dataset_report,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
CONFIG_PATH: Path = PROJECT_ROOT / "config" / "phase3_config.yaml"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  [%(name)s]  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Layer groups for progressive unfreezing
_LAYER_GROUPS: Dict[str, List[str]] = {
    "cnn": ["conv1", "pool1", "conv2", "pool2"],
    "bilstm1": ["bilstm1", "drop1"],
    "bilstm2": ["bilstm2", "drop2"],
    "attention": ["attention"],
}


# ===================================================================
# Utility Functions
# ===================================================================


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


def _get_git_commit() -> str:
    """Get current git commit hash for model versioning."""
    try:
        result = subprocess.run(  # noqa: S603, S607
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=str(PROJECT_ROOT),
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except (subprocess.SubprocessError, FileNotFoundError):
        return "unknown"


def _detect_hardware() -> Dict[str, str]:
    """Detect GPU/CPU availability and return hardware info dict."""
    gpus = tf.config.list_physical_devices("GPU")

    if gpus:
        device_name = gpus[0].name
        logger.info("  Training on GPU: %s", device_name)
        cuda_version = getattr(tf.sysconfig, "get_build_info", lambda: {})()
        info = {
            "device": f"GPU: {device_name}",
            "cuda": cuda_version.get("cuda_version", "N/A"),
        }
    else:
        cpu_info = platform.processor() or platform.machine()
        logger.info("  CPU fallback: %s", cpu_info)
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
        info = {"device": f"CPU: {cpu_info}", "cuda": "N/A (CPU execution)"}

    info["tensorflow"] = tf.__version__
    info["python"] = platform.python_version()
    info["platform"] = platform.platform()
    return info


# ===================================================================
# Phase 2 Artifact Verification
# ===================================================================


def _verify_phase2_artifacts(config: Dict[str, Any]) -> Dict[str, Any]:
    """Verify Phase 2 artifacts via SHA-256 from detection_metadata.json.

    Args:
        config: Parsed phase3_config.yaml.

    Returns:
        Loaded detection_metadata dict.

    Raises:
        FileNotFoundError: If required artifacts are missing.
        ValueError: If SHA-256 verification fails.
    """
    logger.info("── Phase 2 artifact verification ──")

    metadata_path = PROJECT_ROOT / config["data"]["phase2_metadata"]
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing: {metadata_path}")

    with open(metadata_path) as f:
        metadata = json.load(f)

    phase2_dir = PROJECT_ROOT / config["data"]["phase2_dir"]
    for artifact_name, hash_info in metadata["artifact_hashes"].items():
        artifact_path = phase2_dir / artifact_name
        if not artifact_path.exists():
            raise FileNotFoundError(f"Missing: {artifact_path}")

        expected = hash_info["sha256"]
        actual = _compute_sha256(artifact_path)
        if actual != expected:
            raise ValueError(
                f"SHA-256 mismatch for {artifact_name}: "
                f"expected={expected[:16]}…, actual={actual[:16]}…"
            )
        logger.info("  ✓ SHA-256 verified: %s", artifact_name)

    logger.info("  All Phase 2 artifacts verified.")
    return metadata


# ===================================================================
# Data Loading
# ===================================================================


def _load_phase1_data(
    config: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load Phase 1 train/test parquets and separate features/labels.

    Args:
        config: Parsed phase3_config.yaml.

    Returns:
        (X_train, y_train, X_test, y_test) as numpy arrays.
    """
    logger.info("── Loading Phase 1 data ──")

    label_col = config["data"]["label_column"]

    train_path = PROJECT_ROOT / config["data"]["phase1_train"]
    test_path = PROJECT_ROOT / config["data"]["phase1_test"]

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    y_train = train_df[label_col].values
    X_train = train_df.drop(columns=[label_col]).values.astype(np.float32)

    y_test = test_df[label_col].values
    X_test = test_df.drop(columns=[label_col]).values.astype(np.float32)

    logger.info("  Train: %s, Test: %s", X_train.shape, X_test.shape)
    dist = dict(zip(*np.unique(y_train, return_counts=True)))
    logger.info("  Label distribution (train): %s", dist)
    return X_train, y_train, X_test, y_test


# ===================================================================
# Model Construction
# ===================================================================


def _rebuild_detection_model(metadata: Dict[str, Any], config: Dict[str, Any]) -> tf.keras.Model:
    """Rebuild Phase 2 detection model architecture and load weights.

    Architecture is rebuilt from Phase 2 SOLID builders, then weights
    loaded from ``detection_model.weights.h5``.  Does NOT retrain.

    Args:
        metadata: Loaded detection_metadata.json.
        config: Parsed phase3_config.yaml.

    Returns:
        Detection model with loaded weights (no classification head).
    """
    logger.info("── Rebuilding detection model ──")

    hp = metadata["hyperparameters"]

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
        n_features=29,
        builders=builders,
    )
    detection_model = assembler.assemble()

    weights_path = PROJECT_ROOT / config["data"]["phase2_dir"] / "detection_model.weights.h5"
    detection_model.load_weights(str(weights_path))
    logger.info("  Loaded weights from %s", weights_path.name)
    logger.info("  Detection model params: %d", detection_model.count_params())

    return detection_model


def _auto_classify_head(
    y_train: np.ndarray,
) -> Tuple[int, str, str]:
    """Auto-detect classification type from training labels.

    Args:
        y_train: Training labels.

    Returns:
        (output_units, activation, loss) tuple.
    """
    n_classes = len(np.unique(y_train))
    logger.info("  Auto-detected %d classes", n_classes)

    if n_classes == 2:
        return 1, "sigmoid", "binary_crossentropy"
    else:
        return n_classes, "softmax", "categorical_crossentropy"


def _build_full_model(
    detection_model: tf.keras.Model,
    output_units: int,
    activation: str,
    head_config: Dict[str, Any],
) -> tf.keras.Model:
    """Extend detection model with classification head.

    Args:
        detection_model: Pre-trained detection model (output: 128-dim).
        output_units: Number of output units (1 for binary).
        activation: Output activation ("sigmoid" or "softmax").
        head_config: Classification head hyperparameters.

    Returns:
        Full classification model.
    """
    logger.info("── Building classification model ──")

    x = detection_model.output
    x = tf.keras.layers.Dense(
        head_config["dense_units"],
        activation=head_config["dense_activation"],
        name="dense_head",
    )(x)
    x = tf.keras.layers.Dropout(
        head_config["dropout_rate"],
        name="drop_head",
    )(x)
    x = tf.keras.layers.Dense(
        output_units,
        activation=activation,
        name="output",
    )(x)

    full_model = tf.keras.Model(detection_model.input, x, name="classification_engine")

    total_params = full_model.count_params()
    head_params = total_params - detection_model.count_params()
    logger.info("  Total params: %d (head: %d)", total_params, head_params)

    return full_model


# ===================================================================
# Progressive Unfreezing
# ===================================================================


def _set_trainable(model: tf.keras.Model, frozen_groups: List[str]) -> None:
    """Freeze layers belonging to specified groups.

    Args:
        model: Full classification model.
        frozen_groups: List of group names to freeze (e.g., ["cnn", "bilstm1"]).
    """
    frozen_names: set = set()
    for group in frozen_groups:
        frozen_names.update(_LAYER_GROUPS[group])

    for layer in model.layers:
        if layer.name in frozen_names:
            layer.trainable = False
        elif layer.name not in ("input",):
            layer.trainable = True


def _train_phase(
    model: tf.keras.Model,
    phase_cfg: Dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    loss: str,
    batch_size: int,
    val_split: float,
    callbacks: List[tf.keras.callbacks.Callback],
) -> tf.keras.callbacks.History:
    """Run a single training phase (compile + fit).

    Args:
        model: Full classification model.
        phase_cfg: Phase config (name, epochs, learning_rate, frozen).
        X_train: Windowed training features.
        y_train: Windowed training labels.
        loss: Loss function name.
        batch_size: Training batch size.
        val_split: Validation split fraction.
        callbacks: Keras callbacks.

    Returns:
        Keras History object.
    """
    phase_name = phase_cfg["name"]
    logger.info("── %s ──", phase_name)

    # Set trainable/frozen layers
    _set_trainable(model, phase_cfg["frozen"])

    trainable_count = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
    frozen_count = sum(tf.keras.backend.count_params(w) for w in model.non_trainable_weights)
    logger.info("  Trainable: %d, Frozen: %d", trainable_count, frozen_count)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=phase_cfg["learning_rate"]),
        loss=loss,
        metrics=["accuracy"],
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=phase_cfg["epochs"],
        batch_size=batch_size,
        validation_split=val_split,
        callbacks=callbacks,
        verbose=1,
    )
    return history


def _progressive_unfreezing(
    model: tf.keras.Model,
    config: Dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    loss: str,
) -> List[Dict[str, Any]]:
    """Orchestrate 3-phase progressive unfreezing.

    Args:
        model: Full classification model.
        config: Parsed phase3_config.yaml.
        X_train: Windowed training features.
        y_train: Windowed training labels.
        loss: Loss function name.

    Returns:
        List of per-phase training history dicts.
    """
    logger.info("═══ Progressive Unfreezing ═══")

    training_cfg = config["training"]
    callback_cfg = config["callbacks"]
    output_cfg = config["output"]
    output_dir = PROJECT_ROOT / output_cfg["output_dir"]

    all_histories: List[Dict[str, Any]] = []

    for i, phase_cfg in enumerate(training_cfg["phases"]):
        checkpoint_path = output_dir / f"checkpoint_phase_{i}.weights.h5"

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=callback_cfg["early_stopping_patience"],
                restore_best_weights=True,
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(checkpoint_path),
                save_best_only=True,
                save_weights_only=True,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=callback_cfg["reduce_lr_factor"],
                patience=callback_cfg["reduce_lr_patience"],
            ),
        ]

        history = _train_phase(
            model=model,
            phase_cfg=phase_cfg,
            X_train=X_train,
            y_train=y_train,
            loss=loss,
            batch_size=training_cfg["batch_size"],
            val_split=training_cfg["validation_split"],
            callbacks=callbacks,
        )

        phase_history = {
            "phase": phase_cfg["name"],
            "epochs_run": len(history.history["loss"]),
            "final_train_loss": float(history.history["loss"][-1]),
            "final_train_acc": float(history.history["accuracy"][-1]),
            "final_val_loss": float(history.history["val_loss"][-1]),
            "final_val_acc": float(history.history["val_accuracy"][-1]),
            "history": {k: [float(v) for v in vals] for k, vals in history.history.items()},
        }
        all_histories.append(phase_history)

        logger.info(
            "  %s → val_loss=%.4f, val_acc=%.4f",
            phase_cfg["name"],
            phase_history["final_val_loss"],
            phase_history["final_val_acc"],
        )

    return all_histories


# ===================================================================
# Evaluation
# ===================================================================


def _evaluate(
    model: tf.keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float,
) -> Dict[str, Any]:
    """Compute all evaluation metrics on the test set.

    Args:
        model: Trained classification model.
        X_test: Windowed test features.
        y_test: Windowed test labels.
        threshold: Classification threshold for binary.

    Returns:
        Dict of metrics (accuracy, f1, precision, recall, auc_roc, confusion_matrix).
    """
    logger.info("── Evaluation ──")

    y_pred_prob = model.predict(X_test, verbose=0)

    if y_pred_prob.shape[-1] == 1:
        y_pred_prob = y_pred_prob.ravel()
        y_pred = (y_pred_prob > threshold).astype(int)
    else:
        y_pred = np.argmax(y_pred_prob, axis=1)

    acc = float(accuracy_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred, average="weighted"))
    prec = float(precision_score(y_test, y_pred, average="weighted"))
    rec = float(recall_score(y_test, y_pred, average="weighted"))
    auc = float(roc_auc_score(y_test, y_pred_prob))
    cm = confusion_matrix(y_test, y_pred)
    cls_report = classification_report(y_test, y_pred, output_dict=True)

    logger.info("  Accuracy:  %.4f", acc)
    logger.info("  F1-score:  %.4f", f1)
    logger.info("  Precision: %.4f", prec)
    logger.info("  Recall:    %.4f", rec)
    logger.info("  AUC-ROC:   %.4f", auc)

    return {
        "accuracy": acc,
        "f1_score": f1,
        "precision": prec,
        "recall": rec,
        "auc_roc": auc,
        "confusion_matrix": cm.tolist(),
        "classification_report": cls_report,
        "threshold": threshold,
        "test_samples": len(y_test),
    }


# ===================================================================
# Cross-Dataset Validation
# ===================================================================


def _run_cross_dataset_validation(
    model: tf.keras.Model,
    wustl_metrics: Dict[str, Any],
    config: Dict[str, Any],
    reshaper: DataReshaper,
) -> Tuple[
    "Dict[str, Any] | None",
    "Dict[str, Any] | None",
    "Dict[str, Any] | None",
]:
    """Run CICIoMT2024 cross-dataset validation (steps 6-8).

    Args:
        model: Trained classification model.
        wustl_metrics: WUSTL test set metrics for comparison.
        config: Parsed phase3_config.yaml.
        reshaper: DataReshaper (same timesteps/stride as training).

    Returns:
        Tuple of (ciciomt_metrics, load_report, comparison) or
        (None, None, None) if skipped.
    """
    cross_cfg = config.get("cross_dataset")
    if not cross_cfg or not cross_cfg.get("enabled", False):
        logger.info("  Cross-dataset validation: DISABLED in config")
        return None, None, None

    csv_path = PROJECT_ROOT / cross_cfg["csv_path"]
    loader = CICIoMTLoader(
        csv_path=csv_path,
        column_mapping=cross_cfg.get("column_mapping", {}),
        label_column=cross_cfg.get("label_column", "Label"),
        label_mapping=cross_cfg.get("label_mapping"),
        scaler_path=PROJECT_ROOT / cross_cfg["scaler_path"],
        wustl_train_path=PROJECT_ROOT / config["data"]["phase1_train"],
    )

    if not loader.is_available():
        logger.info(
            "  Cross-dataset: CICIoMT2024 CSV not found at %s — SKIPPED",
            csv_path,
        )
        return None, None, None

    logger.info("── Cross-Dataset Validation: CICIoMT2024 ──")

    # Step 6: Load and prepare CICIoMT2024
    X_scaled, y, load_report = loader.load_and_prepare()

    if y is None:
        logger.warning("  No labels in CICIoMT2024 — cannot evaluate")
        return None, None, None

    # Reshape to sliding windows (same as training)
    X_windowed, y_windowed = reshaper.reshape(X_scaled, y)
    logger.info("  CICIoMT2024 windowed: %s", X_windowed.shape)

    # Step 7: Evaluate on CICIoMT2024
    ciciomt_metrics = _evaluate(
        model=model,
        X_test=X_windowed,
        y_test=y_windowed,
        threshold=config["evaluation"]["threshold"],
    )

    # Step 8: Build comparison
    comparison = build_comparison_report(wustl_metrics, ciciomt_metrics)

    delta_acc = comparison["accuracy"]["delta_pct"]
    delta_f1 = comparison["f1_score"]["delta_pct"]
    logger.info(
        "  Generalization gap: accuracy delta=%.1f%%, F1 delta=%.1f%%",
        delta_acc,
        delta_f1,
    )

    return ciciomt_metrics, load_report, comparison


# ===================================================================
# Export Artifacts
# ===================================================================


def _export_artifacts(
    model: tf.keras.Model,
    metrics: Dict[str, Any],
    histories: List[Dict[str, Any]],
    config: Dict[str, Any],
    hw_info: Dict[str, str],
    duration_s: float,
) -> Path:
    """Save model weights, metrics, confusion matrix, and training history.

    Args:
        model: Trained classification model.
        metrics: Evaluation metrics dict.
        histories: Per-phase training history list.
        config: Parsed phase3_config.yaml.
        hw_info: Hardware info dict.
        duration_s: Pipeline execution time in seconds.

    Returns:
        Output directory Path.
    """
    logger.info("── Exporting artifacts ──")

    output_cfg = config["output"]
    output_dir = PROJECT_ROOT / output_cfg["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Model weights
    weights_path = output_dir / output_cfg["model_file"]
    model.save_weights(str(weights_path))
    logger.info("  Saved model weights: %s", weights_path.name)

    # 2. Metrics report
    metrics_report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pipeline": "phase3_classification",
        "git_commit": _get_git_commit(),
        "hardware": hw_info,
        "duration_seconds": round(duration_s, 2),
        "metrics": metrics,
        "model_summary": {
            "name": model.name,
            "total_params": model.count_params(),
            "layers": len(model.layers),
        },
    }
    metrics_path = output_dir / output_cfg["metrics_file"]
    with open(metrics_path, "w") as f:
        json.dump(metrics_report, f, indent=2)
    logger.info("  Saved metrics: %s", metrics_path.name)

    # 3. Confusion matrix CSV
    cm = np.array(metrics["confusion_matrix"])
    cm_path = output_dir / output_cfg["confusion_matrix_file"]
    labels = ["Normal", "Attack"] if cm.shape[0] == 2 else [str(i) for i in range(cm.shape[0])]
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_df.index.name = "Actual"
    cm_df.to_csv(cm_path)
    logger.info("  Saved confusion matrix: %s", cm_path.name)

    # 4. Training history
    history_path = output_dir / output_cfg["history_file"]
    with open(history_path, "w") as f:
        json.dump(histories, f, indent=2)
    logger.info("  Saved training history: %s", history_path.name)

    return output_dir


def _export_cross_dataset_artifacts(
    ciciomt_metrics: Dict[str, Any],
    load_report: Dict[str, Any],
    comparison: Dict[str, Any],
    config: Dict[str, Any],
) -> None:
    """Export CICIoMT2024 cross-dataset validation artifacts.

    Args:
        ciciomt_metrics: CICIoMT2024 evaluation metrics.
        load_report: CICIoMTLoader load report.
        comparison: Per-metric delta comparison.
        config: Parsed phase3_config.yaml.
    """
    logger.info("── Exporting cross-dataset artifacts ──")

    cross_cfg = config["cross_dataset"]
    output_dir = PROJECT_ROOT / config["output"]["output_dir"]

    # 1. CICIoMT2024 metrics
    ciciomt_report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pipeline": "cross_dataset_ciciomt2024",
        "metrics": ciciomt_metrics,
        "load_report": load_report,
    }
    metrics_path = output_dir / cross_cfg["metrics_file"]
    with open(metrics_path, "w") as f:
        json.dump(ciciomt_report, f, indent=2)
    logger.info("  Saved: %s", metrics_path.name)

    # 2. CICIoMT2024 confusion matrix
    cm = np.array(ciciomt_metrics["confusion_matrix"])
    labels = ["Normal", "Attack"] if cm.shape[0] == 2 else [str(i) for i in range(cm.shape[0])]
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_df.index.name = "Actual"
    cm_path = output_dir / cross_cfg["confusion_matrix_file"]
    cm_df.to_csv(cm_path)
    logger.info("  Saved: %s", cm_path.name)

    # 3. Comparison report
    comp_path = output_dir / cross_cfg["comparison_report_file"]
    with open(comp_path, "w") as f:
        json.dump(comparison, f, indent=2)
    logger.info("  Saved: %s", comp_path.name)


# ===================================================================
# Report Generation
# ===================================================================


def _generate_report(
    model: tf.keras.Model,
    metrics: Dict[str, Any],
    histories: List[Dict[str, Any]],
    config: Dict[str, Any],
    hw_info: Dict[str, str],
    duration_s: float,
    detection_params: int,
) -> None:
    """Render §6.1 Classification Engine report as markdown.

    Args:
        model: Trained classification model.
        metrics: Evaluation metrics dict.
        histories: Per-phase training history list.
        config: Parsed phase3_config.yaml.
        hw_info: Hardware info dict.
        duration_s: Pipeline execution time in seconds.
        detection_params: Detection model parameter count.
    """
    logger.info("── Generating classification report ──")

    head_params = model.count_params() - detection_params
    cm = metrics["confusion_matrix"]
    head_cfg = config["classification_head"]

    # §6.1.1 Architecture
    arch_diagram = (
        "```\n"
        "Phase 1 parquets (19980×29, 4896×29)\n"
        "  ↓ reshape (timesteps=20, stride=1)\n"
        "Windows (19961×20×29, 4877×20×29)\n"
        "  ↓ CNN → BiLSTM → Attention (474,496 params, frozen/unfrozen)\n"
        "Context vectors (batch, 128)\n"
        f"  ↓ Dense({head_cfg['dense_units']}, {head_cfg['dense_activation']})\n"
        f"  ↓ Dropout({head_cfg['dropout_rate']})\n"
        "  ↓ Dense(1, sigmoid)\n"
        "Predictions (batch, 1)\n"
        "```"
    )

    # §6.1.2 Progressive unfreezing table
    phase_rows = ""
    for phase_cfg in config["training"]["phases"]:
        frozen = ", ".join(phase_cfg["frozen"])
        trainable_groups = [g for g in _LAYER_GROUPS if g not in phase_cfg["frozen"]]
        trainable = ", ".join(trainable_groups) if trainable_groups else "—"
        phase_rows += (
            f"| {phase_cfg['name']} | {phase_cfg['epochs']} "
            f"| {phase_cfg['learning_rate']} | {frozen} "
            f"| {trainable} + head |\n"
        )

    # §6.1.3 Training history table
    hist_rows = ""
    for h in histories:
        hist_rows += (
            f"| {h['phase']} | {h['epochs_run']} "
            f"| {h['final_train_loss']:.4f} | {h['final_train_acc']:.4f} "
            f"| {h['final_val_loss']:.4f} | {h['final_val_acc']:.4f} |\n"
        )

    # §6.1.5 Confusion matrix
    if len(cm) == 2:
        tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
        cm_table = (
            "| | Predicted Normal | Predicted Attack |\n"
            "|---|---|---|\n"
            f"| **Actual Normal** | TN={tn} | FP={fp} |\n"
            f"| **Actual Attack** | FN={fn} | TP={tp} |"
        )
    else:
        cm_table = "See `confusion_matrix.csv` for full matrix."

    report = f"""## 6.1 Classification Engine Results

This section documents the Phase 3 Classification Engine: architecture,
progressive unfreezing strategy, training history, and evaluation metrics.

### 6.1.1 Classification Architecture

{arch_diagram}

| Component | Parameters |
|-----------|-----------|
| Detection backbone (CNN→BiLSTM→Attention) | {detection_params:,} |
| Classification head (Dense→Dropout→Dense) | {head_params:,} |
| **Total** | **{model.count_params():,}** |

### 6.1.2 Progressive Unfreezing Strategy

| Phase | Epochs | Learning Rate | Frozen Groups | Trainable |
|-------|--------|---------------|---------------|-----------|
{phase_rows}
### 6.1.3 Training History

| Phase | Epochs Run | Train Loss | Train Acc | Val Loss | Val Acc |
|-------|-----------|------------|-----------|----------|---------|
{hist_rows}
### 6.1.4 Evaluation Metrics

| Metric | Value |
|--------|-------|
| Accuracy | {metrics['accuracy']:.4f} |
| F1-score (weighted) | {metrics['f1_score']:.4f} |
| Precision (weighted) | {metrics['precision']:.4f} |
| Recall (weighted) | {metrics['recall']:.4f} |
| AUC-ROC | {metrics['auc_roc']:.4f} |
| Test samples | {metrics['test_samples']:,} |
| Threshold | {metrics['threshold']} |

### 6.1.5 Confusion Matrix

{cm_table}

### 6.1.6 Output Artifacts

| Artifact | Path |
|----------|------|
| Model weights | `data/phase3/{config['output']['model_file']}` |
| Metrics report | `data/phase3/{config['output']['metrics_file']}` |
| Confusion matrix | `data/phase3/{config['output']['confusion_matrix_file']}` |
| Training history | `data/phase3/{config['output']['history_file']}` |

### 6.1.7 Execution Summary

| Property | Value |
|----------|-------|
| Device | {hw_info['device']} |
| TensorFlow | {hw_info['tensorflow']} |
| CUDA | {hw_info['cuda']} |
| Python | {hw_info['python']} |
| Platform | {hw_info['platform']} |
| Duration | {duration_s:.2f}s |
| Git commit | `{_get_git_commit()[:12]}` |
| Config file | `config/phase3_config.yaml` (version-controlled) |
| Random state | {config['random_state']} |

### 6.1.9 Cross-Dataset Validation Reference

Cross-dataset generalization evaluation using CICIoMT2024 is documented
in **Section 6.2** (`report_section_crossdataset.md`). To enable,
set `cross_dataset.enabled: true` in `config/phase3_config.yaml`.

---

**Reviewer signature:** ____________________  **Date:** ____/____/______
"""

    report_path = PROJECT_ROOT / "results" / "phase0_analysis" / "report_section_classification.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        f.write(report)
    logger.info("  Report saved: %s", report_path.name)


def _generate_cross_dataset_report(
    wustl_metrics: Dict[str, Any],
    ciciomt_metrics: Dict[str, Any],
    load_report: Dict[str, Any],
    comparison: Dict[str, Any],
) -> None:
    """Render and save Section 6.2 Cross-Dataset Validation report.

    Args:
        wustl_metrics: WUSTL test set metrics.
        ciciomt_metrics: CICIoMT2024 evaluation metrics.
        load_report: CICIoMTLoader load report dict.
        comparison: Per-metric delta comparison dict.
    """
    logger.info("── Generating cross-dataset report ──")

    report_md = render_cross_dataset_report(
        wustl_metrics=wustl_metrics,
        ciciomt_metrics=ciciomt_metrics,
        load_report=load_report,
        comparison=comparison,
    )

    report_path = PROJECT_ROOT / "results" / "phase0_analysis" / "report_section_crossdataset.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        f.write(report_md)
    logger.info("  Cross-dataset report saved: %s", report_path.name)


# ===================================================================
# Pipeline Orchestrator
# ===================================================================


def run_pipeline() -> None:
    """Run the full Phase 3 classification pipeline."""
    t0 = time.time()

    logger.info("═══════════════════════════════════════════════════")
    logger.info("  Phase 3 Classification Engine")
    logger.info("═══════════════════════════════════════════════════")

    # ── Reproducibility seeds ──
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    seed = config["random_state"]
    np.random.seed(seed)  # noqa: NPY002
    tf.random.set_seed(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    logger.info("  Random state: %d", seed)

    # ── Hardware detection ──
    hw_info = _detect_hardware()

    # ── Verify Phase 2 artifacts (SHA-256) ──
    metadata = _verify_phase2_artifacts(config)

    # ── Load Phase 1 data ──
    X_train, y_train, X_test, y_test = _load_phase1_data(config)

    # ── Reshape (sliding windows) ──
    hp = metadata["hyperparameters"]
    reshaper = DataReshaper(timesteps=hp["timesteps"], stride=hp["stride"])
    X_train_w, y_train_w = reshaper.reshape(X_train, y_train)
    X_test_w, y_test_w = reshaper.reshape(X_test, y_test)

    # ── Rebuild detection model + load weights ──
    detection_model = _rebuild_detection_model(metadata, config)
    detection_params = detection_model.count_params()

    # ── Auto classification head ──
    output_units, activation, loss = _auto_classify_head(y_train_w)

    # ── Build full model ──
    full_model = _build_full_model(
        detection_model=detection_model,
        output_units=output_units,
        activation=activation,
        head_config=config["classification_head"],
    )

    # ── Create output directory ──
    output_dir = PROJECT_ROOT / config["output"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Progressive unfreezing training ──
    histories = _progressive_unfreezing(
        model=full_model,
        config=config,
        X_train=X_train_w,
        y_train=y_train_w,
        loss=loss,
    )

    # ── Evaluate on WUSTL test set (primary) ──
    metrics = _evaluate(
        model=full_model,
        X_test=X_test_w,
        y_test=y_test_w,
        threshold=config["evaluation"]["threshold"],
    )

    # ── Cross-dataset validation (CICIoMT2024) ──
    ciciomt_metrics, load_report, comparison = _run_cross_dataset_validation(
        model=full_model,
        wustl_metrics=metrics,
        config=config,
        reshaper=reshaper,
    )

    duration_s = time.time() - t0

    # ── Export WUSTL artifacts ──
    _export_artifacts(
        model=full_model,
        metrics=metrics,
        histories=histories,
        config=config,
        hw_info=hw_info,
        duration_s=duration_s,
    )

    # ── Export cross-dataset artifacts ──
    if ciciomt_metrics is not None:
        _export_cross_dataset_artifacts(
            ciciomt_metrics=ciciomt_metrics,
            load_report=load_report,
            comparison=comparison,
            config=config,
        )

    # ── Generate classification report ──
    _generate_report(
        model=full_model,
        metrics=metrics,
        histories=histories,
        config=config,
        hw_info=hw_info,
        duration_s=duration_s,
        detection_params=detection_params,
    )

    # ── Generate cross-dataset report ──
    if ciciomt_metrics is not None:
        _generate_cross_dataset_report(
            wustl_metrics=metrics,
            ciciomt_metrics=ciciomt_metrics,
            load_report=load_report,
            comparison=comparison,
        )

    logger.info("═══════════════════════════════════════════════════")
    logger.info("  Phase 3 complete — %.2fs", duration_s)
    logger.info("═══════════════════════════════════════════════════")


# ===================================================================
# Entry Point
# ===================================================================


def main() -> None:
    """Entry point for ``python -m src.phase3_classification_engine.phase3_classification``."""
    run_pipeline()


if __name__ == "__main__":
    main()
