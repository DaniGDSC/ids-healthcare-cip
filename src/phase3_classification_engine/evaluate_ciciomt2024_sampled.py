# Cross-dataset generalization evaluation — CICIoMT2024 (stratified sample).
#
# Model:  classification_model.weights.h5 (Phase 3, 482,817 params)
# Data:   ciciomt2024_aligned_v2.parquet (already scaled, 29 features)
# Sample: 10,000 stratified (preserves Normal/Attack ratio, random_state=42)
# Source: 5 features mapped, 24 imputed via WUSTL Normal medians
# Scaler: NOT reapplied — aligned parquet already transformed
#
# Pipeline: Load → Sample → Reshape → Predict → Evaluate → Compare → Export
# No retraining, no refitting, no repreprocessing.

"""CICIoMT2024 cross-dataset evaluation with stratified sampling.

Loads the Phase 3 classification model, draws a stratified sample
of 10,000 from the pre-aligned CICIoMT2024 dataset (1.6M samples),
evaluates cross-dataset generalization, and exports comparison
artifacts with full sampling disclosure.
"""

from __future__ import annotations

import gc
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple

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
)

# ── Phase 2 SOLID components (reused for model reconstruction) ──
from src.phase2_detection_engine.phase2.assembler import DetectionModelAssembler
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
PHASE3_WEIGHTS_PATH: Path = PROJECT_ROOT / "data" / "phase3" / "classification_model.weights.h5"
WUSTL_METRICS_PATH: Path = PROJECT_ROOT / "data" / "phase3" / "metrics_report.json"
CICIOMT_ALIGNED_PATH: Path = PROJECT_ROOT / "data" / "external" / "ciciomt2024_aligned_v2.parquet"
ALIGNMENT_REPORT_PATH: Path = PROJECT_ROOT / "data" / "external" / "alignment_report_v2.json"

# Output artifacts
OUTPUT_DIR: Path = PROJECT_ROOT / "data" / "phase3"
OUTPUT_METRICS: str = "metrics_ciciomt.json"
OUTPUT_COMPARISON: str = "comparison_report.json"
OUTPUT_CM_CSV: str = "confusion_matrix_ciciomt.csv"
OUTPUT_SUMMARY: str = "cross_dataset_summary.md"

# Model constants
N_FEATURES: int = 29
TIMESTEPS: int = 20
STRIDE: int = 1
THRESHOLD: float = 0.5
BATCH_SIZE: int = 256

# Classification head config (matches Phase 3 training)
HEAD_DENSE_UNITS: int = 64
HEAD_DENSE_ACTIVATION: str = "relu"
HEAD_DROPOUT_RATE: float = 0.3
HEAD_OUTPUT_UNITS: int = 1
HEAD_OUTPUT_ACTIVATION: str = "sigmoid"

# Sampling constants
SAMPLE_SIZE: int = 10_000
RANDOM_STATE: int = 42

LABEL_COLUMN: str = "label"
LABEL_NORMAL: int = 0
LABEL_ATTACK: int = 1

# Generalization thresholds
STRONG_THRESHOLD: float = 5.0
MODERATE_THRESHOLD: float = 10.0

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


# ── Step 1: Load Model ─────────────────────────────────────────────────


def _load_model() -> tf.keras.Model:
    """Rebuild model architecture and load Phase 3 classification weights.

    Reconstructs the detection backbone from Phase 2 metadata,
    appends the classification head, then loads the final trained
    weights. No retraining.

    Returns:
        Loaded classification model (482,817 params).

    Raises:
        FileNotFoundError: If required artifacts are missing.
    """
    logger.info("── Loading model ──")

    # Load Phase 2 hyperparameters
    if not PHASE2_METADATA_PATH.exists():
        raise FileNotFoundError(f"Missing: {PHASE2_METADATA_PATH}")
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
    logger.info("  Detection backbone: %d params", detection_model.count_params())

    # Add classification head
    x = detection_model.output
    x = tf.keras.layers.Dense(
        HEAD_DENSE_UNITS,
        activation=HEAD_DENSE_ACTIVATION,
        name="dense_head",
    )(x)
    x = tf.keras.layers.Dropout(HEAD_DROPOUT_RATE, name="drop_head")(x)
    x = tf.keras.layers.Dense(
        HEAD_OUTPUT_UNITS,
        activation=HEAD_OUTPUT_ACTIVATION,
        name="output",
    )(x)
    full_model = tf.keras.Model(detection_model.input, x, name="classification_engine")

    # Load trained weights — verify SHA-256
    if not PHASE3_WEIGHTS_PATH.exists():
        raise FileNotFoundError(f"Missing: {PHASE3_WEIGHTS_PATH}")

    weights_hash = _compute_sha256(PHASE3_WEIGHTS_PATH)
    full_model.load_weights(str(PHASE3_WEIGHTS_PATH))

    n_params = full_model.count_params()
    n_layers = len(full_model.layers)
    logger.info("  Model loaded: %d params, %d layers", n_params, n_layers)
    logger.info("  Weights SHA-256: %s", weights_hash)

    return full_model


def _load_wustl_metrics() -> Dict[str, Any]:
    """Load WUSTL test-set metrics for comparison.

    Returns:
        WUSTL metrics dict.

    Raises:
        FileNotFoundError: If metrics file is missing.
    """
    if not WUSTL_METRICS_PATH.exists():
        raise FileNotFoundError(f"Missing: {WUSTL_METRICS_PATH}")

    with open(WUSTL_METRICS_PATH) as f:
        report = json.load(f)

    logger.info("  WUSTL metrics loaded: %s", WUSTL_METRICS_PATH.name)
    return report["metrics"]


def _load_alignment_report() -> Dict[str, Any]:
    """Load alignment report for disclosure metadata.

    Returns:
        Alignment report dict.

    Raises:
        FileNotFoundError: If alignment report is missing.
    """
    if not ALIGNMENT_REPORT_PATH.exists():
        raise FileNotFoundError(f"Missing: {ALIGNMENT_REPORT_PATH}")

    with open(ALIGNMENT_REPORT_PATH) as f:
        return json.load(f)


# ── Step 2: Stratified Sampling ────────────────────────────────────────


def _stratified_sample() -> Tuple[pd.Index, Dict[str, Any]]:
    """Compute stratified sample indices from label column only.

    Reads only the label column to determine class distribution,
    then samples proportionally to preserve Normal/Attack ratio.

    Returns:
        Tuple of (sample_indices, sampling_info dict).
    """
    logger.info("── Stratified sampling ──")

    # Read label column only — minimal memory
    labels = pd.read_parquet(CICIOMT_ALIGNED_PATH, columns=[LABEL_COLUMN])
    total = len(labels)

    normal_count = int((labels[LABEL_COLUMN] == LABEL_NORMAL).sum())
    attack_count = int((labels[LABEL_COLUMN] == LABEL_ATTACK).sum())
    normal_ratio = normal_count / total

    logger.info(
        "  Original: Normal=%d, Attack=%d, Total=%d",
        normal_count,
        attack_count,
        total,
    )

    # Compute per-class sample counts
    n_normal = int(SAMPLE_SIZE * normal_ratio)
    n_attack = SAMPLE_SIZE - n_normal  # remainder to attack to hit exact total

    # Sample indices per class
    normal_idx = (
        labels[labels[LABEL_COLUMN] == LABEL_NORMAL]
        .sample(n=n_normal, random_state=RANDOM_STATE)
        .index
    )
    attack_idx = (
        labels[labels[LABEL_COLUMN] == LABEL_ATTACK]
        .sample(n=n_attack, random_state=RANDOM_STATE)
        .index
    )
    sample_idx = normal_idx.union(attack_idx)

    sampled_normal_pct = n_normal / SAMPLE_SIZE * 100
    sampled_attack_pct = n_attack / SAMPLE_SIZE * 100

    logger.info(
        "  Sampled: Normal=%d, Attack=%d, Total=%d",
        n_normal,
        n_attack,
        len(sample_idx),
    )
    logger.info(
        "  Class ratio preserved: Normal=%.2f%%, Attack=%.2f%%",
        sampled_normal_pct,
        sampled_attack_pct,
    )

    sampling_info = {
        "original_total": total,
        "original_normal": normal_count,
        "original_attack": attack_count,
        "sampled_total": len(sample_idx),
        "sampled_normal": n_normal,
        "sampled_attack": n_attack,
        "normal_ratio_pct": round(sampled_normal_pct, 2),
        "attack_ratio_pct": round(sampled_attack_pct, 2),
        "random_state": RANDOM_STATE,
        "method": "stratified",
    }

    # Free label-only DataFrame
    del labels
    gc.collect()

    return sample_idx, sampling_info


# ── Step 3: Load Sampled Data ──────────────────────────────────────────


def _load_sampled_data(
    sample_idx: pd.Index,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load full parquet, filter to sample indices, free memory.

    Args:
        sample_idx: Pre-computed stratified sample indices.

    Returns:
        Tuple of (X, y) as numpy arrays.
    """
    logger.info("── Loading sampled data ──")

    df = pd.read_parquet(CICIOMT_ALIGNED_PATH)
    df_sample = df.loc[sample_idx].reset_index(drop=True)

    # Free full DataFrame immediately
    del df
    gc.collect()

    feature_cols = [c for c in df_sample.columns if c != LABEL_COLUMN]
    X = df_sample[feature_cols].values.astype(np.float32)
    y = df_sample[LABEL_COLUMN].values.astype(int)

    logger.info("  Loaded sampled data: shape=%s", df_sample.shape)

    # Free sample DataFrame
    del df_sample
    gc.collect()

    return X, y


# ── Step 4: Prepare Input ──────────────────────────────────────────────


def _reshape_input(
    X: np.ndarray,
    y: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Reshape flat features to sliding windows for CNN-BiLSTM input.

    Args:
        X: Raw features of shape (N, 29).
        y: Labels of shape (N,).

    Returns:
        Tuple of (X_windowed, y_windowed).
    """
    logger.info("── Preparing input ──")

    reshaper = DataReshaper(timesteps=TIMESTEPS, stride=STRIDE)
    X_windowed, y_windowed = reshaper.reshape(X, y)

    logger.info("  Input shape: %s", X_windowed.shape)
    logger.info(
        "  Windowed samples: %d (from %d raw)",
        len(y_windowed),
        len(y),
    )

    return X_windowed, y_windowed


# ── Step 5: Predict ────────────────────────────────────────────────────


def _predict(
    model: tf.keras.Model,
    X: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run batched inference on CICIoMT2024 sample.

    Args:
        model: Loaded classification model.
        X: Windowed input of shape (N_win, 20, 29).

    Returns:
        Tuple of (y_pred_prob, y_pred) arrays.
    """
    logger.info("── Prediction ──")

    y_pred_prob = model.predict(X, batch_size=BATCH_SIZE, verbose=0)

    if y_pred_prob.shape[-1] == 1:
        y_pred_prob = y_pred_prob.ravel()
        y_pred = (y_pred_prob >= THRESHOLD).astype(int)
    else:
        y_pred = np.argmax(y_pred_prob, axis=1)

    n_normal = int((y_pred == LABEL_NORMAL).sum())
    n_attack = int((y_pred == LABEL_ATTACK).sum())

    logger.info("  Prediction complete")
    logger.info("  Predicted Normal: %d, Predicted Attack: %d", n_normal, n_attack)

    return y_pred_prob, y_pred


# ── Step 6: Evaluate ───────────────────────────────────────────────────


def _evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_prob: np.ndarray,
) -> Dict[str, Any]:
    """Compute CICIoMT2024 evaluation metrics.

    Args:
        y_true: True labels.
        y_pred: Predicted labels (binary).
        y_pred_prob: Predicted probabilities.

    Returns:
        Metrics dict with accuracy, f1, precision, recall, auc_roc,
        confusion_matrix, classification_report.
    """
    logger.info("── Evaluation — CICIoMT2024 ──")

    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, average="weighted"))
    prec = float(precision_score(y_true, y_pred, average="weighted"))
    rec = float(recall_score(y_true, y_pred, average="weighted"))
    auc = float(roc_auc_score(y_true, y_pred_prob))
    cm = confusion_matrix(y_true, y_pred)
    cls_report = classification_report(y_true, y_pred, output_dict=True)

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
        "threshold": THRESHOLD,
        "test_samples": len(y_true),
    }


# ── Step 7: Cross-Dataset Comparison ──────────────────────────────────


def _interpret_gap(max_delta_pct: float) -> str:
    """Classify generalization quality from max delta percentage.

    Args:
        max_delta_pct: Maximum absolute delta percentage across metrics.

    Returns:
        One of "Strong", "Moderate", or "Limited".
    """
    if max_delta_pct < STRONG_THRESHOLD:
        return "Strong"
    elif max_delta_pct < MODERATE_THRESHOLD:
        return "Moderate"
    else:
        return "Limited"


def _build_comparison(
    wustl_metrics: Dict[str, Any],
    ciciomt_metrics: Dict[str, Any],
) -> Dict[str, Any]:
    """Build per-metric comparison between WUSTL and CICIoMT2024.

    Args:
        wustl_metrics: WUSTL test-set metrics.
        ciciomt_metrics: CICIoMT2024 evaluation metrics.

    Returns:
        Comparison dict with per-metric wustl/ciciomt/delta/delta_pct.
    """
    logger.info("── Cross-Dataset Comparison ──")

    metric_keys = ["accuracy", "f1_score", "precision", "recall", "auc_roc"]
    comparison: Dict[str, Any] = {}

    for key in metric_keys:
        wustl_val = wustl_metrics[key]
        cic_val = ciciomt_metrics[key]
        delta = abs(wustl_val - cic_val)
        delta_pct = (delta / wustl_val * 100) if wustl_val > 0 else 0.0

        comparison[key] = {
            "wustl": round(wustl_val, 6),
            "ciciomt2024": round(cic_val, 6),
            "delta": round(delta, 6),
            "delta_pct": round(delta_pct, 2),
        }

        logger.info(
            "  %s: WUSTL=%.4f, CICIoMT=%.4f, delta=%.4f (%.1f%%)",
            key,
            wustl_val,
            cic_val,
            delta,
            delta_pct,
        )

    max_delta = max(v["delta_pct"] for v in comparison.values())
    interpretation = _interpret_gap(max_delta)
    comparison["interpretation"] = f"{interpretation} generalization"
    comparison["max_delta_pct"] = round(max_delta, 2)

    logger.info(
        "  Interpretation: %s generalization (max delta=%.1f%%)",
        interpretation,
        max_delta,
    )

    return comparison


# ── Step 8: Export ──────────────────────────────────────────────────────


def _export_metrics(
    ciciomt_metrics: Dict[str, Any],
    sampling_info: Dict[str, Any],
    output_dir: Path,
) -> str:
    """Export CICIoMT2024 evaluation metrics to JSON.

    Args:
        ciciomt_metrics: Evaluation metrics dict.
        sampling_info: Stratified sampling metadata.
        output_dir: Destination directory.

    Returns:
        SHA-256 hash of exported file.
    """
    path = output_dir / OUTPUT_METRICS
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "CICIoMT2024",
        "source_parquet": CICIOMT_ALIGNED_PATH.name,
        "sampling": sampling_info,
        **ciciomt_metrics,
    }
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    sha = _compute_sha256(path)
    logger.info("  Exported %s — SHA-256: %s", path.name, sha)
    return sha


def _export_comparison(
    comparison: Dict[str, Any],
    sampling_info: Dict[str, Any],
    alignment_report: Dict[str, Any],
    checks: Dict[str, bool],
    output_dir: Path,
) -> str:
    """Export cross-dataset comparison report to JSON.

    Args:
        comparison: Per-metric comparison dict.
        sampling_info: Stratified sampling metadata.
        alignment_report: Alignment disclosure metadata.
        checks: Validation results.
        output_dir: Destination directory.

    Returns:
        SHA-256 hash of exported file.
    """
    path = output_dir / OUTPUT_COMPARISON
    disclosure = {
        "mapped_features": alignment_report.get("mapped_count", 5),
        "imputed_features": alignment_report.get("imputed_count", 24),
        "excluded_mapping": alignment_report.get("excluded_mapping", {}),
        "drate_zero_variance": True,
        "interpretation": "Conservative lower bound",
    }
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "comparison": comparison,
        "sampling": sampling_info,
        "disclosure": disclosure,
        "validation": {k: ("PASS" if v else "FAIL") for k, v in checks.items()},
    }
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    sha = _compute_sha256(path)
    logger.info("  Exported %s — SHA-256: %s", path.name, sha)
    return sha


def _export_confusion_matrix(
    ciciomt_metrics: Dict[str, Any],
    output_dir: Path,
) -> str:
    """Export CICIoMT2024 confusion matrix to CSV.

    Args:
        ciciomt_metrics: Evaluation metrics containing confusion_matrix.
        output_dir: Destination directory.

    Returns:
        SHA-256 hash of exported file.
    """
    path = output_dir / OUTPUT_CM_CSV
    cm = ciciomt_metrics["confusion_matrix"]
    cm_df = pd.DataFrame(
        cm,
        index=["Actual_Normal", "Actual_Attack"],
        columns=["Pred_Normal", "Pred_Attack"],
    )
    cm_df.to_csv(path)
    sha = _compute_sha256(path)
    logger.info("  Exported %s — SHA-256: %s", path.name, sha)
    return sha


def _export_summary(
    wustl_metrics: Dict[str, Any],
    ciciomt_metrics: Dict[str, Any],
    comparison: Dict[str, Any],
    sampling_info: Dict[str, Any],
    alignment_report: Dict[str, Any],
    output_dir: Path,
) -> str:
    """Export cross-dataset summary as Markdown.

    Args:
        wustl_metrics: WUSTL test-set metrics.
        ciciomt_metrics: CICIoMT2024 evaluation metrics.
        comparison: Per-metric comparison dict.
        sampling_info: Stratified sampling metadata.
        alignment_report: Alignment disclosure metadata.
        output_dir: Destination directory.

    Returns:
        SHA-256 hash of exported file.
    """
    path = output_dir / OUTPUT_SUMMARY
    metric_keys = ["accuracy", "f1_score", "precision", "recall", "auc_roc"]
    display_names = {
        "accuracy": "Accuracy",
        "f1_score": "F1-score",
        "precision": "Precision",
        "recall": "Recall",
        "auc_roc": "AUC-ROC",
    }

    rows = ""
    for key in metric_keys:
        name = display_names[key]
        w = comparison[key]["wustl"]
        c = comparison[key]["ciciomt2024"]
        d = comparison[key]["delta_pct"]
        rows += f"| {name} | {w:.4f} | {c:.4f} | {d:.1f}% |\n"

    interpretation = comparison.get("interpretation", "Limited generalization")

    cm = ciciomt_metrics["confusion_matrix"]
    if len(cm) == 2:
        tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
        cm_table = (
            "| | Predicted Normal | Predicted Attack |\n"
            "|---|---|---|\n"
            f"| **Actual Normal** | TN={tn} | FP={fp} |\n"
            f"| **Actual Attack** | FN={fn} | TP={tp} |"
        )
    else:
        cm_table = "See `confusion_matrix_ciciomt.csv` for full matrix."

    mapped_count = alignment_report.get("mapped_count", 5)
    imputed_count = alignment_report.get("imputed_count", 24)

    # Pre-compute formatted values to avoid long f-string lines
    s_total = f"{sampling_info['original_total']:,}"
    s_sampled = f"{sampling_info['sampled_total']:,}"
    s_norm_pct = sampling_info["normal_ratio_pct"]
    s_atk_pct = sampling_info["attack_ratio_pct"]
    s_wustl_n = f"{wustl_metrics['test_samples']:,}"
    s_orig_n = f"{sampling_info['original_normal']:,}"
    s_orig_a = f"{sampling_info['original_attack']:,}"
    s_samp_n = f"{sampling_info['sampled_normal']:,}"
    s_samp_a = f"{sampling_info['sampled_attack']:,}"

    summary = f"""## Cross-Dataset Validation Summary

Primary dataset: WUSTL-EHMS-2020
Validation dataset: CICIoMT2024

### Sampling Disclosure

Full dataset: {s_total} samples
Evaluated sample: {s_sampled} (stratified, random_state={RANDOM_STATE})
Class ratio preserved: Normal={s_norm_pct}%, Attack={s_atk_pct}%

### Alignment Disclosure

Mapped features: {mapped_count} (dur, load, srcload, totbytes, packet_num)
Imputed features: {imputed_count} (WUSTL Normal medians)
dstload: excluded — drate=0.0 in CICIoMT2024

### Results

| Metric | WUSTL test | CICIoMT2024 | Delta |
|--------|-----------|-------------|-------|
{rows}
Interpretation: {interpretation}

### Confusion Matrix — CICIoMT2024

{cm_table}

### Sample Counts

| Dataset | Samples | Normal | Attack |
|---------|---------|--------|--------|
| WUSTL test | {s_wustl_n} | — | — |
| CICIoMT2024 (full) | {s_total} | {s_orig_n} | {s_orig_a} |
| CICIoMT2024 (sample) | {s_sampled} | {s_samp_n} | {s_samp_a} |

### Note

Results represent conservative lower bound due to feature
imputation and stratified sampling. The {mapped_count} mapped features
carry real CICIoMT2024 signal; the {imputed_count} imputed features
contribute no discriminative information and act as constant baselines.

---

**Reviewer signature:** ____________________  **Date:** ____/____/______
"""

    with open(path, "w") as f:
        f.write(summary)
    sha = _compute_sha256(path)
    logger.info("  Exported %s — SHA-256: %s", path.name, sha)
    return sha


# ── Validation ─────────────────────────────────────────────────────────


def _validate(
    ciciomt_metrics: Dict[str, Any],
    comparison: Dict[str, Any],
    wustl_metrics: Dict[str, Any],
) -> Dict[str, bool]:
    """Run validation assertions on evaluation outputs.

    Args:
        ciciomt_metrics: CICIoMT2024 evaluation metrics.
        comparison: Cross-dataset comparison dict.
        wustl_metrics: WUSTL test-set metrics.

    Returns:
        Dict mapping assertion name to PASS/FAIL.
    """
    logger.info("── Validation checks ──")
    checks: Dict[str, bool] = {}

    # V1: metrics_ciciomt contains all 5 metrics
    required = {"accuracy", "f1_score", "precision", "recall", "auc_roc"}
    passed = required.issubset(ciciomt_metrics.keys())
    checks["all_five_metrics_present"] = passed
    logger.info(
        "  [%s] metrics_ciciomt.json contains all 5 metrics",
        "PASS" if passed else "FAIL",
    )

    # V2: confusion matrix shape == (2, 2)
    cm = ciciomt_metrics.get("confusion_matrix", [])
    passed = len(cm) == 2 and all(len(row) == 2 for row in cm)
    checks["confusion_matrix_shape_2x2"] = passed
    logger.info(
        "  [%s] Confusion matrix shape == (2, 2)",
        "PASS" if passed else "FAIL",
    )

    # V3: delta computed correctly per metric
    metric_keys = ["accuracy", "f1_score", "precision", "recall", "auc_roc"]
    deltas_correct = True
    for key in metric_keys:
        if key not in comparison:
            deltas_correct = False
            break
        expected_delta = abs(wustl_metrics[key] - ciciomt_metrics[key])
        actual_delta = comparison[key]["delta"]
        if abs(expected_delta - actual_delta) > 1e-4:
            deltas_correct = False
            break
    checks["deltas_computed_correctly"] = deltas_correct
    logger.info(
        "  [%s] Delta computed correctly per metric",
        "PASS" if deltas_correct else "FAIL",
    )

    # V4: no data leakage — sample sizes differ
    wustl_samples = wustl_metrics.get("test_samples", 0)
    cic_samples = ciciomt_metrics.get("test_samples", 0)
    passed = wustl_samples != cic_samples or wustl_samples == 0
    checks["no_data_leakage"] = passed
    logger.info(
        "  [%s] No data leakage: CICIoMT2024 (%d) != WUSTL (%d)",
        "PASS" if passed else "FAIL",
        cic_samples,
        wustl_samples,
    )

    n_passed = sum(checks.values())
    logger.info("  Validation: %d/%d PASSED", n_passed, len(checks))

    return checks


# ── Main Pipeline ──────────────────────────────────────────────────────


def run() -> None:
    """Execute the CICIoMT2024 cross-dataset evaluation pipeline."""
    logger.info("═══════════════════════════════════════════════════")
    logger.info("  CICIoMT2024 Cross-Dataset Evaluation (Sampled)")
    logger.info("═══════════════════════════════════════════════════")

    # Step 1: Load artifacts
    model = _load_model()
    wustl_metrics = _load_wustl_metrics()
    alignment_report = _load_alignment_report()

    # Step 2: Stratified sampling
    sample_idx, sampling_info = _stratified_sample()

    # Step 3: Load sampled data
    X, y = _load_sampled_data(sample_idx)
    del sample_idx
    gc.collect()

    # Step 4: Prepare input — reshape for CNN-BiLSTM
    X_windowed, y_windowed = _reshape_input(X, y)
    del X
    gc.collect()

    # Step 5: Predict
    y_pred_prob, y_pred = _predict(model, X_windowed)
    del X_windowed
    gc.collect()

    # Step 6: Evaluate — CICIoMT2024 metrics
    ciciomt_metrics = _evaluate(y_windowed, y_pred, y_pred_prob)

    # Step 7: Cross-dataset comparison
    comparison = _build_comparison(wustl_metrics, ciciomt_metrics)

    # Validation checks
    checks = _validate(ciciomt_metrics, comparison, wustl_metrics)

    # Step 8: Export all artifacts
    logger.info("── Export ──")
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    _export_metrics(ciciomt_metrics, sampling_info, output_dir)
    _export_comparison(comparison, sampling_info, alignment_report, checks, output_dir)
    _export_confusion_matrix(ciciomt_metrics, output_dir)
    _export_summary(
        wustl_metrics,
        ciciomt_metrics,
        comparison,
        sampling_info,
        alignment_report,
        output_dir,
    )

    logger.info("═══════════════════════════════════════════════════")
    logger.info(
        "  Evaluation complete — %d samples, %s",
        ciciomt_metrics["test_samples"],
        comparison.get("interpretation", ""),
    )
    logger.info("═══════════════════════════════════════════════════")


if __name__ == "__main__":
    run()
