# Training diagnosis for Phase 3 Classification Engine.
#
# Analyzes training_history.json, metrics_report.json, and test set
# class distribution to identify root causes of suboptimal performance.
# Model: CNN-BiLSTM-Attention (482,817 params) on WUSTL-EHMS-2020.

"""Phase 3 training diagnosis — identify root causes of suboptimal metrics.

Loads training history and evaluation artifacts, analyzes class
distribution, training convergence, overfitting/underfitting signals,
and per-class performance to produce a diagnosis report.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

# ── Constants ──────────────────────────────────────────────────────────
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

TRAINING_HISTORY_PATH: Path = PROJECT_ROOT / "data" / "phase3" / "training_history.json"
METRICS_PATH: Path = PROJECT_ROOT / "data" / "phase3" / "metrics_report.json"
TEST_PARQUET_PATH: Path = PROJECT_ROOT / "data" / "processed" / "test_phase1.parquet"
OUTPUT_DIR: Path = PROJECT_ROOT / "data" / "phase3"
OUTPUT_REPORT: str = "diagnosis_report.json"

LABEL_COLUMN: str = "Label"
LABEL_NORMAL: int = 0
LABEL_ATTACK: int = 1

# Diagnostic thresholds
OVERFIT_GAP_THRESHOLD: float = 0.1
UNDERFIT_LOSS_THRESHOLD: float = 0.3
LR_OSCILLATION_THRESHOLD: float = 0.05
RANDOM_AUC_THRESHOLD: float = 0.55
CRITICAL_RECALL_THRESHOLD: float = 0.5

# ── Logging ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  [%(name)s]  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Step 1: Load Artifacts ─────────────────────────────────────────────


def _load_artifacts() -> Tuple[List[Dict[str, Any]], Dict[str, Any], pd.DataFrame]:
    """Load training history, metrics report, and test set.

    Returns:
        Tuple of (training_history, metrics, test_df).

    Raises:
        FileNotFoundError: If any artifact is missing.
    """
    logger.info("── Loading artifacts ──")

    if not TRAINING_HISTORY_PATH.exists():
        raise FileNotFoundError(f"Missing: {TRAINING_HISTORY_PATH}")
    with open(TRAINING_HISTORY_PATH) as f:
        history = json.load(f)
    logger.info("  Loaded training_history.json (%d phases)", len(history))

    if not METRICS_PATH.exists():
        raise FileNotFoundError(f"Missing: {METRICS_PATH}")
    with open(METRICS_PATH) as f:
        metrics_report = json.load(f)
    metrics = metrics_report["metrics"]
    logger.info(
        "  Loaded metrics_report.json (accuracy=%.4f)",
        metrics["accuracy"],
    )

    if not TEST_PARQUET_PATH.exists():
        raise FileNotFoundError(f"Missing: {TEST_PARQUET_PATH}")
    test_df = pd.read_parquet(TEST_PARQUET_PATH, columns=[LABEL_COLUMN])
    logger.info("  Loaded test_phase1.parquet (%d samples)", len(test_df))

    return history, metrics, test_df


# ── Step 2: Class Distribution ─────────────────────────────────────────


def _analyze_class_distribution(
    test_df: pd.DataFrame,
    metrics: Dict[str, Any],
) -> Dict[str, Any]:
    """Analyze test set class distribution and compare to naive baseline.

    Args:
        test_df: Test DataFrame with label column.
        metrics: Evaluation metrics dict.

    Returns:
        Class distribution analysis dict.
    """
    logger.info("── Class distribution analysis ──")

    total = len(test_df)
    normal_count = int((test_df[LABEL_COLUMN] == LABEL_NORMAL).sum())
    attack_count = int((test_df[LABEL_COLUMN] == LABEL_ATTACK).sum())
    normal_pct = normal_count / total * 100
    attack_pct = attack_count / total * 100

    logger.info(
        "  Test set: Normal=%d (%.1f%%), Attack=%d (%.1f%%)",
        normal_count,
        normal_pct,
        attack_count,
        attack_pct,
    )

    # Naive baseline: always predict majority class
    naive_accuracy = max(normal_pct, attack_pct) / 100
    actual_accuracy = metrics["accuracy"]
    beats_baseline = actual_accuracy > naive_accuracy
    margin = (actual_accuracy - naive_accuracy) * 100

    if beats_baseline:
        logger.info(
            "  Model beats naive baseline by %.1f%% (%.4f > %.4f)",
            margin,
            actual_accuracy,
            naive_accuracy,
        )
    else:
        logger.warning(
            "  WARNING: Model performs WORSE than naive baseline " "(%.4f < %.4f, margin=%.1f%%)",
            actual_accuracy,
            naive_accuracy,
            margin,
        )

    return {
        "total_samples": total,
        "normal_count": normal_count,
        "attack_count": attack_count,
        "normal_pct": round(normal_pct, 2),
        "attack_pct": round(attack_pct, 2),
        "imbalance_ratio": round(normal_count / attack_count, 2),
        "naive_baseline_accuracy": round(naive_accuracy, 6),
        "actual_accuracy": round(actual_accuracy, 6),
        "beats_baseline": beats_baseline,
        "margin_pct": round(margin, 2),
    }


# ── Step 3: Training History Analysis ──────────────────────────────────


def _flatten_history(
    phases: List[Dict[str, Any]],
) -> Dict[str, List[float]]:
    """Flatten per-phase histories into single epoch-level lists.

    Args:
        phases: List of per-phase training history dicts.

    Returns:
        Dict with keys: train_loss, val_loss, train_acc, val_acc, lr.
    """
    flat: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "lr": [],
    }

    for phase in phases:
        h = phase["history"]
        flat["train_loss"].extend(h["loss"])
        flat["val_loss"].extend(h["val_loss"])
        flat["train_acc"].extend(h["accuracy"])
        flat["val_acc"].extend(h["val_accuracy"])
        flat["lr"].extend(h["learning_rate"])

    return flat


def _analyze_training(
    phases: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Analyze training curves for overfitting, underfitting, convergence.

    Args:
        phases: List of per-phase training history dicts.

    Returns:
        Training analysis dict with diagnostics.
    """
    logger.info("── Training history analysis ──")

    flat = _flatten_history(phases)
    total_epochs = len(flat["train_loss"])

    # a. Overfitting check
    gap = flat["val_loss"][-1] - flat["train_loss"][-1]
    val_increasing = False
    if total_epochs >= 3:
        last_3_val = flat["val_loss"][-3:]
        val_increasing = last_3_val[-1] > last_3_val[0]

    overfit = gap > OVERFIT_GAP_THRESHOLD and val_increasing

    logger.info(
        "  Overfitting: gap=%.4f (threshold=%.1f), " "val_increasing=%s → %s",
        gap,
        OVERFIT_GAP_THRESHOLD,
        val_increasing,
        "OVERFIT" if overfit else "OK",
    )

    # b. Underfitting check
    train_still_decreasing = False
    if total_epochs >= 2:
        train_still_decreasing = flat["train_loss"][-1] < flat["train_loss"][-2]
    final_train_loss = flat["train_loss"][-1]
    underfit = final_train_loss > UNDERFIT_LOSS_THRESHOLD

    logger.info(
        "  Underfitting: train_loss=%.4f (threshold=%.1f), " "still_decreasing=%s → %s",
        final_train_loss,
        UNDERFIT_LOSS_THRESHOLD,
        train_still_decreasing,
        "UNDERFIT" if underfit else "OK",
    )

    # c. Convergence check
    best_epoch = int(np.argmin(flat["val_loss"])) + 1
    best_val_loss = float(min(flat["val_loss"]))
    logger.info(
        "  Best epoch: %d / %d (val_loss=%.4f)",
        best_epoch,
        total_epochs,
        best_val_loss,
    )

    # Check per-phase early stopping
    es_triggered = False
    for phase in phases:
        max_epochs = 5  # configured in phase3_config
        if phase["epochs_run"] < max_epochs:
            es_triggered = True
            logger.info(
                "  EarlyStopping triggered in %s (%d/%d epochs)",
                phase["phase"],
                phase["epochs_run"],
                max_epochs,
            )

    if not es_triggered:
        logger.info("  EarlyStopping: NOT triggered (all phases ran full)")

    # d. Learning rate oscillation
    if len(flat["val_loss"]) >= 5:
        last_5_std = float(np.std(flat["val_loss"][-5:]))
        oscillating = last_5_std > LR_OSCILLATION_THRESHOLD
    else:
        last_5_std = 0.0
        oscillating = False

    logger.info(
        "  LR oscillation: std(last 5 val_loss)=%.4f → %s",
        last_5_std,
        "OSCILLATING" if oscillating else "STABLE",
    )

    # e. Val < Train gap (unusual pattern)
    val_lower_than_train = flat["val_loss"][-1] < flat["train_loss"][-1]
    val_train_gap = flat["train_loss"][-1] - flat["val_loss"][-1]

    if val_lower_than_train:
        logger.info(
            "  UNUSUAL: val_loss (%.4f) < train_loss (%.4f), "
            "gap=%.4f — likely due to Dropout(0.3) active during "
            "training but inactive during validation",
            flat["val_loss"][-1],
            flat["train_loss"][-1],
            val_train_gap,
        )

    # Determine overall behavior
    if overfit:
        behavior = "OVERFIT"
    elif underfit:
        behavior = "UNDERFIT"
    else:
        behavior = "CONVERGED"

    return {
        "total_epochs": total_epochs,
        "best_epoch": best_epoch,
        "best_val_loss": round(best_val_loss, 6),
        "final_train_loss": round(final_train_loss, 6),
        "final_val_loss": round(float(flat["val_loss"][-1]), 6),
        "final_train_acc": round(float(flat["train_acc"][-1]), 6),
        "final_val_acc": round(float(flat["val_acc"][-1]), 6),
        "overfit_gap": round(gap, 6),
        "overfit_detected": overfit,
        "underfit_detected": underfit,
        "train_still_decreasing": train_still_decreasing,
        "early_stopping_triggered": es_triggered,
        "val_lower_than_train": val_lower_than_train,
        "val_train_gap": round(val_train_gap, 6),
        "lr_oscillation_std": round(last_5_std, 6),
        "lr_oscillating": oscillating,
        "training_behavior": behavior,
        "flat_history": flat,
    }


# ── Step 4: Training Curves ───────────────────────────────────────────


def _print_training_curves(flat: Dict[str, List[float]]) -> str:
    """Print ASCII summary of training curves.

    Args:
        flat: Flattened epoch-level history.

    Returns:
        Formatted table string.
    """
    logger.info("── Training curves ──")

    header = (
        f"{'Epoch':>5} | {'Train Loss':>10} | {'Val Loss':>10} "
        f"| {'Train Acc':>9} | {'Val Acc':>9}"
    )
    separator = "-" * len(header)

    lines = [header, separator]

    for i in range(len(flat["train_loss"])):
        line = (
            f"{i + 1:>5} | "
            f"{flat['train_loss'][i]:>10.4f} | "
            f"{flat['val_loss'][i]:>10.4f} | "
            f"{flat['train_acc'][i] * 100:>8.2f}% | "
            f"{flat['val_acc'][i] * 100:>8.2f}%"
        )
        lines.append(line)

    # Best epoch
    best_idx = int(np.argmin(flat["val_loss"]))
    best_line = (
        f"{'best':>5} | "
        f"{flat['train_loss'][best_idx]:>10.4f} | "
        f"{flat['val_loss'][best_idx]:>10.4f} | "
        f"{flat['train_acc'][best_idx] * 100:>8.2f}% | "
        f"{flat['val_acc'][best_idx] * 100:>8.2f}%"
    )
    lines.append(separator)
    lines.append(best_line)

    table = "\n".join(lines)
    logger.info("\n%s", table)

    return table


# ── Step 5: Per-Class Analysis ─────────────────────────────────────────


def _analyze_per_class(
    metrics: Dict[str, Any],
) -> Dict[str, Any]:
    """Analyze per-class performance from classification report.

    Args:
        metrics: Evaluation metrics dict.

    Returns:
        Per-class analysis dict.
    """
    logger.info("── Per-class performance ──")

    cls_report = metrics.get("classification_report", {})
    cm = metrics.get("confusion_matrix", [[0, 0], [0, 0]])

    # Extract per-class metrics
    normal_cls = cls_report.get("0", {})
    attack_cls = cls_report.get("1", {})

    normal_recall = normal_cls.get("recall", 0.0)
    attack_recall = attack_cls.get("recall", 0.0)
    attack_precision = attack_cls.get("precision", 0.0)
    attack_f1 = attack_cls.get("f1-score", 0.0)

    logger.info(
        "  Normal class: precision=%.4f, recall=%.4f, F1=%.4f",
        normal_cls.get("precision", 0.0),
        normal_recall,
        normal_cls.get("f1-score", 0.0),
    )
    logger.info(
        "  Attack class: precision=%.4f, recall=%.4f, F1=%.4f",
        attack_precision,
        attack_recall,
        attack_f1,
    )

    # Confusion matrix interpretation
    if len(cm) == 2:
        tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
        logger.info("  Confusion: TN=%d, FP=%d, FN=%d, TP=%d", tn, fp, fn, tp)
        logger.info(
            "  Attack detection rate: %d/%d = %.1f%%",
            tp,
            tp + fn,
            tp / (tp + fn) * 100 if (tp + fn) > 0 else 0,
        )
        logger.info(
            "  False negative rate: %d/%d = %.1f%%",
            fn,
            tp + fn,
            fn / (tp + fn) * 100 if (tp + fn) > 0 else 0,
        )
    else:
        tn = fp = fn = tp = 0

    # Critical assessment
    attack_recall_critical = attack_recall < CRITICAL_RECALL_THRESHOLD
    near_random_auc = metrics.get("auc_roc", 0.5) < RANDOM_AUC_THRESHOLD

    if attack_recall_critical:
        logger.warning(
            "  CRITICAL: Attack recall (%.4f) below %.0f%% — " "model misses %.0f%% of attacks",
            attack_recall,
            CRITICAL_RECALL_THRESHOLD * 100,
            (1 - attack_recall) * 100,
        )

    if near_random_auc:
        logger.warning(
            "  WARNING: AUC-ROC (%.4f) near random (0.5) — "
            "model has minimal discriminative power",
            metrics.get("auc_roc", 0.5),
        )

    return {
        "normal_precision": round(normal_cls.get("precision", 0.0), 6),
        "normal_recall": round(normal_recall, 6),
        "normal_f1": round(normal_cls.get("f1-score", 0.0), 6),
        "attack_precision": round(attack_precision, 6),
        "attack_recall": round(attack_recall, 6),
        "attack_f1": round(attack_f1, 6),
        "attack_recall_critical": attack_recall_critical,
        "near_random_auc": near_random_auc,
        "auc_roc": round(metrics.get("auc_roc", 0.5), 6),
        "confusion_matrix": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        },
    }


# ── Step 6: Diagnosis Summary ──────────────────────────────────────────


def _build_diagnosis(
    class_dist: Dict[str, Any],
    training: Dict[str, Any],
    per_class: Dict[str, Any],
) -> Dict[str, Any]:
    """Build final diagnosis with root causes and recommendations.

    Args:
        class_dist: Class distribution analysis.
        training: Training history analysis.
        per_class: Per-class performance analysis.

    Returns:
        Complete diagnosis dict.
    """
    logger.info("══════════════════════════════════════════════════")
    logger.info("  DIAGNOSIS REPORT")
    logger.info("══════════════════════════════════════════════════")

    # Identify root causes
    root_causes: List[str] = []

    # 1. Class imbalance
    if class_dist["imbalance_ratio"] > 3.0:
        root_causes.append(
            f"Class imbalance — Normal:Attack ratio = "
            f"{class_dist['imbalance_ratio']}:1, "
            f"model biased toward Normal class"
        )

    # 2. Underfitting
    if training["underfit_detected"]:
        root_causes.append(
            f"Underfitting — train_loss={training['final_train_loss']:.4f} "
            f"still above {UNDERFIT_LOSS_THRESHOLD}, "
            f"model has not converged"
        )

    # 3. Overfitting
    if training["overfit_detected"]:
        root_causes.append(
            f"Overfitting — val_loss > train_loss by " f"{training['overfit_gap']:.4f}"
        )

    # 4. Attack recall critical
    if per_class["attack_recall_critical"]:
        root_causes.append(
            f"Attack class collapse — recall={per_class['attack_recall']:.4f}"
            f", model misses {(1 - per_class['attack_recall']) * 100:.0f}% "
            f"of attacks"
        )

    # 5. Near-random AUC
    if per_class["near_random_auc"]:
        root_causes.append(
            f"Near-random discriminative power — "
            f"AUC-ROC={per_class['auc_roc']:.4f} (random=0.5)"
        )

    # 6. Baseline failure
    if not class_dist["beats_baseline"]:
        root_causes.append(
            f"Below naive baseline — accuracy "
            f"{class_dist['actual_accuracy']:.4f} < "
            f"naive {class_dist['naive_baseline_accuracy']:.4f}"
        )

    # 7. Insufficient epochs
    if training["train_still_decreasing"] and not training["early_stopping_triggered"]:
        root_causes.append(
            "Insufficient training — loss still decreasing at final "
            "epoch, EarlyStopping not triggered"
        )

    # Build recommendations
    recommendations: List[str] = []

    if class_dist["imbalance_ratio"] > 3.0:
        recommendations.append(
            "Apply class weighting: class_weight={0: 1.0, 1: "
            f"{class_dist['imbalance_ratio']:.1f}" + "} in model.fit()"
        )
        recommendations.append("Consider SMOTE or random oversampling of attack class")

    if training["underfit_detected"]:
        recommendations.append("Increase epochs per phase (current: 5, try: 15-20)")
        recommendations.append("Consider higher initial learning rate (Phase A: 0.003)")

    if per_class["attack_recall_critical"]:
        recommendations.append(
            "Lower classification threshold from 0.5 to 0.3 to "
            "increase attack recall at cost of precision"
        )
        recommendations.append(
            "Use focal loss instead of binary_crossentropy to " "down-weight easy Normal examples"
        )

    if not class_dist["beats_baseline"]:
        recommendations.append(
            "Verify Phase 1 data quality — check for label noise, "
            "feature scaling, or data leakage"
        )

    # Print diagnosis
    logger.info(
        "  Naive baseline accuracy: %.2f%%",
        class_dist["naive_baseline_accuracy"] * 100,
    )
    logger.info(
        "  Model accuracy: %.2f%%",
        class_dist["actual_accuracy"] * 100,
    )
    logger.info(
        "  Beats baseline: %s",
        "YES" if class_dist["beats_baseline"] else "NO",
    )
    logger.info("")
    logger.info("  Training behavior: %s", training["training_behavior"])
    logger.info("  Best epoch: %d", training["best_epoch"])
    logger.info(
        "  Early stopping triggered: %s",
        "YES" if training["early_stopping_triggered"] else "NO",
    )
    logger.info("")
    logger.info("  Root causes identified: %d", len(root_causes))
    for i, cause in enumerate(root_causes, 1):
        logger.info("    [%d] %s", i, cause)
    logger.info("")
    logger.info("  Recommended actions:")
    for i, rec in enumerate(recommendations, 1):
        logger.info("    %d. %s", i, rec)
    logger.info("══════════════════════════════════════════════════")

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "naive_baseline": round(class_dist["naive_baseline_accuracy"], 6),
        "model_accuracy": round(class_dist["actual_accuracy"], 6),
        "beats_baseline": class_dist["beats_baseline"],
        "margin_pct": class_dist["margin_pct"],
        "training_behavior": training["training_behavior"],
        "best_epoch": training["best_epoch"],
        "total_epochs": training["total_epochs"],
        "early_stopping_triggered": training["early_stopping_triggered"],
        "class_distribution": {
            "normal": class_dist["normal_count"],
            "attack": class_dist["attack_count"],
            "imbalance_ratio": class_dist["imbalance_ratio"],
        },
        "per_class_metrics": {
            "normal_f1": per_class["normal_f1"],
            "attack_f1": per_class["attack_f1"],
            "attack_recall": per_class["attack_recall"],
            "attack_precision": per_class["attack_precision"],
        },
        "auc_roc": per_class["auc_roc"],
        "root_causes": root_causes,
        "recommended_actions": recommendations,
    }


# ── Export ──────────────────────────────────────────────────────────────


def _export_report(
    diagnosis: Dict[str, Any],
    output_dir: Path,
) -> None:
    """Export diagnosis report to JSON.

    Args:
        diagnosis: Complete diagnosis dict.
        output_dir: Destination directory.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / OUTPUT_REPORT

    with open(path, "w") as f:
        json.dump(diagnosis, f, indent=2)

    logger.info("  Exported %s", path.name)


# ── Main Pipeline ──────────────────────────────────────────────────────


def run() -> None:
    """Execute the training diagnosis pipeline."""
    logger.info("═══════════════════════════════════════════════════")
    logger.info("  Phase 3 Training Diagnosis")
    logger.info("═══════════════════════════════════════════════════")

    # Step 1: Load artifacts
    history, metrics, test_df = _load_artifacts()

    # Step 2: Class distribution analysis
    class_dist = _analyze_class_distribution(test_df, metrics)

    # Step 3: Training history analysis
    training = _analyze_training(history)

    # Step 4: Print training curves
    _print_training_curves(training["flat_history"])

    # Remove flat_history before building final diagnosis
    # (too large for JSON report)
    del training["flat_history"]

    # Step 5: Per-class analysis
    per_class = _analyze_per_class(metrics)

    # Step 6: Build diagnosis
    diagnosis = _build_diagnosis(class_dist, training, per_class)

    # Export
    _export_report(diagnosis, OUTPUT_DIR)

    logger.info("═══════════════════════════════════════════════════")
    logger.info("  Diagnosis complete")
    logger.info("═══════════════════════════════════════════════════")


if __name__ == "__main__":
    run()
