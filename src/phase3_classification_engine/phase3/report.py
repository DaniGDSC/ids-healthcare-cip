"""Classification report renderer — generates §6.1 markdown.

Pure function with no I/O: the pipeline writes the returned string
to disk.
"""

from __future__ import annotations

from typing import Any, Dict, List

import tensorflow as tf

from .unfreezer import _LAYER_GROUPS


def render_classification_report(
    model: tf.keras.Model,
    metrics: Dict[str, Any],
    histories: List[Dict[str, Any]],
    config: Any,
    hw_info: Dict[str, str],
    duration_s: float,
    detection_params: int,
    git_commit: str,
) -> str:
    """Render §6.1 Classification Engine report as markdown.

    Args:
        model: Trained classification model.
        metrics: Evaluation metrics dict.
        histories: Per-phase training history list.
        config: Phase3Config instance.
        hw_info: Hardware info dict.
        duration_s: Pipeline execution time in seconds.
        detection_params: Detection model parameter count.
        git_commit: Git commit hash string.

    Returns:
        Complete markdown string.
    """
    head_params = model.count_params() - detection_params
    cm = metrics["confusion_matrix"]

    # §6.1.1 Architecture
    arch_diagram = (
        "```\n"
        "Phase 1 parquets (19980×29, 4896×29)\n"
        "  ↓ reshape (timesteps=20, stride=1)\n"
        "Windows (19961×20×29, 4877×20×29)\n"
        "  ↓ CNN → BiLSTM → Attention (474,496 params, frozen/unfrozen)\n"
        "Context vectors (batch, 128)\n"
        f"  ↓ Dense({config.dense_units}, {config.dense_activation})\n"
        f"  ↓ Dropout({config.head_dropout_rate})\n"
        "  ↓ Dense(1, sigmoid)\n"
        "Predictions (batch, 1)\n"
        "```"
    )

    # §6.1.2 Progressive unfreezing table
    phase_rows = ""
    for phase_cfg in config.training_phases:
        frozen = ", ".join(phase_cfg.frozen)
        trainable_groups = [g for g in _LAYER_GROUPS if g not in phase_cfg.frozen]
        trainable = ", ".join(trainable_groups) if trainable_groups else "—"
        phase_rows += (
            f"| {phase_cfg.name} | {phase_cfg.epochs} "
            f"| {phase_cfg.learning_rate} | {frozen} "
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

    # §6.1.4 Auto classification head table
    auto_head_table = (
        "| n_classes | Activation | Loss | Output units |\n"
        "|-----------|------------|------|--------------|\n"
        "| 2 | sigmoid | binary_crossentropy | 1 |\n"
        "| >2 | softmax | categorical_crossentropy | n |"
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

### 6.1.2 Auto Classification Head

{auto_head_table}

### 6.1.3 Progressive Unfreezing Strategy

Progressive unfreezing chosen to prevent catastrophic forgetting
of Phase 2 feature extraction weights while adapting to
classification task.

| Phase | Epochs | Learning Rate | Frozen Groups | Trainable |
|-------|--------|---------------|---------------|-----------|
{phase_rows}
### 6.1.4 Training History

| Phase | Epochs Run | Train Loss | Train Acc | Val Loss | Val Acc |
|-------|-----------|------------|-----------|----------|---------|
{hist_rows}
### 6.1.5 Evaluation Metrics

| Metric | Value |
|--------|-------|
| Accuracy | {metrics['accuracy']:.4f} |
| F1-score (weighted) | {metrics['f1_score']:.4f} |
| Precision (weighted) | {metrics['precision']:.4f} |
| Recall (weighted) | {metrics['recall']:.4f} |
| AUC-ROC | {metrics['auc_roc']:.4f} |
| Test samples | {metrics['test_samples']:,} |
| Threshold | {metrics['threshold']} |

### 6.1.6 Confusion Matrix

{cm_table}

### 6.1.7 Output Artifacts

| Artifact | Path |
|----------|------|
| Model weights | `data/phase3/{config.model_file}` |
| Metrics report | `data/phase3/{config.metrics_file}` |
| Confusion matrix | `data/phase3/{config.confusion_matrix_file}` |
| Training history | `data/phase3/{config.history_file}` |

### 6.1.8 Execution Summary

| Property | Value |
|----------|-------|
| Device | {hw_info['device']} |
| TensorFlow | {hw_info['tensorflow']} |
| CUDA | {hw_info['cuda']} |
| Python | {hw_info['python']} |
| Platform | {hw_info['platform']} |
| Duration | {duration_s:.2f}s |
| Git commit | `{git_commit[:12]}` |
| Config file | `config/phase3_config.yaml` (version-controlled) |
| Random state | {config.random_state} |

---

**Reviewer signature:** ____________________  **Date:** ____/____/______
"""
    return report
