"""Tuning report renderer — generates tuning & ablation markdown.

Pure function with no I/O: the pipeline writes the returned string to disk.
"""

from __future__ import annotations

from typing import Any, Dict, List


def render_tuning_report(
    tuning_results: Dict[str, Any],
    ablation_results: Dict[str, Any],
    importance_results: Dict[str, Any],
    multi_seed_results: Dict[str, Any],
    hw_info: Dict[str, str],
    duration_s: float,
    git_commit: str,
) -> str:
    """Render Phase 2.5 Fine-Tuning & Ablation report as markdown."""
    best_config = tuning_results.get("best_config", {})
    best_metrics = tuning_results.get("best_metrics", {})
    metric = tuning_results.get("metric", "attack_f1")

    # Best config table
    best_config_rows = ""
    for k, v in sorted(best_config.items()):
        best_config_rows += f"| `{k}` | {f'{v:.6g}' if isinstance(v, float) else v} |\n"

    best_metrics_rows = ""
    for k, v in sorted(best_metrics.items()):
        if isinstance(v, float):
            best_metrics_rows += f"| {k} | {v:.4f} |\n"

    # Top-5 trials
    trials = tuning_results.get("trials", [])
    completed = [t for t in trials if t.get("status") == "completed"]
    top5 = sorted(
        completed,
        key=lambda t: t["metrics"].get(metric, 0),
        reverse=tuning_results.get("direction") == "maximize",
    )[:5]

    top5_rows = ""
    for t in top5:
        m = t["metrics"]
        top5_rows += (
            f"| {t['trial_index']} | {m.get('attack_f1', 0):.4f} "
            f"| {m.get('attack_recall', 0):.4f} "
            f"| {m.get('auc_roc', 0):.4f} "
            f"| {m.get('accuracy', 0):.4f} "
            f"| {t.get('duration_seconds', 0):.1f}s |\n"
        )

    # Parameter importance
    importances = importance_results.get("importances", {})
    importance_rows = ""
    for name, score in list(importances.items())[:10]:
        bar = "#" * int(score * 40) if score else ""
        importance_rows += f"| `{name}` | {score:.4f} | {bar} |\n"

    # Multi-seed
    ms_section = ""
    if multi_seed_results.get("enabled"):
        ms_rows = ""
        for c in multi_seed_results.get("configs", []):
            stats = c.get("statistics", {})
            ms_rows += (
                f"| {c['rank']} | {c.get('original_score', 0):.4f} "
                f"| {stats.get('mean', 0):.4f} | {stats.get('std', 0):.4f} "
                f"| {stats.get('n_seeds', 0)} |\n"
            )
        ms_section = f"""### 5.3.5 Multi-Seed Validation

| Rank | Original | Mean | Std | Seeds |
|------|----------|------|-----|-------|
{ms_rows}
"""

    # Ablation
    comparison: List[Dict[str, Any]] = ablation_results.get("comparison", [])
    ablation_rows = ""
    for row in comparison:
        f1 = row.get("f1_score")
        if f1 is None:
            ablation_rows += f"| {row['variant']} | FAILED | — | — | — | — |\n"
        else:
            d_f1 = row.get("delta_f1", 0)
            d_auc = row.get("delta_auc", 0)
            ablation_rows += (
                f"| {row['variant']} | {row.get('attack_f1', f1):.4f} "
                f"| {row.get('auc_roc', 0):.4f} | {row.get('accuracy', 0):.4f} "
                f"| {d_f1:+.4f if d_f1 else '—'} | {d_auc:+.4f if d_auc else '—'} |\n"
            )

    return f"""## 5.3 Hyperparameter Fine-Tuning & Ablation Study

### 5.3.1 Search Configuration

| Property | Value |
|----------|-------|
| Strategy | Bayesian TPE (Optuna) |
| Target metric | {metric} ({tuning_results.get('direction', '—')}) |
| Total trials | {tuning_results.get('total_trials', 0)} |
| Completed | {tuning_results.get('completed_trials', 0)} |
| Failed | {tuning_results.get('failed_trials', 0)} |

### 5.3.2 Best Hyperparameters

| Parameter | Value |
|-----------|-------|
{best_config_rows}
### 5.3.3 Best Trial Metrics

| Metric | Value |
|--------|-------|
{best_metrics_rows}
### 5.3.4 Top-5 Trials

| Trial | Attack F1 | Attack Recall | AUC-ROC | Accuracy | Duration |
|-------|-----------|---------------|---------|----------|----------|
{top5_rows}
{ms_section}### 5.3.6 Parameter Importance ({importance_results.get('method', '—')})

| Parameter | Importance | |
|-----------|-----------|---|
{importance_rows}
### 5.3.7 Ablation Study Results

| Variant | Attack F1 | AUC-ROC | Accuracy | delta F1 | delta AUC |
|---------|-----------|---------|----------|---------|----------|
{ablation_rows}
### 5.3.8 Execution Summary

| Property | Value |
|----------|-------|
| Device | {hw_info.get('device', '—')} |
| TensorFlow | {hw_info.get('tensorflow', '—')} |
| Duration | {duration_s:.2f}s |
| Git commit | `{git_commit[:12]}` |

---

**Reviewer signature:** ____________________  **Date:** ____/____/______
"""
