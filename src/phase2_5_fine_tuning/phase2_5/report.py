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
    # ── Tuning Results ──
    best_config = tuning_results.get("best_config", {})
    best_metrics = tuning_results.get("best_metrics", {})

    best_config_rows = ""
    for k, v in sorted(best_config.items()):
        if isinstance(v, float):
            best_config_rows += f"| `{k}` | {v:.6g} |\n"
        else:
            best_config_rows += f"| `{k}` | {v} |\n"

    best_metrics_rows = ""
    for k, v in sorted(best_metrics.items()):
        best_metrics_rows += f"| {k} | {v:.4f} |\n"

    # ── Top-5 Trials ──
    trials = tuning_results.get("trials", [])
    metric = tuning_results.get("metric", "f1_score")
    completed = [t for t in trials if t.get("status") == "completed"]
    top5 = sorted(
        completed,
        key=lambda t: t["metrics"].get(metric, 0),
        reverse=tuning_results.get("direction") == "maximize",
    )[:5]

    top5_rows = ""
    for t in top5:
        idx = t["trial_index"]
        m = t["metrics"]
        top5_rows += (
            f"| {idx} | {m.get('f1_score', 0):.4f} "
            f"| {m.get('auc_roc', 0):.4f} "
            f"| {m.get('accuracy', 0):.4f} "
            f"| {t.get('total_params', 0):,} "
            f"| {t.get('duration_seconds', 0):.1f}s |\n"
        )

    # ── Parameter Importance ──
    importances = importance_results.get("importances", {})
    importance_method = importance_results.get("method", "unknown")
    importance_rows = ""
    for name, score in list(importances.items())[:10]:
        bar = "#" * int(score * 40) if score else ""
        importance_rows += f"| `{name}` | {score:.4f} | {bar} |\n"

    # ── Multi-Seed Validation ──
    ms_rows = ""
    ms_configs = multi_seed_results.get("configs", [])
    for c in ms_configs:
        stats = c.get("statistics", {})
        mean = stats.get("mean", 0)
        std = stats.get("std", 0)
        n = stats.get("n_seeds", 0)
        ms_rows += (
            f"| {c['rank']} | {c.get('original_score', 0):.4f} "
            f"| {mean:.4f} | {std:.4f} | {n} |\n"
        )

    # ── Ablation Comparison ──
    comparison: List[Dict[str, Any]] = ablation_results.get("comparison", [])
    ablation_rows = ""
    for row in comparison:
        variant = row["variant"]
        f1 = row.get("f1_score")
        auc = row.get("auc_roc")
        acc = row.get("accuracy")
        params = row.get("total_params")
        d_f1 = row.get("delta_f1")
        d_auc = row.get("delta_auc")

        if f1 is None:
            ablation_rows += f"| {variant} | FAILED | — | — | — | — | — |\n"
        else:
            d_f1_str = f"{d_f1:+.4f}" if d_f1 is not None else "—"
            d_auc_str = f"{d_auc:+.4f}" if d_auc is not None else "—"
            ablation_rows += (
                f"| {variant} | {f1:.4f} | {auc:.4f} "
                f"| {acc:.4f} | {params:,} "
                f"| {d_f1_str} | {d_auc_str} |\n"
            )

    pruned = tuning_results.get("pruned_trials", 0)
    pruned_row = f"| Pruned trials | {pruned} |\n" if pruned else ""

    ms_section = ""
    if multi_seed_results.get("enabled"):
        ms_section = f"""### 5.3.5 Multi-Seed Validation

Top-{multi_seed_results.get('top_k', 3)} configs retrained with {len(multi_seed_results.get('seeds', []))} seeds
({multi_seed_results.get('full_epochs', 10)} epochs each) for confidence intervals.

| Rank | Original {metric} | Mean | Std | Seeds |
|------|------------|------|-----|-------|
{ms_rows}
"""

    report = f"""## 5.3 Hyperparameter Fine-Tuning & Ablation Study

This section documents the Phase 2.5 hyperparameter search and component
ablation study for the CNN-BiLSTM-Attention detection architecture.

### 5.3.1 Search Configuration

| Property | Value |
|----------|-------|
| Strategy | {tuning_results.get('strategy', '—')} |
| Target metric | {metric} ({tuning_results.get('direction', '—')}) |
| Total trials | {tuning_results.get('total_trials', 0)} |
| Completed trials | {tuning_results.get('completed_trials', 0)} |
{pruned_row}| Failed trials | {tuning_results.get('failed_trials', 0)} |

### 5.3.2 Best Hyperparameters

| Parameter | Value |
|-----------|-------|
{best_config_rows}
### 5.3.3 Best Trial Metrics

| Metric | Value |
|--------|-------|
{best_metrics_rows}
### 5.3.4 Top-5 Trials

| Trial | F1 | AUC-ROC | Accuracy | Params | Duration |
|-------|-----|---------|----------|--------|----------|
{top5_rows}
{ms_section}### 5.3.6 Parameter Importance ({importance_method})

| Parameter | Importance | |
|-----------|-----------|---|
{importance_rows}
### 5.3.7 Ablation Study Results

Component ablation reveals the contribution of each architectural block.
Negative delta indicates performance degradation when the component is
removed or modified.

| Variant | F1 | AUC-ROC | Accuracy | Params | delta F1 | delta AUC |
|---------|-----|---------|----------|--------|---------|----------|
{ablation_rows}
### 5.3.8 Execution Summary

| Property | Value |
|----------|-------|
| Device | {hw_info.get('device', '—')} |
| TensorFlow | {hw_info.get('tensorflow', '—')} |
| CUDA | {hw_info.get('cuda', '—')} |
| Python | {hw_info.get('python', '—')} |
| Platform | {hw_info.get('platform', '—')} |
| Duration | {duration_s:.2f}s |
| Git commit | `{git_commit[:12]}` |
| Config file | `config/phase2_5_config.yaml` (version-controlled) |

---

**Reviewer signature:** ____________________  **Date:** ____/____/______
"""
    return report
