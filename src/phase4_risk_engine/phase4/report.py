"""Risk-adaptive report renderer — generates §7.1 Markdown.

Pure function with no I/O — the caller writes the returned string to disk.
"""

from __future__ import annotations

from typing import Any, Dict, List

from .config import Phase4Config
from .risk_level import RiskLevel


def render_risk_adaptive_report(
    baseline: Dict[str, Any],
    risk_results: List[Dict[str, Any]],
    drift_events: List[Dict[str, Any]],
    window_log: List[Dict[str, Any]],
    config: Phase4Config,
    hw_info: Dict[str, str],
    duration_s: float,
    p3_metrics: Dict[str, Any],
    git_commit: str,
) -> str:
    """Render §7.1 risk-adaptive report as Markdown.

    Args:
        baseline: Baseline config dict.
        risk_results: Per-sample risk assessments.
        drift_events: Concept drift events.
        window_log: Dynamic threshold window entries.
        config: Phase4Config instance.
        hw_info: Hardware info dict.
        duration_s: Pipeline duration in seconds.
        p3_metrics: Phase 3 evaluation metrics.
        git_commit: Current git commit hash.

    Returns:
        Complete Markdown report string.
    """
    # Risk distribution
    level_counts: Dict[str, int] = {}
    for r in risk_results:
        lvl = r["risk_level"]
        level_counts[lvl] = level_counts.get(lvl, 0) + 1

    n_total = len(risk_results)
    risk_rows = ""
    for lvl in RiskLevel:
        count = level_counts.get(lvl.value, 0)
        pct = count / n_total * 100 if n_total > 0 else 0
        risk_rows += f"| {lvl.value} | {count} | {pct:.1f}% |\n"

    # Drift events summary
    drift_rows = ""
    for d in drift_events:
        drift_rows += (
            f"| {d['sample_index']} | {d['drift_ratio']:.4f}"
            f" | {d['action']} | {d['dynamic_threshold']:.6f} |\n"
        )
    if not drift_rows:
        drift_rows = "| — | — | No drift events detected | — |\n"

    # k(t) schedule
    k_rows = ""
    for entry in config.k_schedule:
        k_rows += f"| {entry.start_hour:02d}:00–{entry.end_hour:02d}:00" f" | {entry.k} |\n"

    # Window log sample (first 5)
    win_rows = ""
    for w in window_log[:5]:
        win_rows += (
            f"| {w['sample_index']} | {w['hour']:02d}:00"
            f" | {w['k_t']} | {w['window_median']:.6f}"
            f" | {w['dynamic_threshold']:.6f} |\n"
        )
    if len(window_log) > 5:
        win_rows += "| ... | ... | ... | ... | ... |\n"

    lo = config.low_upper
    med = config.medium_upper
    hi = config.high_upper

    report = f"""## 7.1 Risk-Adaptive Engine

This section documents the Phase 4 Risk-Adaptive Engine, which applies
dynamic thresholding, concept drift detection, and multi-level risk scoring
to the Phase 3 classification output for IoMT healthcare environments.

### 7.1.1 Baseline Computation

Baseline computed from Normal-only training samples using Median + k*MAD:

| Parameter | Value |
|-----------|-------|
| Normal samples (train) | {baseline['n_normal_samples']} |
| Attention dimensions | {baseline['n_attention_dims']} |
| Median | {baseline['median']:.6f} |
| MAD | {baseline['mad']:.6f} |
| k (multiplier) | {baseline['mad_multiplier']} |
| **Baseline threshold** | **{baseline['baseline_threshold']:.6f}** |

Formula: `baseline_threshold = Median + {baseline['mad_multiplier']} * MAD`

### 7.1.2 Dynamic Thresholding

Rolling window Median + k(t)*MAD with time-of-day sensitivity:

| Time Window | k(t) |
|-------------|------|
{k_rows}
**Window size:** {config.window_size} samples

Sample window log:

| Sample | Hour | k(t) | Window Median | Dynamic Threshold |
|--------|------|------|---------------|-------------------|
{win_rows}
### 7.1.3 Concept Drift Detection

| Parameter | Value |
|-----------|-------|
| Drift threshold | {config.drift_threshold} ({config.drift_threshold * 100:.0f}%) |
| Recovery threshold | {config.recovery_threshold} ({config.recovery_threshold * 100:.0f}%) |
| Recovery windows | {config.recovery_windows} consecutive |
| Drift events detected | {len(drift_events)} |

Drift events:

| Sample Index | Drift Ratio | Action | Dynamic Threshold |
|-------------|-------------|--------|-------------------|
{drift_rows}
### 7.1.4 Risk Level Classification

| Risk Level | Count | Percentage |
|------------|-------|------------|
{risk_rows}
**Total samples scored:** {n_total}

Risk thresholds (MAD-relative):

| Level | Condition |
|-------|-----------|
| NORMAL | distance < 0 |
| LOW | 0 <= distance < {lo}*MAD |
| MEDIUM | {lo}*MAD <= distance < {med}*MAD |
| HIGH | {med}*MAD <= distance < {hi}*MAD |
| CRITICAL | distance >= {hi}*MAD AND cross-modal |

### 7.1.5 CRITICAL Risk Protocol

When a CRITICAL risk level is assigned:

1. Immediate on-site alert dispatched
2. Suspicious device isolated from network
3. Escalation chain: IT admin + doctor on duty + manager
4. Medical device is **NOT** shut down (patient safety)
5. Full context logged for human review

### 7.1.6 Phase 3 Model Performance (Inherited)

| Metric | Value |
|--------|-------|
| Accuracy | {p3_metrics.get('accuracy', 0):.4f} |
| F1-score | {p3_metrics.get('f1_score', 0):.4f} |
| AUC-ROC | {p3_metrics.get('auc_roc', 0):.4f} |
| Test samples | {p3_metrics.get('test_samples', 0)} |

### 7.1.7 Execution Summary

| Metric | Value |
|--------|-------|
| Device | {hw_info.get('device', 'N/A')} |
| TensorFlow | {hw_info.get('tensorflow', 'N/A')} |
| Python | {hw_info.get('python', 'N/A')} |
| Platform | {hw_info.get('platform', 'N/A')} |
| Duration | {duration_s:.2f}s |
| Git commit | `{git_commit[:12]}` |

### 7.1.8 Artifacts Exported

| Artifact | Description |
|----------|-------------|
| `baseline_config.json` | Median, MAD, baseline threshold (IMMUTABLE) |
| `threshold_config.json` | Current dynamic threshold, k(t) schedule |
| `risk_report.json` | Per-sample risk levels with distances |
| `drift_log.csv` | Concept drift events and fallback triggers |

---

**Reviewer signature:** ____________________  **Date:** ____/____/______
"""
    return report
