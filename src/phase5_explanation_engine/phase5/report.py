"""Enhanced report generator for Phase 5 Explanation Engine.

Generates section 8.1 of the final report with SHAP methodology
justification, feature interpretations, and human-centric reasoning.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List

import pandas as pd

from .config import Phase5Config

_TIMESTEPS: int = 20
from dashboard.streaming.feature_aligner import N_FEATURES as _N_FEATURES  # canonical: 24


def render_explanation_report(
    enriched_samples: List[Dict[str, Any]],
    importance_df: pd.DataFrame,
    level_counts: Dict[str, int],
    chart_files: List[str],
    baseline_threshold: float,
    hw_info: Dict[str, str],
    duration_s: float,
    git_commit: str,
    config: Phase5Config,
) -> str:
    """Render enhanced section 8.1 explanation report as Markdown.

    Args:
        enriched_samples: Enriched sample dicts.
        importance_df: Feature importance DataFrame.
        level_counts: Risk level counts.
        chart_files: Generated chart filenames.
        baseline_threshold: Baseline threshold value.
        hw_info: Hardware info dict.
        duration_s: Pipeline duration.
        git_commit: Git commit hash.
        config: Phase5Config instance.

    Returns:
        Enhanced Markdown report string.
    """
    top_k = config.top_features
    bio_cols = set(config.biometric_columns)
    top10 = importance_df.head(top_k)

    feat_rows = ""
    for _, row in top10.iterrows():
        feat_rows += f"| {int(row['rank'])} | {row['feature']} " f"| {row['mean_abs_shap']:.6f} |\n"

    # Feature interpretation
    bio_features = [r["feature"] for _, r in top10.iterrows() if r["feature"] in bio_cols]
    net_features = [r["feature"] for _, r in top10.iterrows() if r["feature"] not in bio_cols]
    bio_count = len(bio_features)
    net_count = len(net_features)

    interpretation = (
        f"Among the top {top_k} features, "
        f"{net_count} are network traffic indicators "
        f"({', '.join(net_features[:3])}{'...' if net_count > 3 else ''}) "
        f"and {bio_count} are biometric signals "
        f"({', '.join(bio_features[:3])}{'...' if bio_count > 3 else ''}). "
    )
    if net_count > bio_count:
        interpretation += (
            "Network features dominate, suggesting that traffic anomalies "
            "are the primary indicators of intrusion attempts."
        )
    elif bio_count > net_count:
        interpretation += (
            "Biometric features dominate, indicating that physiological "
            "signal tampering is the primary attack vector."
        )
    else:
        interpretation += (
            "Both modalities contribute equally, confirming the importance "
            "of cross-modal fusion in IoMT intrusion detection."
        )

    risk_levels = ("LOW", "MEDIUM", "HIGH", "CRITICAL")
    dist_rows = ""
    for level in risk_levels:
        count = level_counts.get(level, 0)
        dist_rows += f"| {level} | {count} |\n"

    example_rows = ""
    seen_levels: set = set()
    for s in enriched_samples:
        if s["risk_level"] not in seen_levels:
            seen_levels.add(s["risk_level"])
            expl = s["explanation"][:120]
            example_rows += f"| {s['risk_level']} | {s['sample_index']} " f"| {expl} |\n"

    wf_list = "\n".join(f"- `{f}`" for f in chart_files if f.startswith("waterfall_"))
    tl_list = "\n".join(f"- `{f}`" for f in chart_files if f.startswith("timeline_"))

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    bg_samples = config.background_samples

    report = f"""\
## 8.1 Explanation Engine — SHAP-Based Feature Attribution

This section presents the Phase 5 Explanation Engine results,
providing interpretable feature attributions for all non-NORMAL
risk samples using SHAP (SHapley Additive exPlanations).

### 8.1.1 Samples Explained

| Risk Level | Count |
|------------|-------|
{dist_rows}\
| **Total** | **{len(enriched_samples)}** |

Baseline threshold: {baseline_threshold:.6f}

### 8.1.2 Global Feature Importance (Top {top_k})

| Rank | Feature | Mean |SHAP| |
|------|---------|--------------|
{feat_rows}
Features are ranked by mean absolute SHAP value across all
explained samples. SHAP values aggregated over {_TIMESTEPS}
timesteps per sliding window.

**Feature Interpretation:**
{interpretation}

### 8.1.3 Explanation Examples

| Level | Sample | Explanation |
|-------|--------|-------------|
{example_rows}
### 8.1.4 Visualizations

**Feature importance bar chart:**
- `charts/feature_importance.png`

**Waterfall charts (CRITICAL/HIGH samples):**
{wf_list}

**Anomaly timeline charts:**
{tl_list}

### 8.1.5 SHAP Methodology

| Parameter | Value |
|-----------|-------|
| Method | GradientExplainer (integrated gradients fallback) |
| Background samples | {bg_samples} (Normal class, training set) |
| Explained samples | {len(enriched_samples)} |
| Input shape | ({_TIMESTEPS}, {_N_FEATURES}) per window |
| Aggregation | mean(abs(SHAP)) over timesteps |
| Feature count | {_N_FEATURES} |

**Method Justification:**
GradientExplainer is selected as the primary attribution method because
it is neural-network-native and efficiently handles deep learning models
with 3D temporal input (batch, timesteps, features). Unlike KernelExplainer,
which treats the model as a black box and requires exponential perturbations
for high-dimensional input, GradientExplainer leverages backpropagation
gradients to compute attributions in a single forward-backward pass per
background sample. The integrated gradients fallback provides theoretical
guarantees (completeness and sensitivity axioms) when the GradientExplainer
encounters compatibility issues with custom layers such as BahdanauAttention.
SHAP values are aggregated via mean(|SHAP|) over the temporal dimension to
produce per-feature importance scores that are interpretable by clinicians
and security analysts.

### 8.1.6 Execution Details

| Parameter | Value |
|-----------|-------|
| Hardware | {hw_info.get('device', 'N/A')} |
| TensorFlow | {hw_info.get('tensorflow', 'N/A')} |
| Duration | {duration_s:.2f}s |
| Git commit | `{git_commit[:12]}` |

### 8.1.7 Human-Centric Justification

Explainability is a critical requirement in healthcare IoMT intrusion
detection systems. SHAP-based feature attributions serve three purposes:

1. **Clinical Decision Support:** When an alert is raised, clinicians need
   to understand *why* the system flagged a particular sample. Waterfall
   charts trace each feature's contribution to the final prediction,
   enabling clinicians to assess whether the alert reflects genuine
   physiological distress or a network-based attack.

2. **Audit Trail Compliance:** Healthcare regulations (HIPAA, FDA
   cybersecurity guidance) require that automated decision systems provide
   transparent reasoning. The per-sample explanations and feature
   importance rankings form an auditable trail from raw sensor data to
   risk classification.

3. **Trust Calibration:** By showing the relative importance of biometric
   vs. network features, the explanation engine helps security analysts
   calibrate their trust in the system. When network features dominate
   (e.g., DIntPkt, TotBytes), the alert likely reflects a network
   intrusion; when biometric features dominate (e.g., SpO2, Heart_rate),
   the alert may indicate device tampering or sensor manipulation.

---

**Generated:** {timestamp}
**Pipeline:** Phase 5 Explanation Engine (SOLID)
**Artifacts:** data/phase5/
"""
    return report
