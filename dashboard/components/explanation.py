"""Page 3 - SHAP Explanations: feature importance, waterfall, timeline."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from dashboard.utils.loader import (
    load_explanation_report,
    load_feature_names,
    load_shap_values,
)

RISK_COLORS = {
    "NORMAL": "#2ecc71",
    "LOW": "#3498db",
    "MEDIUM": "#f39c12",
    "HIGH": "#e67e22",
    "CRITICAL": "#e74c3c",
}


def render_global_importance(explanation: Dict[str, Any]) -> None:
    """Render top-10 global feature importance bar chart."""
    importance = explanation.get("feature_importance", [])
    if not importance:
        st.info("No feature importance data.")
        return

    top10 = sorted(importance, key=lambda x: x["mean_abs_shap"], reverse=True)[:10]
    features = [f["feature"] for f in reversed(top10)]
    values = [f["mean_abs_shap"] for f in reversed(top10)]

    fig = go.Figure(go.Bar(
        x=values,
        y=features,
        orientation="h",
        marker_color="#3498db",
        text=[f"{v:.6f}" for v in values],
        textposition="outside",
        hovertemplate="%{y}: %{x:.6f}<extra></extra>",
    ))
    fig.update_layout(
        title="Global Feature Importance (mean |SHAP|)",
        xaxis_title="Mean |SHAP Value|",
        height=420,
        margin=dict(t=50, b=30, l=120),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e0e0e0",
        xaxis=dict(gridcolor="#333"),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_waterfall(explanation: Dict[str, Any]) -> None:
    """Render per-alert waterfall chart with dropdown selector."""
    explanations = explanation.get("explanations", [])
    if not explanations:
        st.info("No per-alert explanations available.")
        return

    st.markdown("#### Per-Alert Waterfall")

    options = []
    for ex in explanations:
        idx = ex.get("sample_index", 0)
        level = ex.get("risk_level", "UNKNOWN")
        score = ex.get("anomaly_score", 0)
        options.append(f"Sample {idx} - {level} (score={score:.4f})")

    selected_idx = st.selectbox("Select alert:", range(len(options)), format_func=lambda i: options[i])
    alert = explanations[selected_idx]

    top_features = alert.get("top_features", [])
    if not top_features:
        st.info("No feature contributions for this alert.")
        return

    features = [f["feature"] for f in top_features]
    shap_vals = [f["shap_value"] for f in top_features]
    pcts = [f.get("contribution_pct", 0) for f in top_features]

    colors = ["#e74c3c" if v > 0 else "#3498db" for v in shap_vals]

    fig = go.Figure(go.Bar(
        x=shap_vals,
        y=features,
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.5f} ({p:.1f}%)" for v, p in zip(shap_vals, pcts)],
        textposition="outside",
        hovertemplate="%{y}: %{x:+.6f}<extra></extra>",
    ))

    level = alert.get("risk_level", "UNKNOWN")
    score = alert.get("anomaly_score", 0)
    color = RISK_COLORS.get(level, "#95a5a6")

    fig.update_layout(
        title=f"Feature Contributions - Sample {alert.get('sample_index', '?')} "
              f"[{level}, score={score:.4f}]",
        xaxis_title="SHAP Value (contribution to prediction)",
        height=max(300, len(features) * 35 + 100),
        margin=dict(t=60, b=40, l=100),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e0e0e0",
        xaxis=dict(gridcolor="#333", zeroline=True, zerolinecolor="#555"),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        f"<div style='padding:8px; border-left:4px solid {color}; "
        f"background:{color}22; border-radius:4px;'>"
        f"<b>Risk Level:</b> <span style='color:{color}'>{level}</span> | "
        f"<b>Anomaly Score:</b> {score:.6f} | "
        f"<b>Threshold:</b> {alert.get('threshold', 0):.6f}</div>",
        unsafe_allow_html=True,
    )


def render_timeline(shap_values: Optional[np.ndarray], explanation: Dict[str, Any]) -> None:
    """Render anomaly score timeline across 20 timesteps."""
    if shap_values is None:
        st.info("SHAP values parquet not available for timeline.")
        return

    feature_names = load_feature_names()
    explanations = explanation.get("explanations", [])
    if not explanations:
        return

    st.markdown("#### Temporal Anomaly Timeline")

    # Show first few critical/high alerts
    high_alerts = [e for e in explanations if e.get("risk_level") in ("CRITICAL", "HIGH")]
    if not high_alerts:
        high_alerts = explanations[:3]
    else:
        high_alerts = high_alerts[:3]

    if len(shap_values) == 0:
        st.info("No SHAP values to plot.")
        return

    sample_idx = st.selectbox(
        "Select sample for timeline:",
        range(min(len(shap_values), len(explanations))),
        format_func=lambda i: (
            f"Sample {explanations[i].get('sample_index', i)} - "
            f"{explanations[i].get('risk_level', '?')}"
            if i < len(explanations) else f"Sample {i}"
        ),
        key="timeline_select",
    )

    if sample_idx >= len(shap_values):
        st.warning("Sample index out of SHAP values range.")
        return

    shap_matrix = shap_values[sample_idx]  # (20, 29)
    mean_abs_per_step = np.mean(np.abs(shap_matrix), axis=1)  # (20,)

    # Get threshold for this sample
    threshold_val = 0.0
    if sample_idx < len(explanations):
        alert = explanations[sample_idx]
        threshold_val = alert.get("threshold", 0.204)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(1, 21)),
        y=mean_abs_per_step,
        mode="lines+markers",
        name="Mean |SHAP|",
        line=dict(color="#3498db", width=2),
        marker=dict(size=6),
    ))

    # Find top feature per timestep
    top_feat_indices = np.argmax(np.abs(shap_matrix), axis=1)
    hover_text = []
    for t in range(20):
        feat_idx = top_feat_indices[t]
        feat_name = feature_names[feat_idx] if feat_idx < len(feature_names) else f"f{feat_idx}"
        hover_text.append(f"t={t+1}, top: {feat_name}")
    fig.data[0].hovertext = hover_text

    fig.update_layout(
        title=f"SHAP Magnitude Across 20 Timesteps (Sample {sample_idx})",
        xaxis_title="Timestep",
        yaxis_title="Mean |SHAP Value|",
        height=350,
        margin=dict(t=50, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e0e0e0",
        xaxis=dict(gridcolor="#333", dtick=1),
        yaxis=dict(gridcolor="#333"),
    )
    st.plotly_chart(fig, use_container_width=True)


def render() -> None:
    """Render the full SHAP Explanations page."""
    st.header("SHAP Explanations")
    st.caption("Feature attribution for non-Normal predictions using GradientExplainer")

    explanation = load_explanation_report()
    if explanation is None:
        st.error("Explanation report not available. Check: data/phase5/explanation_report.json")
        return

    total = explanation.get("total_explained", 0)
    level_counts = explanation.get("risk_level_counts", {})
    st.info(
        f"Explained **{total}** non-Normal samples: "
        + ", ".join(f"{k}={v}" for k, v in level_counts.items())
    )

    render_global_importance(explanation)
    st.divider()

    col1, col2 = st.columns([1, 1])
    with col1:
        render_waterfall(explanation)
    with col2:
        shap_values = load_shap_values()
        render_timeline(shap_values, explanation)
