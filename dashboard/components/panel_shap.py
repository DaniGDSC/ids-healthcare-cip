"""Panel 3 — SHAP Explanation Panel (Analyst View — On-Demand).

Three-tab structure: Waterfall chart, Global feature importance,
and Temporal anomaly timeline. Supports both Phase 5 SHAP data
and Phase 4 conditional explainer (gradient + attention) data.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import plotly.graph_objects as go
import streamlit as st

_BIOMETRIC = {"Temp", "SpO2", "Pulse_Rate", "SYS", "DIA", "Heart_rate", "Resp_Rate", "ST"}


def _feature_color(feature_name: str) -> str:
    """Orange for biometric, blue for network."""
    return "#e67e22" if feature_name in _BIOMETRIC else "#3498db"


def render_waterfall(alert: Dict[str, Any]) -> None:
    """Render feature importance waterfall for a single alert.

    Supports both Phase 5 SHAP format (shap_value key) and
    Phase 4 conditional explainer format (importance key).
    """
    explanation = alert.get("explanation") or {}
    top_features = explanation.get("top_features", [])

    # Fallback to legacy SHAP format (top_features directly on alert)
    if not top_features:
        top_features = alert.get("top_features", [])

    if not top_features:
        st.info("No feature attribution data for this alert")
        return

    names = [f.get("feature", f"f{i}") for i, f in enumerate(top_features)]
    values = [float(f.get("importance", f.get("shap_value", 0)) or 0) for f in top_features]
    colors = ["#e74c3c" if v > 0 else "#3498db" for v in values]

    fig = go.Figure(go.Bar(
        x=values,
        y=names,
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.5f}" for v in values],
        textposition="outside",
    ))

    fig.update_layout(
        title=f"Feature Attribution — Alert #{alert.get('sample_index', '?')} "
              f"({alert.get('risk_level', '')})",
        xaxis_title="Feature Importance",
        height=max(300, len(names) * 30 + 100),
        margin=dict(l=120, r=60, t=50, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e0e0e0",
        yaxis=dict(autorange="reversed"),
    )

    st.plotly_chart(fig, use_container_width=True)

    exp_level = explanation.get("level", "")
    if exp_level == "attention_and_shap":
        st.caption("Source: Gradient-based attribution (conditional explainer)")
    elif exp_level == "attention_only":
        st.caption("Source: Attention weights (lightweight explainer)")
    elif explanation:
        st.caption("Source: SHAP GradientExplainer")


def render_global_importance(gt: Dict[str, Any]) -> None:
    """Render global feature importance bar chart from Phase 5."""
    explanation = gt.get("explanation", {})
    top_10 = explanation.get("top_10_features", [])

    if not top_10:
        st.info("SHAP feature importance data not available")
        return

    names = [f.get("feature_name", f"f{i}") for i, f in enumerate(reversed(top_10))]
    values = [float(f.get("mean_abs_shap", 0)) for f in reversed(top_10)]
    colors = [_feature_color(n) for n in names]

    fig = go.Figure(go.Bar(
        x=values,
        y=names,
        orientation="h",
        marker_color=colors,
        text=[f"{v:.6f}" for v in values],
        textposition="outside",
    ))

    fig.update_layout(
        title="Global Feature Importance (Mean |SHAP|)",
        xaxis_title="Mean Absolute SHAP Value",
        height=400,
        margin=dict(l=120, r=80, t=50, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e0e0e0",
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        '<span style="color:#3498db;">&#9632;</span> Network features &nbsp; '
        '<span style="color:#e67e22;">&#9632;</span> Biometric features',
        unsafe_allow_html=True,
    )

    samples = explanation.get("shap_samples_computed",
                              explanation.get("total_explained", "N/A"))
    time_s = explanation.get("computation_time_s", "N/A")
    st.caption(f"Samples explained: {samples} | Computation time: {time_s}s")


def render_temporal_timeline(
    alert: Dict[str, Any],
    baseline_threshold: float = 0.204,
) -> None:
    """Render temporal attention weight timeline for a single alert.

    Uses real attention weights from conditional explainer when available,
    falls back to anomaly score visualization.
    """
    explanation = alert.get("explanation") or {}
    raw_weights = explanation.get("timestep_importance", [])

    # Sanitise: convert to floats, replace None with 0
    timestep_weights = [float(w) if w is not None else 0.0 for w in raw_weights]

    if timestep_weights:
        n_steps = len(timestep_weights)
        timesteps = list(range(n_steps))
        scores = timestep_weights
        y_title = "Attention Weight"
        title_suffix = "(attention weights)"
    else:
        score = float(alert.get("anomaly_score", 0) or 0)
        n_steps = 20
        timesteps = list(range(n_steps))
        scores = [score] * n_steps
        y_title = "Anomaly Score"
        title_suffix = "(no per-timestep data)"

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=timesteps,
        y=scores,
        mode="lines+markers",
        line=dict(color="#3498db", width=2),
        marker=dict(size=5),
        name=y_title,
    ))

    if not timestep_weights:
        fig.add_hline(
            y=baseline_threshold,
            line_dash="dash",
            line_color="#e74c3c",
            annotation_text=f"MAD Threshold ({baseline_threshold:.3f})",
        )

    fig.update_layout(
        title=f"Temporal Timeline — Alert #{alert.get('sample_index', '?')} {title_suffix}",
        xaxis_title="Timestep",
        yaxis_title=y_title,
        height=350,
        margin=dict(t=50, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e0e0e0",
    )

    st.plotly_chart(fig, use_container_width=True)

    if timestep_weights:
        peak = int(np.argmax(timestep_weights))
        st.caption(f"Peak attention at timestep {peak} (weight={timestep_weights[peak]:.4f})")


def _load_alerts() -> List[Dict[str, Any]]:
    """Load alerts from Phase 4 risk report or Phase 5 explanation report."""
    from dashboard.utils.loader import load_risk_report, load_explanation_report

    risk_report = load_risk_report()
    if risk_report:
        assessments = risk_report.get("sample_assessments", risk_report.get("risk_results", []))
        alerts = [a for a in assessments if a.get("risk_level", "NORMAL") != "NORMAL"]
        if alerts:
            return alerts

    report = load_explanation_report()
    if report and "explanations" in report:
        return report["explanations"]

    return []


def render(
    gt: Dict[str, Any],
    selected_alert_idx: Optional[int] = None,
) -> None:
    """Render the full SHAP Explanation panel."""
    st.header("SHAP Explanations")

    alerts = _load_alerts()

    # Shared alert selector (persists across tabs)
    default_idx = selected_alert_idx if selected_alert_idx is not None else 0

    tab1, tab2, tab3 = st.tabs([
        "Feature Attribution",
        "Global Importance",
        "Temporal Timeline",
    ])

    with tab2:
        render_global_importance(gt)

    if not alerts:
        with tab1:
            st.info("No alerts available for feature attribution analysis")
        with tab3:
            st.info("No temporal data available")
        return

    options = [
        f"#{a.get('sample_index', i)} — {a.get('risk_level', '?')}"
        for i, a in enumerate(alerts)
    ]
    safe_default = min(default_idx, len(options) - 1)
    selected = st.selectbox("Select alert", options, index=safe_default)
    idx = options.index(selected)

    with tab1:
        render_waterfall(alerts[idx])

    with tab3:
        baseline_thresh = gt.get("risk_adaptive", {}).get(
            "baseline_threshold", 0.204,
        )
        render_temporal_timeline(alerts[idx], baseline_thresh)
