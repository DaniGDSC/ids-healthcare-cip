"""Panel — Live Monitor (primary view).

System state, threat posture gauge, network traffic visualization,
anomaly score timeline, and risk distribution — all live-updating
from the WindowBuffer during streaming simulation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from dashboard.streaming.feature_aligner import MODEL_FEATURES
from dashboard.utils.metrics import risk_color, severity_color


def _threat_score(risk_counts: Dict[str, int]) -> float:
    weights = {"NORMAL": 0, "LOW": 10, "MEDIUM": 30, "HIGH": 60, "CRITICAL": 100}
    total = sum(risk_counts.values())
    if total == 0:
        return 0.0
    return sum(weights.get(k, 0) * v for k, v in risk_counts.items()) / total


def render(
    buffer_status: Dict[str, Any],
    sim_status: Dict[str, Any],
    predictions: List[Dict[str, Any]],
    flow_vectors: List[Any],
) -> None:
    """Render the live monitor panel."""
    st.header("Live Monitor")

    is_running = sim_status.get("running", False)
    flows = buffer_status.get("flow_count", 0)
    state = buffer_status.get("state", "INITIALIZING")

    # ── State banner ──
    colors = {
        "INITIALIZING": ("#95a5a6", "Waiting for data..."),
        "CALIBRATING": ("#3498db", f"Calibrating baseline ({flows} flows)"),
        "OPERATIONAL": ("#2ecc71", "Monitoring active"),
        "DEGRADED": ("#f39c12", "Degraded — check health"),
        "ALERT": ("#e74c3c", "Active threat detected"),
    }
    color, text = colors.get(state, ("#95a5a6", state))
    if is_running:
        phase = sim_status.get("current_phase", "")
        scenario = sim_status.get("scenario", "")
        text = f"Streaming — {scenario}: {phase}" if phase else text

    st.markdown(
        f'<div style="background:{color}22; border:1px solid {color}; '
        f'padding:8px 16px; border-radius:6px; text-align:center; margin-bottom:12px;">'
        f'<span style="color:{color}; font-weight:700;">{text}</span></div>',
        unsafe_allow_html=True,
    )

    # ── Drift warning (⚠️ 4.3.3) ──
    if buffer_status.get("drift_detected"):
        st.warning("Score distribution drift detected — calibrator thresholds may be stale. "
                   "Analyst feedback or recalibration recommended.")

    # ── Metrics row ──
    risk_counts = buffer_status.get("risk_counts", {})
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Flows Processed", f"{flows:,}")
    c2.metric("Inferences", sim_status.get("inferences_run", 0))
    n_alerts = sum(v for k, v in risk_counts.items() if k in ("HIGH", "CRITICAL"))
    c3.metric("Alerts", n_alerts)
    c4.metric("Latency (p50)", f"{sim_status.get('latency_p50_ms', 0):.0f}ms")

    col_left, col_right = st.columns([1, 1])

    with col_left:
        # ── Threat posture gauge ──
        score = _threat_score(risk_counts)
        threat_color = "#2ecc71" if score < 20 else "#f39c12" if score < 50 else "#e74c3c"
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            title={"text": "Threat Posture", "font": {"size": 14, "color": "#e0e0e0"}},
            number={"font": {"size": 28, "color": threat_color}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": threat_color},
                "bgcolor": "#1a1a2e",
                "steps": [
                    {"range": [0, 20], "color": "rgba(46,204,113,0.1)"},
                    {"range": [20, 50], "color": "rgba(243,156,18,0.1)"},
                    {"range": [50, 100], "color": "rgba(231,76,60,0.1)"},
                ],
            },
        ))
        fig.update_layout(height=220, margin=dict(t=40, b=0, l=20, r=20),
                          paper_bgcolor="rgba(0,0,0,0)", font_color="#e0e0e0")
        st.plotly_chart(fig, width="stretch")

    with col_right:
        # ── Risk distribution ──
        if any(risk_counts.values()):
            levels = list(risk_counts.keys())
            counts = [risk_counts[l] for l in levels]
            colors_list = [risk_color(l) for l in levels]
            fig = go.Figure(go.Bar(
                x=counts, y=levels, orientation="h",
                marker_color=colors_list,
                text=[f"{c}" for c in counts], textposition="auto",
            ))
            fig.update_layout(
                title="Risk Distribution", height=220,
                margin=dict(t=40, b=0, l=60, r=20),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e0e0e0",
            )
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("No predictions yet")

    # ── Anomaly score timeline ──
    if predictions:
        st.markdown("##### Anomaly Score Timeline")
        indices = [p["index"] for p in predictions]
        scores = [p["score"] for p in predictions]
        risk_levels = [p["risk"] for p in predictions]
        point_colors = [risk_color(r) for r in risk_levels]
        attn_flags = [p["attention"] for p in predictions]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=indices, y=scores, mode="lines+markers",
            line=dict(color="#3498db", width=1),
            marker=dict(size=4, color=point_colors),
            name="Anomaly Score",
        ))
        # Mark attention anomalies
        attn_idx = [i for i, a in zip(indices, attn_flags) if a]
        attn_scores = [s for s, a in zip(scores, attn_flags) if a]
        if attn_idx:
            fig.add_trace(go.Scatter(
                x=attn_idx, y=attn_scores, mode="markers",
                marker=dict(size=10, symbol="triangle-up", color="#e74c3c"),
                name="Novel Threat",
            ))

        fig.add_hline(y=0.1, line_dash="dash", line_color="#f39c12",
                      annotation_text="Classification threshold (0.1)")
        fig.update_layout(
            height=250, margin=dict(t=10, b=30, l=40, r=20),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e0e0e0", xaxis_title="Prediction #", yaxis_title="Score",
            showlegend=True, legend=dict(orientation="h", y=1.1),
        )
        st.plotly_chart(fig, width="stretch")

    # ── Streaming complete summary ──
    exhausted = sim_status.get("exhausted", False)
    if exhausted and predictions:
        detection_counts = buffer_status.get("detection_counts", {})
        _render_summary(predictions, risk_counts, sim_status, detection_counts)

    # ── Network feature heatmap (last 50 flows) ──
    if flow_vectors and len(flow_vectors) >= 5:
        st.markdown("##### Network Traffic (last 50 flows)")
        data = np.array(flow_vectors[-50:])
        fig = go.Figure(go.Heatmap(
            z=data.T, x=list(range(len(data))),
            y=MODEL_FEATURES,
            colorscale="RdBu_r", zmid=0,
        ))
        fig.update_layout(
            height=350, margin=dict(t=10, b=30, l=100, r=20),
            paper_bgcolor="rgba(0,0,0,0)", font_color="#e0e0e0",
            xaxis_title="Flow #",
        )
        st.plotly_chart(fig, width="stretch")


def _render_summary(
    predictions: List[Dict[str, Any]],
    risk_counts: Dict[str, int],
    sim_status: Dict[str, Any],
    detection_counts: Optional[Dict[str, int]] = None,
) -> None:
    """Render streaming complete summary with accuracy and recall.

    Uses cumulative detection counters from the buffer (tracked over ALL
    flows) rather than recomputing from the truncated prediction deque
    which only holds the last 500 predictions.
    """
    st.markdown("---")
    st.markdown("### Streaming Complete — Final Results")

    if not detection_counts or detection_counts.get("total_with_gt", 0) == 0:
        st.warning("No ground truth available for accuracy computation.")
        return

    tp = detection_counts["tp"]
    fn = detection_counts["fn"]
    fp = detection_counts["fp"]
    tn = detection_counts["tn"]
    total = detection_counts["total_with_gt"]

    accuracy = (tp + tn) / max(total, 1)
    recall = tp / max(tp + fn, 1)
    precision = tp / max(tp + fp, 1)
    fpr = fp / max(fp + tn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)

    n_attacks = tp + fn
    n_benign = fp + tn

    # Metrics
    st.markdown("##### Detection Performance")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy", f"{accuracy:.1%}")
    m2.metric("Recall", f"{recall:.1%}",
              help=f"Detected {tp}/{n_attacks} attacks")
    m3.metric("Precision", f"{precision:.1%}",
              help=f"{fp} false positives out of {tp + fp} alerts")
    m4.metric("F1 Score", f"{f1:.1%}")
    m5.metric("FPR", f"{fpr:.1%}",
              help=f"{fp} false alarms out of {n_benign} benign flows")

    # Confusion matrix + stats
    col_cm, col_stats = st.columns([1, 1])

    with col_cm:
        st.markdown("##### Confusion Matrix")
        fig = go.Figure(go.Heatmap(
            z=[[tn, fp], [fn, tp]],
            x=["Predicted Normal", "Predicted Alert"],
            y=["Actual Benign", "Actual Attack"],
            text=[[f"TN={tn}", f"FP={fp}"], [f"FN={fn}", f"TP={tp}"]],
            texttemplate="%{text}",
            colorscale="RdBu_r",
            showscale=False,
        ))
        fig.update_layout(
            height=250, margin=dict(t=10, b=30, l=100, r=20),
            paper_bgcolor="rgba(0,0,0,0)", font_color="#e0e0e0",
        )
        st.plotly_chart(fig, width="stretch")

    with col_stats:
        st.markdown("##### Streaming Statistics")
        st.markdown(f"- **Total flows:** {sim_status.get('flows_injected', 0):,}")
        st.markdown(f"- **Total inferences:** {sim_status.get('inferences_run', 0):,}")
        st.markdown(f"- **Actual attacks:** {n_attacks} ({n_attacks/max(total,1):.1%})")
        st.markdown(f"- **Actual benign:** {n_benign} ({n_benign/max(total,1):.1%})")

        total_risk = sum(risk_counts.values())
        for level in ["NORMAL", "LOW", "MEDIUM", "HIGH", "CRITICAL"]:
            count = risk_counts.get(level, 0)
            pct = count / max(total_risk, 1) * 100
            st.markdown(f"- **{level}:** {count} ({pct:.1f}%)")

        st.markdown(f"- **Latency (P50):** {sim_status.get('latency_p50_ms', 0):.1f}ms")

        st.markdown("---")
        st.caption(
            "Supplementary anomaly detection layer — designed to complement existing "
            "hospital network security tools (firewall, NAC, SIEM). "
            "Precision 96%: alerts are trustworthy. "
            "Recall improves through analyst feedback over time. "
            "Evaluated on independent holdout (model never saw this data)."
        )
