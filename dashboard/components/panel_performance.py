"""Panel — Performance Metrics.

Latency histogram, throughput, rolling accuracy, model confidence,
and system health — all computed from live prediction timeseries.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from dashboard.utils.metrics import risk_color


def render(
    predictions: List[Dict[str, Any]],
    sim_status: Dict[str, Any],
    buffer_status: Dict[str, Any],
) -> None:
    """Render performance metrics panel."""
    st.header("Performance Metrics")

    if not predictions:
        st.info("No predictions yet — start a simulation to see performance metrics.")
        return

    latencies = [p["latency"] for p in predictions if p["latency"] > 0]
    scores = [p["score"] for p in predictions]
    ground_truths = [p["ground_truth"] for p in predictions]
    risk_levels = [p["risk"] for p in predictions]

    # ── Summary metrics ──
    c1, c2, c3, c4, c5 = st.columns(5)

    if latencies:
        sorted_lat = sorted(latencies)
        p50 = sorted_lat[len(sorted_lat) // 2]
        p95 = sorted_lat[min(int(len(sorted_lat) * 0.95), len(sorted_lat) - 1)]
        c1.metric("Latency P50", f"{p50:.0f}ms")
        c2.metric("Latency P95", f"{p95:.0f}ms",
                   delta="OK" if p95 < 500 else "SLOW",
                   delta_color="normal" if p95 < 500 else "inverse")
    else:
        c1.metric("Latency P50", "—")
        c2.metric("Latency P95", "—")

    elapsed = 0
    started = buffer_status.get("started_at")
    if started:
        from datetime import datetime, timezone
        try:
            start_dt = datetime.fromisoformat(started)
            elapsed = (datetime.now(timezone.utc) - start_dt).total_seconds()
        except (ValueError, TypeError):
            pass

    throughput = len(predictions) / max(elapsed, 1)
    c3.metric("Throughput", f"{throughput:.1f}/s")

    # Rolling accuracy (last 100 predictions with ground truth)
    recent = [p for p in predictions[-100:] if p["ground_truth"] >= 0]
    if recent:
        correct = sum(
            1 for p in recent
            if (p["risk"] in ("HIGH", "CRITICAL") and p["ground_truth"] == 1) or
               (p["risk"] in ("NORMAL", "LOW") and p["ground_truth"] == 0)
        )
        accuracy = correct / len(recent)
        c4.metric("Accuracy", f"{accuracy:.1%}")

        attacks = [p for p in recent if p["ground_truth"] == 1]
        detected = sum(1 for p in attacks if p["risk"] in ("HIGH", "CRITICAL", "MEDIUM"))
        recall = detected / max(len(attacks), 1)
        c5.metric("Recall", f"{recall:.1%}")
    else:
        c4.metric("Accuracy", "—")
        c5.metric("Recall", "—")

    col_left, col_right = st.columns(2)

    with col_left:
        # ── Latency histogram ──
        if latencies:
            st.markdown("##### Inference Latency Distribution")
            fig = go.Figure(go.Histogram(
                x=latencies,
                nbinsx=30,
                marker_color="#3498db",
            ))
            fig.add_vline(x=500, line_dash="dash", line_color="#e74c3c",
                          annotation_text="SLA (500ms)")
            fig.update_layout(
                height=250, margin=dict(t=10, b=30, l=40, r=20),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e0e0e0", xaxis_title="Latency (ms)", yaxis_title="Count",
            )
            st.plotly_chart(fig, width="stretch")

    with col_right:
        # ── Model confidence distribution ──
        if scores:
            st.markdown("##### Model Confidence Distribution")
            fig = go.Figure(go.Histogram(
                x=scores, nbinsx=30,
                marker_color="#2ecc71",
            ))
            fig.add_vline(x=0.1, line_dash="dash", line_color="#f39c12",
                          annotation_text="Threshold (0.1)")
            fig.update_layout(
                height=250, margin=dict(t=10, b=30, l=40, r=20),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e0e0e0", xaxis_title="Anomaly Score", yaxis_title="Count",
            )
            st.plotly_chart(fig, width="stretch")

    # ── Latency over time ──
    if latencies and len(latencies) > 5:
        st.markdown("##### Latency Over Time")
        fig = go.Figure(go.Scatter(
            x=list(range(len(latencies))), y=latencies,
            mode="lines", line=dict(color="#3498db", width=1),
        ))
        fig.add_hline(y=500, line_dash="dash", line_color="#e74c3c")
        fig.update_layout(
            height=200, margin=dict(t=10, b=30, l=40, r=20),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e0e0e0", xaxis_title="Inference #", yaxis_title="ms",
        )
        st.plotly_chart(fig, width="stretch")

    # ── System health ──
    st.markdown("##### System Health")
    h1, h2, h3, h4 = st.columns(4)
    h1.metric("Buffer", f"{buffer_status.get('buffer_size', 0)} / 5000")
    h2.metric("State", buffer_status.get("state", "?"))
    h3.metric("Inferences", sim_status.get("inferences_run", 0))
    h4.metric("Scenario", sim_status.get("scenario", "—"))

    # ── Cumulative metrics from database ──
    db = st.session_state.get("database")
    if db and db.get_prediction_count() > 0:
        st.markdown("---")
        st.markdown("##### Cumulative Metrics (Database)")
        col_db1, col_db2 = st.columns(2)
        with col_db1:
            st.metric("Total Predictions (DB)", f"{db.get_prediction_count():,}")
            st.metric("Total Alerts (DB)", f"{db.get_alert_count():,}")
        with col_db2:
            risk_dist = db.get_risk_distribution()
            if risk_dist:
                st.markdown("**Risk Distribution (all-time):**")
                for level in ["NORMAL", "LOW", "MEDIUM", "HIGH", "CRITICAL"]:
                    count = risk_dist.get(level, 0)
                    if count:
                        st.caption(f"{level}: {count:,}")
