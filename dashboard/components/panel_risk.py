"""Panel 4 — Risk Distribution Analytics (Management View).

Session statistics, alert response metrics, risk distribution
chart, and simulation mode indicator.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

import plotly.graph_objects as go
import streamlit as st

from dashboard.simulation.statistics import k_schedule_display, threshold_drift_status
from dashboard.utils.metrics import risk_color


def render_session_stats(
    gt: Dict[str, Any],
    buffer_status: Optional[Dict[str, Any]] = None,
) -> None:
    """Render session statistics card.

    Args:
        gt: Ground truth data.
        buffer_status: Live buffer status (None for static mode).
    """
    st.markdown("##### Session Statistics")

    risk = gt.get("risk_adaptive", {})
    perf = gt.get("performance", {})

    if buffer_status:
        flows = buffer_status.get("flow_count", 0)
    else:
        flows = risk.get("total_assessments", 0)

    st.metric("Flows Processed", f"{flows:,}")

    # Detection rate
    attack_recall = perf.get("attack_recall", 0)
    if perf.get("status") in ("VERIFIED", "PRESENT_UNVERIFIED"):
        st.metric("Detection Rate", f"{attack_recall:.2%}")

    # Current threshold with time context
    baseline_thresh = risk.get("baseline_threshold", 0.204)
    now_hour = datetime.now().hour
    k_val, mode_label = k_schedule_display(now_hour)
    st.metric(
        "Current Threshold",
        f"{baseline_thresh:.3f}",
        help=f"k={k_val} | {mode_label}",
    )
    st.caption(f"k={k_val} | {mode_label}")

    # Threshold drift status
    status_label, _ = threshold_drift_status(baseline_thresh, baseline_thresh)
    st.metric("Threshold Status", status_label)

    # Concept drift
    drift_events = risk.get("concept_drift_events", 0)
    st.metric("Concept Drift Events", str(drift_events) if drift_events else "None detected")


def render_alert_response_metrics(gt: Dict[str, Any]) -> None:
    """Render alert response metrics card.

    Args:
        gt: Ground truth data.
    """
    st.markdown("##### Alert Response Metrics")

    notif = gt.get("notification", {})
    perf = gt.get("performance", {})

    if notif.get("status") in ("VERIFIED", "PRESENT_UNVERIFIED"):
        total = notif.get("total_notifications", 0)
        st.metric("Total Notifications", f"{total:,}")

        phi = notif.get("phi_violations", 0)
        st.metric("PHI Violations", str(phi),
                  delta="Compliant" if phi == 0 else "VIOLATION",
                  delta_color="off" if phi == 0 else "inverse")

        tls = notif.get("tls_compliance_rate", 1.0)
        st.metric("TLS Compliance", f"{tls:.0%}")

        escalations = notif.get("escalation_activations", 0)
        st.metric("Escalation Activations", str(escalations))
    else:
        st.info("Notification metrics not available")

    # FPR from performance
    fpr = perf.get("false_positive_rate", 0)
    if perf.get("status") in ("VERIFIED", "PRESENT_UNVERIFIED"):
        st.metric("False Positive Rate (est.)",
                  f"{fpr:.1%}",
                  help="WUSTL baseline estimate")


def render_risk_distribution(
    gt: Dict[str, Any],
    live_counts: Optional[Dict[str, int]] = None,
) -> None:
    """Render risk level distribution as horizontal bar chart.

    Args:
        gt: Ground truth data.
        live_counts: Live risk counts (overrides static data).
    """
    risk = gt.get("risk_adaptive", {})

    if live_counts:
        counts = live_counts
    elif risk.get("status") in ("VERIFIED", "PRESENT_UNVERIFIED"):
        dist = risk.get("risk_distribution", {})
        counts = {}
        for level, info in dist.items():
            if isinstance(info, dict):
                counts[level] = info.get("count", 0)
            else:
                counts[level] = info
    else:
        st.info("Risk distribution not available — artifact not found")
        return

    total = sum(counts.values())
    if total == 0:
        st.info("No risk data yet")
        return

    levels = ["NORMAL", "LOW", "MEDIUM", "HIGH", "CRITICAL"]
    values = [counts.get(l, 0) for l in levels]
    colors = [risk_color(l) for l in levels]
    pcts = [f"{v / total:.1%}" for v in values]
    labels = [f"{l}: {v} ({p})" for l, v, p in zip(levels, values, pcts)]

    fig = go.Figure(go.Bar(
        x=values,
        y=labels,
        orientation="h",
        marker_color=colors,
        text=[str(v) for v in values],
        textposition="outside",
    ))

    fig.update_layout(
        title="Risk Level Distribution",
        xaxis_title="Count",
        height=300,
        margin=dict(l=180, r=60, t=50, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e0e0e0",
        yaxis=dict(autorange="reversed"),
    )

    st.plotly_chart(fig, use_container_width=True)
    st.metric("Total Assessments", f"{total:,}")


def render_simulation_indicator(
    sim_status: Optional[Dict[str, Any]] = None,
) -> None:
    """Render simulation mode indicator if active.

    Args:
        sim_status: Simulator status dict.
    """
    if sim_status and sim_status.get("running"):
        st.info(
            f"Simulation Mode: MedSec-25 stochastic injection\n\n"
            f"**Scenario:** {sim_status.get('scenario', '?')} — "
            f"{sim_status.get('scenario_name', '')}\n\n"
            f"**Mode:** {sim_status.get('mode', 'ACCELERATED')}\n\n"
            f"**Flows injected:** {sim_status.get('flows_injected', 0):,}\n\n"
            f"**Feature overlap:** 12/29 mapped, 17/29 imputed"
        )


def render(
    gt: Dict[str, Any],
    buffer_status: Optional[Dict[str, Any]] = None,
    sim_status: Optional[Dict[str, Any]] = None,
) -> None:
    """Render the full Risk Distribution Analytics panel.

    Args:
        gt: Ground truth data.
        buffer_status: Live buffer status.
        sim_status: Simulator status.
    """
    st.header("Risk Distribution Analytics")

    col_left, col_right = st.columns([1, 1])

    with col_left:
        render_session_stats(gt, buffer_status)
        render_alert_response_metrics(gt)

    with col_right:
        live_counts = buffer_status.get("risk_counts") if buffer_status else None
        render_risk_distribution(gt, live_counts)
        render_simulation_indicator(sim_status)
