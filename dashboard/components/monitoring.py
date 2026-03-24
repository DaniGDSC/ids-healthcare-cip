"""Page 1 - Live Monitoring: engine health, risk donut, alert feed."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import plotly.graph_objects as go
import streamlit as st

from dashboard.utils.loader import (
    load_explanation_report,
    load_monitoring_log,
    load_risk_report,
)

RISK_COLORS = {
    "NORMAL": "#2ecc71",
    "LOW": "#3498db",
    "MEDIUM": "#f39c12",
    "HIGH": "#e67e22",
    "CRITICAL": "#e74c3c",
}

STATE_COLORS = {
    "UP": "#2ecc71",
    "STARTING": "#3498db",
    "DEGRADED": "#f39c12",
    "DOWN": "#e74c3c",
    "UNKNOWN": "#95a5a6",
}

ENGINE_LABELS = {
    "phase2_detection": "Detection Engine",
    "phase3_classification": "Classification Engine",
    "phase4_risk_adaptive": "Risk-Adaptive Engine",
    "phase5_explanation": "Explanation Engine",
    "phase6_notification": "Notification Engine",
}


def _get_engine_states(log: List[Dict[str, Any]]) -> Dict[str, Dict[str, str]]:
    """Extract latest state per engine from monitoring log."""
    states: Dict[str, Dict[str, str]] = {}
    for entry in log:
        eid = entry["engine_id"]
        states[eid] = {
            "state": entry["new_state"],
            "timestamp": entry["timestamp"],
            "reason": entry.get("reason", ""),
        }
    return states


def render_engine_cards(log: Optional[List[Dict[str, Any]]]) -> None:
    """Render 5 engine status cards with color indicators."""
    if log is None:
        st.warning("Monitoring log not available. Check: data/phase7/monitoring_log.json")
        return

    states = _get_engine_states(log)
    cols = st.columns(5)

    for i, (eid, label) in enumerate(ENGINE_LABELS.items()):
        info = states.get(eid, {"state": "UNKNOWN", "timestamp": "N/A", "reason": ""})
        state = info["state"]
        color = STATE_COLORS.get(state, "#95a5a6")
        ts = info["timestamp"]
        if isinstance(ts, str) and len(ts) > 19:
            ts = ts[:19].replace("T", " ")

        with cols[i]:
            st.markdown(
                f"""<div style="background:{color}22; border-left:4px solid {color};
                padding:12px; border-radius:6px; margin-bottom:8px;">
                <span style="color:{color}; font-weight:bold; font-size:1.1em;">
                {state}</span><br>
                <span style="font-size:0.85em; font-weight:600;">{label}</span><br>
                <span style="font-size:0.7em; color:#999;">{ts}</span>
                </div>""",
                unsafe_allow_html=True,
            )


def render_risk_donut(risk_report: Optional[Dict[str, Any]]) -> None:
    """Render risk distribution donut chart."""
    if risk_report is None:
        st.warning("Risk report not available. Check: data/phase4/risk_report.json")
        return

    dist = risk_report.get("risk_distribution", {})
    if not dist:
        st.info("No risk distribution data.")
        return

    labels = list(dist.keys())
    values = list(dist.values())
    colors = [RISK_COLORS.get(l, "#95a5a6") for l in labels]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.5,
        marker=dict(colors=colors),
        textinfo="label+value",
        textfont_size=13,
        hovertemplate="%{label}: %{value} samples (%{percent})<extra></extra>",
    )])
    fig.update_layout(
        title=dict(text="Risk Level Distribution", font_size=16),
        height=380,
        margin=dict(t=50, b=20, l=20, r=20),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e0e0e0",
    )
    st.plotly_chart(fig, use_container_width=True)


def render_alert_feed(explanation: Optional[Dict[str, Any]]) -> None:
    """Render last 20 alerts as a styled table."""
    if explanation is None:
        st.warning("Explanation report not available. Check: data/phase5/explanation_report.json")
        return

    explanations = explanation.get("explanations", [])
    if not explanations:
        st.info("No alerts to display.")
        return

    alerts = explanations[:20]
    rows = []
    for alert in alerts:
        level = alert.get("risk_level", "UNKNOWN")
        top_feat = ""
        if alert.get("top_features"):
            top_feat = alert["top_features"][0].get("feature", "")
        color = RISK_COLORS.get(level, "#95a5a6")
        rows.append({
            "Timestamp": alert.get("timestamp", "N/A"),
            "Risk Level": level,
            "Score": f"{alert.get('anomaly_score', 0):.4f}",
            "Top Feature": top_feat,
            "_color": color,
        })

    st.markdown("#### Recent Alerts (Last 20)")

    header = "| Timestamp | Risk Level | Score | Top Feature |\n|---|---|---|---|\n"
    body = ""
    for r in rows:
        level_badge = (
            f'<span style="color:{r["_color"]}; font-weight:bold;">'
            f'{r["Risk Level"]}</span>'
        )
        body += f"| {r['Timestamp']} | {level_badge} | {r['Score']} | {r['Top Feature']} |\n"

    # Use a simpler table approach for reliability
    import pandas as pd
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "_color"} for r in rows])
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Risk Level": st.column_config.TextColumn(width="medium"),
            "Score": st.column_config.TextColumn(width="small"),
        },
    )


def render() -> None:
    """Render the full Live Monitoring page."""
    st.header("Live Monitoring")
    st.caption("Engine health status and real-time risk assessment")

    log = load_monitoring_log()
    risk_report = load_risk_report()
    explanation = load_explanation_report()

    render_engine_cards(log)
    st.divider()

    col1, col2 = st.columns([1, 1])
    with col1:
        render_risk_donut(risk_report)
    with col2:
        render_alert_feed(explanation)
