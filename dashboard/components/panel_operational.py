"""Panel 1 — Operational Status (Primary — Always Visible).

Engine health matrix, real-time threat posture gauge,
startup state indicator, and system state banner.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import plotly.graph_objects as go
import streamlit as st

from dashboard.streaming.window_buffer import SystemState
from dashboard.simulation.statistics import threat_level, weighted_risk_score
from dashboard.utils.metrics import (
    compute_engine_health,
    latency_color,
    state_color,
)


def render_engine_health_matrix(
    monitoring_log: Optional[List[Dict[str, Any]]],
) -> None:
    """Render the engine health matrix as a styled dataframe.

    Args:
        monitoring_log: State transition log from Phase 7.
    """
    if monitoring_log is None:
        st.info("Engine health data not available — artifact not found")
        return

    engines = compute_engine_health(monitoring_log)
    if not engines:
        st.info("No engine health data")
        return

    st.markdown("##### Engine Health Matrix")

    for eng in engines:
        status = eng["status"]
        color = state_color(status)
        lat = eng["latency_p95"]
        lat_color = latency_color(lat)
        ts = eng["last_heartbeat"]
        if isinstance(ts, str) and len(ts) > 19:
            ts = ts[:19].replace("T", " ")

        st.markdown(
            f'<div style="display:flex; align-items:center; gap:12px; '
            f'padding:6px 10px; border-left:3px solid {color}; '
            f'margin-bottom:4px; background:{color}11; border-radius:4px;">'
            f'<span style="color:{color}; font-weight:700; width:70px;">'
            f'{status}</span>'
            f'<span style="flex:1; font-size:0.9em;">{eng["engine"]}</span>'
            f'<span style="color:{lat_color}; font-size:0.85em;">'
            f'{lat:.1f}ms</span>'
            f'<span style="color:#888; font-size:0.75em;">{ts}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )


def render_threat_posture_gauge(
    risk_counts: Dict[str, int],
    previous_score: Optional[float] = None,
) -> None:
    """Render the real-time threat posture gauge.

    Args:
        risk_counts: Current risk level distribution.
        previous_score: Score from previous window for delta display.
    """
    score = weighted_risk_score(risk_counts)
    level, color = threat_level(score)

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        delta={
            "reference": previous_score if previous_score is not None else score,
            "relative": False,
            "valueformat": ".1f",
        },
        title={"text": "Threat Posture", "font": {"size": 16, "color": "#e0e0e0"}},
        number={"font": {"size": 36, "color": color}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#555"},
            "bar": {"color": color},
            "bgcolor": "#1a1a2e",
            "steps": [
                {"range": [0, 20], "color": "rgba(46,204,113,0.13)"},
                {"range": [20, 40], "color": "rgba(243,156,18,0.13)"},
                {"range": [40, 70], "color": "rgba(230,126,34,0.13)"},
                {"range": [70, 100], "color": "rgba(231,76,60,0.13)"},
            ],
            "threshold": {
                "line": {"color": "white", "width": 2},
                "thickness": 0.8,
                "value": score,
            },
        },
    ))

    fig.update_layout(
        height=280,
        margin=dict(t=50, b=10, l=30, r=30),
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#e0e0e0",
    )

    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Level: **{level}** | Score: {score:.1f}/100")


def render_startup_indicator(
    state: str,
    flow_count: int,
    calibration_threshold: int = 100,
) -> None:
    """Render the startup state indicator with progress bar.

    Args:
        state: Current system state.
        flow_count: Total flows ingested.
        calibration_threshold: Flows needed for operational state.
    """
    if state == SystemState.INITIALIZING.value:
        st.warning("System starting up — waiting for flows")
        st.progress(0.0)
        st.caption(f"MAD threshold calibration: {flow_count}/{calibration_threshold} flows")

    elif state == SystemState.CALIBRATING.value:
        progress = min(flow_count / calibration_threshold, 1.0)
        st.info(f"Establishing baseline ({flow_count}/{calibration_threshold})")
        st.progress(progress)
        st.caption("Alert generation suppressed during calibration")

    elif state == SystemState.OPERATIONAL.value:
        st.success("Monitoring active")

    elif state == SystemState.DEGRADED.value:
        st.warning("Engine degraded — see health matrix")

    elif state == SystemState.ALERT.value:
        st.error("Active threat detected")


def render_system_state_banner(state: str) -> None:
    """Render the system state banner with color coding.

    Args:
        state: Current system state string.
    """
    banners = {
        SystemState.INITIALIZING.value: ("#95a5a6", "System starting up"),
        SystemState.CALIBRATING.value: ("#3498db", "Establishing baseline"),
        SystemState.OPERATIONAL.value: ("#2ecc71", "Monitoring active"),
        SystemState.DEGRADED.value: ("#f39c12", "Engine degraded — see health matrix"),
        SystemState.ALERT.value: ("#e74c3c", "Active threat detected"),
    }

    color, text = banners.get(state, ("#95a5a6", state))
    st.markdown(
        f'<div style="background:{color}22; border:1px solid {color}; '
        f'padding:8px 16px; border-radius:6px; text-align:center; '
        f'margin-bottom:12px;">'
        f'<span style="color:{color}; font-weight:700; font-size:1.1em;">'
        f'{text}</span></div>',
        unsafe_allow_html=True,
    )


def render(
    gt: Dict[str, Any],
    buffer_status: Optional[Dict[str, Any]] = None,
) -> None:
    """Render the full Operational Status panel.

    Args:
        gt: Ground truth data from project_ground_truth.json.
        buffer_status: Live buffer status (None for static mode).
    """
    st.header("Operational Status")

    # System state banner
    if buffer_status:
        state = buffer_status.get("state", SystemState.INITIALIZING.value)
        flow_count = buffer_status.get("flow_count", 0)
        render_system_state_banner(state)
    else:
        render_system_state_banner(SystemState.OPERATIONAL.value)

    col_left, col_right = st.columns([1, 1])

    with col_left:
        # Engine health matrix
        monitoring = gt.get("monitoring", {})
        if monitoring.get("status") in ("VERIFIED", "PRESENT_UNVERIFIED"):
            # Load monitoring log directly for detailed engine data
            from dashboard.utils.loader import load_monitoring_log
            log = load_monitoring_log()
            render_engine_health_matrix(log)
        else:
            st.info("Monitoring data not available — artifact not found")

        # Startup indicator (only in streaming mode)
        if buffer_status:
            state = buffer_status.get("state", "")
            if state in (SystemState.INITIALIZING.value,
                         SystemState.CALIBRATING.value):
                render_startup_indicator(
                    state,
                    buffer_status.get("flow_count", 0),
                )

    with col_right:
        # Threat posture gauge
        risk = gt.get("risk_adaptive", {})
        if risk.get("status") in ("VERIFIED", "PRESENT_UNVERIFIED"):
            if buffer_status:
                counts = buffer_status.get("risk_counts", {})
            else:
                dist = risk.get("risk_distribution", {})
                counts = {k: v.get("count", v) if isinstance(v, dict) else v
                          for k, v in dist.items()}
            render_threat_posture_gauge(counts)
        else:
            st.info("Risk data not available — artifact not found")
