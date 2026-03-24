"""Panel 2 — Live Alert Feed (Secondary — Continuously Updating).

Displays real-time alert table with risk-level color coding,
top feature attribution, and SHAP drill-down links.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from dashboard.utils.metrics import MAX_ALERT_DISPLAY, risk_color

HIPAA_HASH_PREFIX_LENGTH: int = 8


def _hash_device_id(device_id: str) -> str:
    """Hash a device ID to HIPAA-compliant prefix."""
    return hashlib.sha256(device_id.encode()).hexdigest()[:HIPAA_HASH_PREFIX_LENGTH] + "..."


def _format_top_feature(
    alert: Dict[str, Any],
    feature_names: Optional[List[str]] = None,
) -> str:
    """Format the top contributing feature with deviation direction.

    Args:
        alert: Alert dictionary with SHAP or feature data.
        feature_names: List of feature names for lookup.

    Returns:
        Formatted string like "DIntPkt +47%".
    """
    if "top_features" in alert and alert["top_features"]:
        top = alert["top_features"][0]
        feat = top.get("feature", "")
        pct = top.get("contribution_pct", 0)
        shap_val = top.get("shap_value", 0)
        direction = "+" if shap_val >= 0 else ""
        return f"{feat} {direction}{pct:.0f}%"

    if "top_feature" in alert:
        return str(alert["top_feature"])

    return "—"


def render_alert_table(
    alerts: List[Dict[str, Any]],
    on_explain_click: Optional[str] = None,
) -> Optional[int]:
    """Render the live alert feed as a styled table.

    Args:
        alerts: List of alert dictionaries.
        on_explain_click: Session state key for SHAP explanation trigger.

    Returns:
        Index of selected alert for explanation, or None.
    """
    if not alerts:
        st.info("No alerts to display. "
                "Alerts appear when flows are classified as HIGH or CRITICAL.")
        return None

    display_alerts = alerts[:MAX_ALERT_DISPLAY]
    selected_idx = None

    rows = []
    for i, alert in enumerate(display_alerts):
        level = alert.get("risk_level", "UNKNOWN")
        color = risk_color(level)

        # Timestamp
        ts = alert.get("alert_time", alert.get("timestamp", ""))
        if isinstance(ts, str) and "T" in ts:
            # Extract HH:MM:SS.mmm
            time_part = ts.split("T")[-1] if "T" in ts else ts
            ts_display = time_part[:12]  # HH:MM:SS.mmm
        else:
            ts_display = str(ts)

        # Device ID (hashed for HIPAA)
        device_raw = alert.get("device_id", f"dev_{alert.get('sample_index', i)}")
        device_display = _hash_device_id(str(device_raw))

        rows.append({
            "Timestamp": ts_display,
            "Device": device_display,
            "Risk Level": level,
            "Score": f"{alert.get('anomaly_score', 0):.4f}",
            "Top Feature": _format_top_feature(alert),
            "Action": _get_action(level),
            "_color": color,
            "_idx": i,
        })

    # Render with HTML for color coding
    st.markdown("##### Live Alert Feed")

    for row in rows:
        level = row["Risk Level"]
        color = row["_color"]
        weight = "bold" if level in ("HIGH", "CRITICAL") else "normal"

        col1, col2, col3, col4, col5, col6, col7 = st.columns(
            [1.2, 0.8, 0.9, 0.6, 1.2, 0.8, 0.5],
        )
        with col1:
            st.text(row["Timestamp"])
        with col2:
            st.text(row["Device"])
        with col3:
            st.markdown(
                f'<span style="color:{color}; font-weight:{weight};">'
                f'{level}</span>',
                unsafe_allow_html=True,
            )
        with col4:
            st.text(row["Score"])
        with col5:
            st.text(row["Top Feature"])
        with col6:
            st.text(row["Action"])
        with col7:
            if st.button("Explain", key=f"explain_{row['_idx']}",
                         type="secondary"):
                selected_idx = row["_idx"]

    # Pagination note
    if len(alerts) > MAX_ALERT_DISPLAY:
        remaining = len(alerts) - MAX_ALERT_DISPLAY
        st.caption(f"{remaining} more alerts — expand to view")

    return selected_idx


def _get_action(risk_level: str) -> str:
    """Map risk level to action taken."""
    actions = {
        "NORMAL": "LOGGED",
        "LOW": "LOGGED",
        "MEDIUM": "DASHBOARD",
        "HIGH": "EMAIL SENT",
        "CRITICAL": "ESCALATED",
    }
    return actions.get(risk_level, "LOGGED")


def render_from_ground_truth(gt: Dict[str, Any]) -> Optional[int]:
    """Render alert feed from static ground truth data.

    Args:
        gt: Ground truth data.

    Returns:
        Selected alert index for SHAP explanation.
    """
    st.header("Alert Feed")

    explanation = gt.get("explanation", {})
    if explanation.get("status") not in ("VERIFIED", "PRESENT_UNVERIFIED"):
        st.info("Alert data not available — explanation artifact not found")
        return None

    # Load explanation report for alert data
    from dashboard.utils.loader import load_explanation_report
    report = load_explanation_report()
    if report and "explanations" in report:
        alerts = report["explanations"]
        return render_alert_table(alerts)

    st.info("No alert explanations available")
    return None


def render(
    gt: Dict[str, Any],
    live_alerts: Optional[List[Dict[str, Any]]] = None,
) -> Optional[int]:
    """Render the full Alert Feed panel.

    Args:
        gt: Ground truth data.
        live_alerts: Live streaming alerts (None for static mode).

    Returns:
        Selected alert index for SHAP drill-down.
    """
    st.header("Alert Feed")

    if live_alerts:
        return render_alert_table(live_alerts)

    return render_from_ground_truth(gt)
