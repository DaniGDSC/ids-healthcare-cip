"""Panel 2 — Live Alert Feed (Secondary — Continuously Updating).

Displays real-time alert table with risk-level color coding,
clinical severity, alert fatigue status, attention anomaly flags,
and explanation drill-down.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional

import streamlit as st

from dashboard.utils.metrics import MAX_ALERT_DISPLAY, risk_color, severity_color, severity_label

HIPAA_HASH_PREFIX_LENGTH: int = 8


def _hash_device_id(device_id: str) -> str:
    """Hash a device ID to HIPAA-compliant prefix."""
    return hashlib.sha256(device_id.encode()).hexdigest()[:HIPAA_HASH_PREFIX_LENGTH] + "..."


def _format_top_feature(alert: Dict[str, Any]) -> str:
    """Format the top contributing feature from explanation data."""
    # New conditional explainer format
    explanation = alert.get("explanation", {})
    top_feats = explanation.get("top_features", [])
    if top_feats:
        top = top_feats[0]
        feat = top.get("feature", "")
        importance = top.get("importance", 0)
        return f"{feat} ({importance:.3f})"

    # Legacy SHAP format
    if "top_features" in alert and alert["top_features"]:
        top = alert["top_features"][0]
        feat = top.get("feature", "")
        pct = top.get("contribution_pct", top.get("importance", 0))
        return f"{feat} ({pct:.3f})"

    return "—"


def _get_action(alert: Dict[str, Any]) -> str:
    """Get the device action from clinical impact assessor."""
    action = alert.get("device_action", "")
    if action == "isolate_network":
        return "ISOLATED"
    if action == "restrict_network":
        return "RESTRICTED"
    risk = alert.get("risk_level", "NORMAL")
    if risk == "CRITICAL":
        return "ESCALATED"
    if risk in ("HIGH", "MEDIUM"):
        return "ALERTED"
    return "LOGGED"


def _generate_suggestion(alert: Dict[str, Any]) -> str:
    """Generate a human-readable suggestion from the alert."""
    risk = alert.get("risk_level", "NORMAL")
    category = alert.get("attack_category", "unknown")
    attention = alert.get("attention_flag", False)
    safety = alert.get("patient_safety_flag", False)

    if risk == "CRITICAL":
        base = f"IMMEDIATE: Isolate device. {category} attack detected."
        if safety:
            base += " Patient safety AT RISK — verify vitals manually."
        return base
    if risk == "HIGH":
        if attention:
            return f"URGENT: Novel threat pattern detected. Restrict network and investigate."
        return f"URGENT: Restrict network access. Anomalous {category} traffic pattern."
    if risk == "MEDIUM":
        if attention:
            return "ADVISORY: Unusual attention pattern — possible zero-day. Monitor closely."
        return f"ADVISORY: Elevated anomaly score in {category} traffic. Monitor."
    return ""


def render_alert_table(
    alerts: List[Dict[str, Any]],
    show_suppressed: bool = False,
) -> Optional[int]:
    """Render the live alert feed as a styled table.

    Args:
        alerts: List of alert dictionaries.
        show_suppressed: Whether to show fatigue-suppressed alerts.

    Returns:
        Index of selected alert for explanation, or None.
    """
    if not alerts:
        st.info("No alerts to display. "
                "Alerts appear when flows are classified as HIGH or CRITICAL.")
        return None

    # Filter by emission status
    if not show_suppressed:
        visible = [a for a in alerts if a.get("alert_emit", True)]
    else:
        visible = list(alerts)

    display_alerts = visible[:MAX_ALERT_DISPLAY]
    selected_idx = None

    # Summary metrics
    total = len(alerts)
    emitted = sum(1 for a in alerts if a.get("alert_emit", True))
    suppressed = total - emitted
    novel = sum(1 for a in alerts if a.get("attention_flag", False))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Alerts", total)
    c2.metric("Emitted", emitted)
    c3.metric("Suppressed", suppressed, help="Reduced by alert fatigue mitigation")
    c4.metric("Novel Threats", novel, help="Flagged by attention anomaly detector")

    st.markdown("##### Alert Feed")

    for i, alert in enumerate(display_alerts):
        level = alert.get("risk_level", "UNKNOWN")
        color = risk_color(level)
        sev = alert.get("clinical_severity", 0)
        sev_lbl = severity_label(sev) if sev else ""
        attn = alert.get("attention_flag", False)
        is_suppressed = not alert.get("alert_emit", True)

        opacity = "0.4" if is_suppressed else "1.0"
        weight = "bold" if level in ("HIGH", "CRITICAL") else "normal"

        # Device ID (hashed for HIPAA)
        device_raw = alert.get("device_id", f"dev_{alert.get('sample_index', i)}")
        device_display = _hash_device_id(str(device_raw))

        col1, col2, col3, col4, col5, col6, col7 = st.columns(
            [0.6, 0.9, 0.7, 1.0, 0.6, 0.7, 0.5],
        )
        with col1:
            st.text(device_display)
        with col2:
            st.markdown(
                f'<span style="color:{color}; font-weight:{weight}; opacity:{opacity};">'
                f'{level}</span>',
                unsafe_allow_html=True,
            )
        with col3:
            if sev_lbl:
                sev_col = severity_color(sev)
                st.markdown(
                    f'<span style="color:{sev_col};">{sev_lbl}</span>',
                    unsafe_allow_html=True,
                )
        with col4:
            feat_text = _format_top_feature(alert)
            if attn:
                st.markdown(f"**NOVEL** | {feat_text}")
            else:
                st.text(feat_text)
        with col5:
            st.text(_get_action(alert))
        with col6:
            if is_suppressed:
                st.caption("suppressed")
            else:
                resp = alert.get("response_time_minutes", 0)
                if resp:
                    st.caption(f"{resp}min")
        with col7:
            if st.button("Detail", key=f"explain_{i}", type="secondary"):
                st.session_state["_alert_detail_idx"] = i

    if len(visible) > MAX_ALERT_DISPLAY:
        remaining = len(visible) - MAX_ALERT_DISPLAY
        st.caption(f"{remaining} more alerts")

    # Pop-up dialog when Detail is clicked
    detail_idx = st.session_state.get("_alert_detail_idx")
    if detail_idx is not None and detail_idx < len(display_alerts):
        _render_dialog(display_alerts[detail_idx], detail_idx)
        selected_idx = detail_idx

    return selected_idx


@st.dialog("Alert Detail", width="large")
def _render_dialog(alert: Dict[str, Any], idx: int) -> None:
    """Render alert detail as a pop-up dialog."""
    sample = alert.get("sample_index", idx)
    level = alert.get("risk_level", "N/A")
    color = risk_color(level)
    sev = alert.get("clinical_severity", 0)

    # Header
    st.markdown(
        f'Sample **{sample}** &mdash; '
        f'<span style="color:{color}; font-weight:bold;">{level}</span>'
        f'{f" / {severity_label(sev)}" if sev else ""}',
        unsafe_allow_html=True,
    )

    # Suggestion
    suggestion = _generate_suggestion(alert)
    if suggestion:
        st.warning(f"**Suggestion:** {suggestion}")

    # Ground truth (demo validation)
    gt = alert.get("ground_truth", -1)
    if gt >= 0:
        gt_label = "ATTACK" if gt == 1 else "BENIGN"
        detected = level in ("HIGH", "CRITICAL", "MEDIUM")
        correct = (gt == 1 and detected) or (gt == 0 and not detected)
        st.caption(f"Ground truth: {gt_label} | Detection: {'CORRECT' if correct else 'MISSED'}")

    if alert.get("attention_flag"):
        st.warning("Potential novel / zero-day threat (attention anomaly)")

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("**Risk Assessment**")
        st.markdown(f"- Anomaly Score: `{alert.get('anomaly_score', 0):.4f}`")
        st.markdown(f"- Threshold: `{alert.get('threshold', 0):.4f}`")
        st.markdown(f"- Distance: `{alert.get('distance', 0):.4f}`")

        action = alert.get("device_action", "none")
        if action != "none":
            st.error(f"Device action: **{action}**")

        resp = alert.get("response_time_minutes", 0)
        if resp:
            st.info(f"Response time: **{resp} minutes**")

    with col_right:
        cia = alert.get("cia_scores", {})
        if cia:
            st.markdown("**CIA Impact**")
            for dim, score in cia.items():
                label = {"C": "Confidentiality", "I": "Integrity", "A": "Availability"}.get(dim, dim)
                st.progress(min(float(score), 1.0), text=f"{label}: {score:.3f}")

    # Explanation
    explanation = alert.get("explanation", {})
    exp_level = explanation.get("level", "none")
    if exp_level != "none":
        st.markdown("**Explanation**")
        top_feats = explanation.get("top_features", [])
        if top_feats:
            for feat in top_feats[:5]:
                st.markdown(f"- **{feat.get('feature', '?')}**: {feat.get('importance', 0):.4f}")

        timesteps = explanation.get("timestep_importance", [])
        if timesteps:
            st.markdown("Temporal attention weights:")
            st.bar_chart(timesteps)

    rationale = alert.get("clinical_rationale", "")
    if rationale:
        st.caption(f"Rationale: {rationale}")

    # Close button clears the selection
    if st.button("Close", use_container_width=True):
        del st.session_state["_alert_detail_idx"]
        st.rerun()


def render_from_ground_truth(gt: Dict[str, Any], show_suppressed: bool = False) -> Optional[int]:  # noqa: ARG001
    """Render alert feed from static ground truth data.

    Args:
        gt: Ground truth data (unused; kept for panel API consistency).
        show_suppressed: Whether to show fatigue-suppressed alerts.

    Returns:
        Selected alert index for SHAP explanation.
    """
    # Try Phase 4 risk report first (has clinical severity + fatigue data)
    from dashboard.utils.loader import load_risk_report
    risk_report = load_risk_report()
    if risk_report:
        assessments = risk_report.get("sample_assessments", risk_report.get("risk_results", []))
        # Filter to non-NORMAL only for alert feed
        alerts = [a for a in assessments if a.get("risk_level", "NORMAL") != "NORMAL"]
        if alerts:
            return render_alert_table(alerts, show_suppressed=show_suppressed)

    # Fallback to Phase 5 explanation report
    from dashboard.utils.loader import load_explanation_report
    report = load_explanation_report()
    if report and "explanations" in report:
        return render_alert_table(report["explanations"], show_suppressed=show_suppressed)

    st.info("No alert data available — run Phase 4 pipeline to generate risk assessments")
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

    show_suppressed = st.checkbox("Show suppressed alerts", value=False,
                                  help="Include alerts suppressed by fatigue mitigation")

    if live_alerts:
        return render_alert_table(live_alerts, show_suppressed=show_suppressed)

    return render_from_ground_truth(gt, show_suppressed=show_suppressed)
