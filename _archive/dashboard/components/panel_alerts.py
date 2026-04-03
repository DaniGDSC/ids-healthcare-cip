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
    if not isinstance(explanation, dict):
        explanation = {}
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
    """Generate clinician-readable suggestion from the alert.

    Uses clinical language, not technical terms. Focuses on patient
    impact and required actions, not network forensics.
    """
    risk = alert.get("risk_level", "NORMAL")
    safety = alert.get("patient_safety_flag", False)
    action = alert.get("device_action", "none")

    if risk == "CRITICAL":
        base = "IMMEDIATE: Device communication compromised."
        if safety:
            base += " Verify patient vitals manually. Do NOT rely on device readings."
        if action == "isolate_network":
            base += " Device has been isolated from network."
        return base
    if risk == "HIGH":
        if action == "restrict_network":
            return ("URGENT: Suspicious device activity detected. "
                    "Network access restricted. Monitor device output.")
        return ("URGENT: Abnormal device behavior. "
                "IT security investigating. Continue manual monitoring.")
    if risk == "MEDIUM":
        return ("ADVISORY: Unusual network activity detected. "
                "No action required. IT security monitoring.")
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
            # Active learning: show uncertainty badge
            from src.production.feedback_loop import FeedbackLoop
            unc = FeedbackLoop.compute_uncertainty(alert)
            if unc["label"] == "HIGH" and not attn:
                st.markdown(f"**REVIEW** | {feat_text}")
            elif attn:
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

    # Clinical explanation (narrative + device-specific + temporal + chain)
    clin_exp = alert.get("clinical_explanation", {})
    if clin_exp:
        narrative = clin_exp.get("narrative", "")
        if narrative:
            st.info(f"**What happened:** {narrative}")

        dev_exp = clin_exp.get("device_explanation", {})
        if dev_exp.get("details"):
            st.markdown("**Device-specific findings:**")
            for detail in dev_exp["details"][:3]:
                st.markdown(f"- {detail}")
        if dev_exp.get("action"):
            st.success(f"**Recommended action:** {dev_exp['action']}")

        temporal = clin_exp.get("temporal_narrative", "")
        if temporal:
            st.caption(f"Timeline: {temporal}")

        patterns = clin_exp.get("attack_patterns", [])
        if patterns:
            top_match = patterns[0]
            st.warning(
                f"**Pattern match:** {top_match['attack_type']} "
                f"({top_match['similarity']:.0%} similarity)"
            )

        counterfactual = clin_exp.get("counterfactual", "")
        if counterfactual:
            st.caption(f"To clear: {counterfactual}")

        chain = clin_exp.get("risk_chain", [])
        if chain:
            with st.expander("Decision chain (why this risk level)"):
                for i, step in enumerate(chain, 1):
                    st.markdown(f"{i}. {step}")
    else:
        # Fallback to raw explanation if clinical explanation not available
        explanation = alert.get("explanation", {})
        if not isinstance(explanation, dict):
            explanation = {}
        exp_level = explanation.get("level", "none")
        if exp_level != "none":
            st.markdown("**Explanation**")
            top_feats = explanation.get("top_features", [])
            if top_feats:
                for feat in top_feats[:5]:
                    st.markdown(f"- **{feat.get('feature', '?')}**: {feat.get('importance', 0):.4f}")
            timesteps = explanation.get("timestep_importance", [])
            if timesteps:
                st.bar_chart(timesteps)

    rationale = alert.get("clinical_rationale", "")
    if rationale:
        st.caption(f"Rationale: {rationale}")

    # Active learning: show model uncertainty + feedback value
    from src.production.feedback_loop import FeedbackLoop
    unc = FeedbackLoop.compute_uncertainty(alert)
    if unc["label"] == "HIGH":
        st.warning(f"**Model uncertainty: {unc['label']}** — {unc['message']}")
    elif unc["label"] == "MEDIUM":
        st.info(f"Model confidence: {unc['label']} — {unc['message']}")
    else:
        st.caption(f"Model confidence: {unc['label']} — {unc['message']}")

    # Alert acknowledgment
    st.markdown("---")
    db = st.session_state.get("database")
    if db:
        # Check if already acknowledged (from DB alert or from dict)
        is_ack = alert.get("acknowledged", False)
        if is_ack:
            ack_by = alert.get("acknowledged_by", "Unknown")
            ack_at = alert.get("acknowledged_at", "")
            st.success(f"Acknowledged by **{ack_by}** at {ack_at[:19]}")
        else:
            col_ack, col_attack, col_safe = st.columns(3)
            user = "anonymous"
            if "auth_session" in st.session_state:
                user = st.session_state.auth_session.username

            def _ensure_db_id():
                db_id = alert.get("db_id")
                if not db_id:
                    db_id = db.insert_alert(alert)
                return db_id

            with col_ack:
                if st.button("Acknowledge", type="primary", width="stretch"):
                    db.acknowledge_alert(_ensure_db_id(), user)
                    st.success("Acknowledged")
            with col_attack:
                if st.button("Confirm Attack", type="secondary", width="stretch"):
                    aid = _ensure_db_id()
                    db.insert_feedback(
                        alert_id=aid, analyst=user, ground_truth=1,
                        confidence=1.0, notes="Analyst confirmed: true positive",
                    )
                    db.acknowledge_alert(aid, user)
                    st.success("Confirmed as attack. Feedback recorded.")
            with col_safe:
                if st.button("Mark Safe", type="secondary", width="stretch"):
                    aid = _ensure_db_id()
                    db.insert_feedback(
                        alert_id=aid, analyst=user, ground_truth=0,
                        confidence=1.0, notes="Clinical override: device verified safe",
                    )
                    db.acknowledge_alert(aid, user)
                    st.success("Marked safe. Override recorded.")

    # Close button
    if st.button("Close", width="stretch"):
        del st.session_state["_alert_detail_idx"]
        st.rerun()


def render_from_ground_truth(
    gt: Dict[str, Any],
    show_suppressed: bool = False,
    risk_filter: Optional[List[str]] = None,
) -> Optional[int]:  # noqa: ARG001
    """Render alert feed from static ground truth data.

    Args:
        gt: Ground truth data (unused; kept for panel API consistency).
        show_suppressed: Whether to show fatigue-suppressed alerts.
        risk_filter: List of risk levels to include.

    Returns:
        Selected alert index for SHAP explanation.
    """
    allowed = set(risk_filter) if risk_filter else {"MEDIUM", "HIGH", "CRITICAL"}

    # Try Phase 4 risk report first (has clinical severity + fatigue data)
    from dashboard.utils.loader import load_risk_report
    risk_report = load_risk_report()
    if risk_report:
        assessments = risk_report.get("sample_assessments", risk_report.get("risk_results", []))
        alerts = [a for a in assessments if a.get("risk_level", "NORMAL") in allowed]
        if alerts:
            return render_alert_table(alerts, show_suppressed=show_suppressed)

    # Fallback to Phase 5 explanation report
    from dashboard.utils.loader import load_explanation_report
    report = load_explanation_report()
    if report and "explanations" in report:
        filtered = [e for e in report["explanations"] if e.get("risk_level", "NORMAL") in allowed]
        return render_alert_table(filtered, show_suppressed=show_suppressed)

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

    col_filter1, col_filter2 = st.columns(2)
    with col_filter1:
        show_suppressed = st.checkbox("Show suppressed alerts", value=False,
                                      help="Include alerts suppressed by fatigue mitigation")
    with col_filter2:
        risk_filter = st.multiselect(
            "Risk levels",
            ["MEDIUM", "HIGH", "CRITICAL"],
            default=["HIGH", "CRITICAL"],
            help="MEDIUM = detected anomaly. HIGH/CRITICAL = confirmed threat.",
        )

    def _apply_risk_filter(alerts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if risk_filter:
            return [a for a in alerts if a.get("risk_level") in risk_filter]
        return alerts

    if live_alerts:
        return render_alert_table(_apply_risk_filter(live_alerts), show_suppressed=show_suppressed)

    return render_from_ground_truth(gt, show_suppressed=show_suppressed, risk_filter=risk_filter)
