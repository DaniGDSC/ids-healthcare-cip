"""Panel 3 — Explanations (role-dispatched).

Renders role-appropriate explanation views:
  - IT Security Analyst: Full SHAP / gradient forensics (3 tabs)
  - Clinical IT Administrator: Device impact + CIA (3 tabs)
  - Attending Physician: Patient safety in plain language (2 tabs)
  - Hospital Manager: Aggregate threat trends (3 tabs)
  - Regulatory Auditor: Full model transparency + audit trail (3 tabs)

Supports both Phase 5 SHAP data and Phase 4 conditional explainer
(gradient + attention) data.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from dashboard.utils.metrics import risk_color, severity_label

# ── Constants ──────────────────────────────────────────────────────────

_BIOMETRIC = {"Temp", "SpO2", "Pulse_Rate", "SYS", "DIA", "Heart_rate", "Resp_Rate", "ST"}

_BIOMETRIC_LABELS: Dict[str, str] = {
    "Temp": "Body Temperature",
    "SpO2": "Blood Oxygen",
    "Pulse_Rate": "Pulse Rate",
    "SYS": "Systolic BP",
    "DIA": "Diastolic BP",
    "Heart_rate": "Heart Rate",
    "Resp_Rate": "Respiratory Rate",
    "ST": "ST Segment",
}

_URGENCY_MAP: Dict[int, tuple] = {
    1: ("Routine", "#95a5a6"),
    2: ("Advisory", "#3498db"),
    3: ("Urgent", "#f39c12"),
    4: ("Emergent", "#e67e22"),
    5: ("Critical", "#e74c3c"),
}

_DEVICE_STATUS: Dict[str, tuple] = {
    "none": ("Operating normally", "#2ecc71"),
    "restrict_network": ("Network traffic restricted", "#f39c12"),
    "isolate_network": ("Network isolated", "#e74c3c"),
}


# ── Shared helpers ─────────────────────────────────────────────────────

def _feature_color(feature_name: str) -> str:
    """Orange for biometric, blue for network."""
    return "#e67e22" if feature_name in _BIOMETRIC else "#3498db"


def _biometric_label(feature: str) -> str:
    """Map internal feature name to clinician-friendly label."""
    return _BIOMETRIC_LABELS.get(feature, feature)


def _aggregate_feature_drivers(alerts: List[Dict[str, Any]]) -> List[tuple]:
    """Count how often each feature appears in top_features across alerts.

    Returns list of (feature_name, count) sorted descending.
    """
    counter: Counter = Counter()
    for a in alerts:
        explanation = a.get("explanation") or {}
        for f in explanation.get("top_features", []):
            name = f.get("feature", "")
            if name:
                counter[name] += 1
    return counter.most_common()


def _render_cia_bars(cia_scores: Dict[str, float]) -> None:
    """Render three progress bars for CIA triad scores."""
    labels = {"C": "Confidentiality", "I": "Integrity", "A": "Availability"}
    colors = {"C": "#9b59b6", "I": "#e67e22", "A": "#3498db"}
    max_dim = max(cia_scores, key=cia_scores.get, default="I")
    for dim in ("C", "I", "A"):
        val = cia_scores.get(dim, 0)
        bold = " **" if dim == max_dim else ""
        end_bold = "**" if dim == max_dim else ""
        st.markdown(f"{bold}{labels[dim]} ({dim}): {val:.2f}{end_bold}")
        st.progress(min(val, 1.0))


def _risk_distribution_chart(alerts: List[Dict[str, Any]]) -> go.Figure:
    """Bar chart of risk level distribution across alerts."""
    levels = ["NORMAL", "LOW", "MEDIUM", "HIGH", "CRITICAL"]
    counts = Counter(a.get("risk_level", "NORMAL") for a in alerts)
    vals = [counts.get(l, 0) for l in levels]
    colors_list = [risk_color(l) for l in levels]

    fig = go.Figure(go.Bar(
        x=levels, y=vals,
        marker_color=colors_list,
        text=[str(v) for v in vals], textposition="auto",
    ))
    fig.update_layout(
        height=280, margin=dict(t=10, b=30, l=40, r=20),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e0e0e0", xaxis_title="Risk Level", yaxis_title="Count",
    )
    return fig


def _load_alerts(live_alerts: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """Load alerts from live buffer, Phase 4 risk report, or Phase 5 explanation report."""
    # Prefer live streaming alerts when available
    if live_alerts:
        return live_alerts

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


def _select_alert(alerts: List[Dict[str, Any]], default_idx: int = 0) -> Optional[Dict[str, Any]]:
    """Render alert selector and return the chosen alert."""
    if not alerts:
        st.info("No alerts available")
        return None
    options = [
        f"#{a.get('sample_index', i)} — {a.get('risk_level', '?')}"
        for i, a in enumerate(alerts)
    ]
    safe_default = min(default_idx, len(options) - 1)
    selected = st.selectbox("Select alert", options, index=safe_default)
    idx = options.index(selected)
    return alerts[idx]


# ── Shared visualization functions ─────────────────────────────────────

def render_waterfall(alert: Dict[str, Any]) -> None:
    """Render feature importance waterfall for a single alert.

    Supports both Phase 5 SHAP format (shap_value key) and
    Phase 4 conditional explainer format (importance key).
    """
    explanation = alert.get("explanation") or {}
    if not isinstance(explanation, dict):
        explanation = {}
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

    st.plotly_chart(fig, width="stretch")

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

    st.plotly_chart(fig, width="stretch")

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
    if not isinstance(explanation, dict):
        explanation = {}
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

    st.plotly_chart(fig, width="stretch")

    if timestep_weights:
        peak = int(np.argmax(timestep_weights))
        st.caption(f"Peak attention at timestep {peak} (weight={timestep_weights[peak]:.4f})")


# ═══════════════════════════════════════════════════════════════════════
# Role-specific renderers
# ═══════════════════════════════════════════════════════════════════════


def _render_analyst(
    gt: Dict[str, Any],
    alerts: List[Dict[str, Any]],
    selected_alert_idx: int,
) -> None:
    """IT Security Analyst — full SHAP / gradient forensics."""
    st.header("Threat Forensics")

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

    alert = _select_alert(alerts, selected_alert_idx)
    if alert is None:
        return

    with tab1:
        render_waterfall(alert)

    with tab3:
        baseline_thresh = gt.get("risk_adaptive", {}).get(
            "baseline_threshold", 0.204,
        )
        render_temporal_timeline(alert, baseline_thresh)


def _render_clinical_it(
    gt: Dict[str, Any],
    alerts: List[Dict[str, Any]],
    selected_alert_idx: int,
) -> None:
    """Clinical IT Administrator — device impact + CIA."""
    st.header("Device Explanation Intelligence")

    tab1, tab2, tab3 = st.tabs([
        "Device Impact",
        "CIA Impact",
        "Network Trends",
    ])

    with tab3:
        render_global_importance(gt)

    if not alerts:
        with tab1:
            st.info("No alerts available for device impact analysis")
        with tab2:
            st.info("No CIA data available")
        return

    alert = _select_alert(alerts, selected_alert_idx)
    if alert is None:
        return

    with tab1:
        explanation = alert.get("explanation") or {}
        top_features = explanation.get("top_features", alert.get("top_features", []))

        if not top_features:
            st.info("No feature data for this alert")
        else:
            # Split features by category
            bio_feats = [(f.get("feature", ""), float(f.get("importance", f.get("shap_value", 0)) or 0))
                         for f in top_features if f.get("feature", "") in _BIOMETRIC]
            net_feats = [(f.get("feature", ""), float(f.get("importance", f.get("shap_value", 0)) or 0))
                         for f in top_features if f.get("feature", "") not in _BIOMETRIC]

            # Combined bar chart with category coloring
            names = [f.get("feature", f"f{i}") for i, f in enumerate(top_features)]
            values = [float(f.get("importance", f.get("shap_value", 0)) or 0) for f in top_features]
            colors = [_feature_color(n) for n in names]

            fig = go.Figure(go.Bar(
                x=values, y=names, orientation="h",
                marker_color=colors,
                text=[f"{v:.5f}" for v in values], textposition="outside",
            ))
            fig.update_layout(
                title=f"Alert #{alert.get('sample_index', '?')} — Feature Drivers",
                xaxis_title="Importance", height=max(250, len(names) * 35 + 80),
                margin=dict(l=120, r=60, t=50, b=40),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e0e0e0", yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig, width="stretch")

            st.markdown(
                '<span style="color:#3498db;">&#9632;</span> Network &nbsp; '
                '<span style="color:#e67e22;">&#9632;</span> Biometric',
                unsafe_allow_html=True,
            )

            # Detail columns
            col_net, col_bio = st.columns(2)
            with col_net:
                st.markdown("**Network Indicators**")
                for name, val in net_feats:
                    st.markdown(f"- {name}: `{val:.6f}`")
                if not net_feats:
                    st.caption("None in top features")
            with col_bio:
                st.markdown("**Biometric Indicators**")
                for name, val in bio_feats:
                    st.markdown(f"- {name}: `{val:.6f}`")
                if not bio_feats:
                    st.caption("None in top features")

        # Device action badge
        action = alert.get("device_action", "none")
        status_text, status_color = _DEVICE_STATUS.get(action, ("Unknown", "#95a5a6"))
        st.markdown(
            f'<div style="background:{status_color}22; border:1px solid {status_color}; '
            f'padding:6px 12px; border-radius:4px; margin-top:12px;">'
            f'<strong style="color:{status_color};">Device Action:</strong> {status_text}</div>',
            unsafe_allow_html=True,
        )

    with tab2:
        cia_scores = alert.get("cia_scores", {})
        if cia_scores:
            scenario = alert.get("scenario", "unknown")
            st.markdown(f"**Operational Scenario:** {scenario.replace('_', ' ').title()}")
            st.markdown("---")
            _render_cia_bars(cia_scores)
        else:
            st.info("No CIA assessment data for this alert")

        # Temporal for selected alert
        st.markdown("---")
        baseline_thresh = gt.get("risk_adaptive", {}).get("baseline_threshold", 0.204)
        render_temporal_timeline(alert, baseline_thresh)


def _render_physician(
    gt: Dict[str, Any],
    alerts: List[Dict[str, Any]],
    selected_alert_idx: int,
) -> None:
    """Attending Physician — patient safety in plain language.

    NO raw scores, NO technical terms (SHAP, gradient, attention, MAD).
    """
    st.header("Patient Safety Explanations")

    # Filter to clinically relevant alerts (severity >= 3 or safety flag)
    clinical_alerts = [
        a for a in alerts
        if a.get("clinical_severity", 1) >= 3 or a.get("patient_safety_flag", False)
    ]

    tab1, tab2 = st.tabs(["Why Was This Flagged?", "Device Status Overview"])

    with tab1:
        if not clinical_alerts:
            st.success("No patient safety concerns at this time.")
            return

        # Simplified navigation: most urgent first
        clinical_alerts.sort(
            key=lambda a: (-a.get("clinical_severity", 1), -a.get("anomaly_score", 0)),
        )

        options = []
        for i, a in enumerate(clinical_alerts):
            sev = a.get("clinical_severity", 1)
            urgency, _ = _URGENCY_MAP.get(sev, ("Unknown", "#95a5a6"))
            options.append(f"Alert {i + 1} — {urgency}")

        selected = st.selectbox("Select concern", options, index=0)
        idx = options.index(selected)
        alert = clinical_alerts[idx]

        # Urgency badge
        sev = alert.get("clinical_severity", 1)
        urgency, urg_color = _URGENCY_MAP.get(sev, ("Unknown", "#95a5a6"))
        st.markdown(
            f'<div style="background:{urg_color}22; border:2px solid {urg_color}; '
            f'padding:10px 16px; border-radius:6px; text-align:center; margin-bottom:16px;">'
            f'<span style="color:{urg_color}; font-size:1.2em; font-weight:700;">'
            f'{urgency}</span></div>',
            unsafe_allow_html=True,
        )

        # Clinical explanation (narrative + device-specific)
        clin_exp = alert.get("clinical_explanation", {})
        if clin_exp:
            narrative = clin_exp.get("narrative", "")
            if narrative:
                st.info(f"**What happened:** {narrative}")

            dev_exp = clin_exp.get("device_explanation", {})
            if dev_exp.get("action"):
                st.success(f"**What to do:** {dev_exp['action']}")

            temporal = clin_exp.get("temporal_narrative", "")
            if temporal:
                st.caption(temporal)
        else:
            # Fallback to raw rationale
            rationale = alert.get("clinical_rationale", "")
            if rationale:
                if sev >= 4:
                    st.error(f"**Reason:** {rationale}")
                else:
                    st.warning(f"**Reason:** {rationale}")

        # Affected vital signs (biometric features only, friendly names)
        explanation = alert.get("explanation") or {}
        if not isinstance(explanation, dict):
            explanation = {}
        top_features = explanation.get("top_features", [])
        bio_features = [f for f in top_features if f.get("feature", "") in _BIOMETRIC]

        if bio_features:
            st.markdown("**Affected Vital Signs:**")
            max_imp = max(float(f.get("importance", f.get("shap_value", 0)) or 0)
                         for f in bio_features) or 1.0
            for f in bio_features:
                name = _biometric_label(f.get("feature", ""))
                imp = float(f.get("importance", f.get("shap_value", 0)) or 0)
                pct = min(abs(imp) / abs(max_imp), 1.0)
                st.markdown(f"**{name}**")
                st.progress(pct)

        # Device status
        action = alert.get("device_action", "none")
        status_text, status_color = _DEVICE_STATUS.get(action, ("Unknown", "#95a5a6"))
        st.markdown(
            f'<div style="background:{status_color}22; border:1px solid {status_color}; '
            f'padding:6px 12px; border-radius:4px; margin-top:12px;">'
            f'<strong>Device Status:</strong> {status_text}</div>',
            unsafe_allow_html=True,
        )

        # Response time
        resp_min = alert.get("response_time_minutes", 0)
        if resp_min and resp_min > 0:
            st.markdown(f"**Expected response within:** {resp_min} minutes")

        # Safety flag
        if alert.get("patient_safety_flag", False):
            st.error("**Patient Safety Flag:** Active — clinical assessment recommended")

    with tab2:
        if not clinical_alerts:
            st.success("All monitored devices operating normally.")
            return

        st.markdown("##### Flagged Devices")
        for i, a in enumerate(clinical_alerts):
            sev = a.get("clinical_severity", 1)
            urgency, urg_color = _URGENCY_MAP.get(sev, ("Unknown", "#95a5a6"))
            action = a.get("device_action", "none")
            status_text, _ = _DEVICE_STATUS.get(action, ("Unknown",))
            resp = a.get("response_time_minutes", 0)
            resp_str = f"{resp} min" if resp else "N/A"

            c1, c2, c3, c4 = st.columns([2, 2, 3, 2])
            c1.markdown(f"Alert {i + 1}")
            c2.markdown(f'<span style="color:{urg_color};">{urgency}</span>',
                        unsafe_allow_html=True)
            c3.markdown(status_text)
            c4.markdown(resp_str)

        # Summary counts
        safety_count = sum(1 for a in clinical_alerts if a.get("patient_safety_flag", False))
        if safety_count:
            st.warning(f"**{safety_count}** device(s) with active patient safety concerns")


def _render_manager(
    gt: Dict[str, Any],
    alerts: List[Dict[str, Any]],
    selected_alert_idx: int,
) -> None:
    """Hospital Manager — aggregate threat trends (no per-sample detail)."""
    st.header("Risk Explanation Summary")

    tab1, tab2, tab3 = st.tabs([
        "Threat Landscape",
        "Risk Posture",
        "Compliance Impact",
    ])

    with tab1:
        if not alerts:
            st.info("No threat data available")
        else:
            # Summary metrics
            total = len(alerts)
            avg_sev = sum(a.get("clinical_severity", 1) for a in alerts) / max(total, 1)
            safety_events = sum(1 for a in alerts if a.get("patient_safety_flag", False))
            novel_threats = sum(1 for a in alerts if a.get("attention_flag", False))

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Alerts", total)
            m2.metric("Avg Severity", f"{avg_sev:.1f}/5")
            m3.metric("Safety Events", safety_events)
            m4.metric("Novel Threats", novel_threats)

            # Primary detection drivers (aggregated)
            st.markdown("##### Primary Detection Drivers")
            drivers = _aggregate_feature_drivers(alerts)
            if drivers:
                names = [d[0] for d in drivers[:10]]
                counts = [d[1] for d in drivers[:10]]
                colors = [_feature_color(n) for n in names]

                fig = go.Figure(go.Bar(
                    x=counts, y=names, orientation="h",
                    marker_color=colors,
                    text=[str(c) for c in counts], textposition="auto",
                ))
                fig.update_layout(
                    height=max(250, len(names) * 30 + 60),
                    margin=dict(l=120, r=40, t=10, b=30),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#e0e0e0", xaxis_title="Frequency in alerts",
                    yaxis=dict(autorange="reversed"),
                )
                st.plotly_chart(fig, width="stretch")
                st.markdown(
                    '<span style="color:#3498db;">&#9632;</span> Network &nbsp; '
                    '<span style="color:#e67e22;">&#9632;</span> Biometric',
                    unsafe_allow_html=True,
                )
            else:
                st.caption("No feature attribution data in alerts")

            # Attack category breakdown
            st.markdown("##### Attack Category Breakdown")
            cat_counts = Counter(a.get("attack_category", "unknown") for a in alerts)
            for cat, count in cat_counts.most_common():
                pct = count / max(total, 1) * 100
                st.markdown(f"- **{cat.replace('_', ' ').title()}:** {count} ({pct:.0f}%)")

    with tab2:
        if not alerts:
            st.info("No risk data available")
        else:
            # Risk distribution chart
            st.markdown("##### Risk Level Distribution")
            fig = _risk_distribution_chart(alerts)
            st.plotly_chart(fig, width="stretch")

            # CIA dimension distribution
            st.markdown("##### CIA Dimension Exposure")
            cia_counts = Counter(a.get("cia_max_dimension", "I") for a in alerts)
            labels = {"C": "Confidentiality", "I": "Integrity", "A": "Availability"}
            for dim, label in labels.items():
                count = cia_counts.get(dim, 0)
                pct = count / max(len(alerts), 1) * 100
                st.markdown(f"- **{label}:** {count} alerts ({pct:.0f}%)")

            # Alert management
            st.markdown("##### Alert Management")
            emitted = sum(1 for a in alerts if a.get("alert_emit", True))
            suppressed = len(alerts) - emitted
            st.markdown(f"- **Emitted:** {emitted}")
            st.markdown(f"- **Suppressed (fatigue mitigation):** {suppressed}")
            if len(alerts) > 0:
                st.markdown(f"- **Suppression rate:** {suppressed / len(alerts) * 100:.0f}%")

    with tab3:
        if not alerts:
            st.info("No compliance data available")
        else:
            st.markdown("##### Regulatory Impact Assessment")

            # Scan for compliance triggers
            high_critical = [a for a in alerts
                            if a.get("risk_level", "") in ("HIGH", "CRITICAL")]
            safety_events = [a for a in alerts if a.get("patient_safety_flag", False)]
            integrity_alerts = [a for a in alerts
                               if a.get("cia_max_dimension", "") == "I"]
            conf_alerts = [a for a in alerts if a.get("cia_max_dimension", "") == "C"]

            # HIPAA
            if conf_alerts:
                st.warning(f"**HIPAA (45 CFR 164):** {len(conf_alerts)} confidentiality "
                           f"alerts — potential PHI exposure. Breach risk assessment required.")
            else:
                st.success("**HIPAA:** No confidentiality concerns detected.")

            # FDA 21 CFR Part 11
            if integrity_alerts:
                st.warning(f"**FDA 21 CFR Part 11:** {len(integrity_alerts)} integrity "
                           f"alerts — data integrity validation recommended.")
            else:
                st.success("**FDA 21 CFR Part 11:** Data integrity maintained.")

            # Joint Commission
            if safety_events:
                st.error(f"**Joint Commission:** {len(safety_events)} patient safety "
                         f"event(s) — incident report filing required.")
            else:
                st.success("**Joint Commission:** No patient safety events.")

            # FDA MDR
            if high_critical:
                st.warning(f"**FDA MDR (21 CFR 803):** {len(high_critical)} HIGH/CRITICAL "
                           f"device alerts — medical device report may be required.")
            else:
                st.success("**FDA MDR:** No reportable device events.")

            # Disclosure requirements
            needs_disclosure = len(safety_events) > 0 or len(conf_alerts) > 0
            if needs_disclosure:
                st.markdown("---")
                st.error("**Disclosure may be required** — review with compliance officer.")


def _render_auditor(
    gt: Dict[str, Any],
    alerts: List[Dict[str, Any]],
    selected_alert_idx: int,
) -> None:
    """Regulatory Auditor — full model transparency + audit trail."""
    st.header("Model Transparency & Decision Audit")

    tab1, tab2, tab3 = st.tabs([
        "Detection Methodology",
        "Threshold Justification",
        "Decision Audit Trail",
    ])

    with tab1:
        st.markdown("##### Model Architecture")
        st.markdown(
            "- **Type:** CNN-BiLSTM-Attention (hybrid deep learning)\n"
            "- **Input:** Sliding window of 20 network flows x 24 features\n"
            "- **Output:** Binary anomaly score (sigmoid, 0-1)\n"
            "- **Training:** Phase 2 backbone + Phase 2.5 fine-tuned classification head"
        )

        st.markdown("##### Explanation Methodology")
        st.markdown(
            "Two-tier explanation approach balances latency with interpretability:\n\n"
            "1. **Tier 1 — Attention Context** (all predictions, ~1ms): "
            "Backbone attention vector magnitude detects novel/unseen attack patterns. "
            "Divergence from baseline triggers escalation.\n"
            "2. **Tier 2 — Gradient Attribution** (MEDIUM+ risk, ~5ms): "
            "TensorFlow GradientTape computes dScore/dInput — identifies which "
            "features and timesteps drove the anomaly score upward."
        )

        st.markdown("##### Feature Space (24 dimensions)")
        from dashboard.streaming.feature_aligner import MODEL_FEATURES
        col_net, col_bio = st.columns(2)
        with col_net:
            st.markdown("**Network (16):**")
            for f in MODEL_FEATURES:
                if f not in _BIOMETRIC:
                    st.markdown(f"- {f}")
        with col_bio:
            st.markdown("**Biometric (8):**")
            for f in MODEL_FEATURES:
                if f in _BIOMETRIC:
                    st.markdown(f"- {f} ({_biometric_label(f)})")

        st.markdown("---")
        render_global_importance(gt)

    with tab2:
        risk_adaptive = gt.get("risk_adaptive", {})

        st.markdown("##### Baseline Statistics")
        median = risk_adaptive.get("median", "N/A")
        mad = risk_adaptive.get("mad", "N/A")
        threshold = risk_adaptive.get("baseline_threshold", "N/A")
        multiplier = risk_adaptive.get("mad_multiplier", 3.0)

        st.markdown(f"- **Median anomaly score (benign):** {median}")
        st.markdown(f"- **MAD (median absolute deviation):** {mad}")
        st.markdown(f"- **MAD multiplier:** {multiplier}")
        st.markdown(f"- **Baseline threshold:** {threshold}")
        st.markdown(f"- **Formula:** threshold = median + {multiplier} x MAD")

        st.markdown("##### Risk Level Mapping")
        st.markdown(
            "| Level | Condition |\n"
            "|-------|-----------|\n"
            "| NORMAL | distance < 0 (score below threshold) |\n"
            "| LOW | 0 <= distance < 1.0 x MAD |\n"
            "| MEDIUM | 1.0 x MAD <= distance < 2.0 x MAD |\n"
            "| HIGH | 2.0 x MAD <= distance < 3.0 x MAD |\n"
            "| CRITICAL | distance >= 3.0 x MAD + cross-modal confirmation |"
        )

        st.markdown("##### Calibration Methodology")
        st.markdown(
            "Raw sigmoid scores cluster in a narrow range (0.88-0.98) where "
            "benign and attack overlap. The **ScoreCalibrator** maps each score "
            "to its percentile rank within the observed benign distribution:\n\n"
            "| Percentile | Risk Level |\n"
            "|-----------|------------|\n"
            "| < P75 | NORMAL |\n"
            "| P75-P85 | LOW |\n"
            "| P85-P93 | MEDIUM |\n"
            "| P93-P97 | HIGH |\n"
            "| >= P97 | CRITICAL |"
        )

        # Actual distribution if alerts available
        if alerts:
            st.markdown("##### Observed Risk Distribution")
            fig = _risk_distribution_chart(alerts)
            st.plotly_chart(fig, width="stretch")

    with tab3:
        if not alerts:
            st.info("No alerts available for audit review")
            return

        alert = _select_alert(alerts, selected_alert_idx)
        if alert is None:
            return

        st.markdown(f"##### Decision Chain — Alert #{alert.get('sample_index', '?')}")

        # Step 1: Raw inference
        st.markdown("**1. Model Inference**")
        score = alert.get("anomaly_score", "N/A")
        threshold = alert.get("threshold", "N/A")
        distance = alert.get("distance", "N/A")
        percentile = alert.get("percentile")
        st.markdown(f"- Anomaly score: `{score}`")
        st.markdown(f"- Threshold: `{threshold}`")
        st.markdown(f"- Distance: `{distance}`")
        if percentile is not None:
            st.markdown(f"- Calibrated percentile: `{percentile}`")

        # Step 2: Risk classification
        st.markdown("**2. Risk Classification**")
        risk = alert.get("risk_level", "N/A")
        attn = alert.get("attention_flag", False)
        st.markdown(f"- Risk level: **{risk}**")
        st.markdown(f"- Attention anomaly flag: `{attn}`")
        if attn:
            st.caption("Attention divergence from baseline — potential novel threat.")

        # Step 3: CIA modification
        st.markdown("**3. CIA Risk Modification**")
        cia = alert.get("cia_scores", {})
        scenario = alert.get("scenario", "N/A")
        max_dim = alert.get("cia_max_dimension", "N/A")
        st.markdown(f"- Scenario: {scenario}")
        st.markdown(f"- CIA scores: C={cia.get('C', 0):.2f}, "
                    f"I={cia.get('I', 0):.2f}, A={cia.get('A', 0):.2f}")
        st.markdown(f"- Primary dimension: **{max_dim}**")

        # Step 4: Clinical assessment
        st.markdown("**4. Clinical Impact Assessment**")
        sev = alert.get("clinical_severity", "N/A")
        sev_name = alert.get("clinical_severity_name", "")
        safety = alert.get("patient_safety_flag", False)
        rationale = alert.get("clinical_rationale", "")
        st.markdown(f"- Clinical severity: **{sev_name}** ({sev}/5)")
        st.markdown(f"- Patient safety flag: `{safety}`")
        st.markdown(f"- Rationale: {rationale}")

        # Step 5: Response protocol
        st.markdown("**5. Response Protocol**")
        resp = alert.get("response_time_minutes", "N/A")
        action = alert.get("device_action", "none")
        st.markdown(f"- Response time: {resp} minutes")
        st.markdown(f"- Device action: {action}")

        # Step 6: Alert fatigue
        st.markdown("**6. Alert Fatigue Decision**")
        emit = alert.get("alert_emit", True)
        reason = alert.get("alert_reason", "N/A")
        st.markdown(f"- Alert emitted: `{emit}`")
        st.markdown(f"- Reason: {reason}")

        # Step 7: Explanation
        st.markdown("**7. Explanation Provided**")
        explanation = alert.get("explanation") or {}
        level = explanation.get("level", "none")
        st.markdown(f"- Explanation level: `{level}`")
        top_features = explanation.get("top_features", [])
        if top_features:
            st.markdown("- Top features:")
            for f in top_features:
                name = f.get("feature", "?")
                imp = f.get("importance", f.get("shap_value", 0))
                cat = "biometric" if name in _BIOMETRIC else "network"
                st.markdown(f"  - {name} ({cat}): `{imp}`")


# ═══════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════

_RENDERERS = {
    "IT Security Analyst": _render_analyst,
    "Clinical IT Administrator": _render_clinical_it,
    "Attending Physician": _render_physician,
    "Hospital Manager": _render_manager,
    "Regulatory Auditor": _render_auditor,
}


def render(
    gt: Dict[str, Any],
    selected_alert_idx: Optional[int] = None,
    role: str = "IT Security Analyst",
    live_alerts: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """Render the Explanations panel with role-appropriate content."""
    alerts = _load_alerts(live_alerts)
    default_idx = selected_alert_idx if selected_alert_idx is not None else 0
    renderer = _RENDERERS.get(role, _render_analyst)
    renderer(gt, alerts, default_idx)
