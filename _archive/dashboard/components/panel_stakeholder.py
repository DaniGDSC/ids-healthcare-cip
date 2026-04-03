"""Panel — Stakeholder-Specific Views (Cognitive Translation).

Provides role-appropriate visualizations of risk assessments:
  SOC Analyst:       Threat matrix, CIA impact, forensic indicators
  Clinician:         Patient safety status, plain-language guidance
  CISO:              Compliance dashboard, incident metrics, reporting
  Biomed Engineer:   Device diagnostics, anomaly indicators, remediation

Reads from live buffer alerts during streaming, or falls back to
Phase 4 risk_report.json for static analysis.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RISK_REPORT_PATH = PROJECT_ROOT / "data" / "phase4" / "risk_report.json"

# ── Inline translation maps (for live predictions without stakeholder_views) ──

_RISK_PLAIN = {
    "NORMAL": "No concerns",
    "LOW": "Minor anomaly detected — no action needed",
    "MEDIUM": "Unusual activity — under investigation",
    "HIGH": "Active threat detected — device being secured",
    "CRITICAL": "Immediate safety concern — medical team notified",
}

_URGENCY_MAP = {
    1: "Routine",
    2: "Non-urgent",
    3: "Urgent — respond within 1 hour",
    4: "Emergent — respond within 15 minutes",
    5: "CRITICAL — respond within 5 minutes",
}

_DEVICE_STATUS_MAP = {
    "none": "Operating normally",
    "restrict_network": "Network traffic restricted",
    "isolate_network": "Network isolated — manual readings recommended",
}


def _default_checks(device_action: str) -> List[str]:
    """Default BioMed checks based on device action."""
    checks = ["Verify sensor readings against manual measurement"]
    if device_action == "restrict_network":
        checks.append("Confirm device data is still reaching monitoring station")
    if device_action == "isolate_network":
        checks.extend([
            "Switch to manual vital sign monitoring",
            "Check physical connections and cables",
            "Verify firmware version is current",
        ])
    return checks


def _ensure_views(a: Dict[str, Any]) -> Dict[str, Any]:
    """Generate minimal stakeholder_views from raw fields if absent.

    During live streaming, predictions arrive without pre-computed
    stakeholder_views. This creates inline translations from raw
    prediction fields so all renderers have data to display.
    """
    if a.get("stakeholder_views"):
        return a

    risk = a.get("risk_level", "NORMAL")
    sev = a.get("clinical_severity", 1)
    explanation = a.get("explanation") or {}

    a["stakeholder_views"] = {
        "soc_analyst": {
            "threat_type": a.get("attack_category", "unknown"),
            "primary_cia_impact": a.get("cia_max_dimension", ""),
        },
        "clinician": {
            "urgency": _URGENCY_MAP.get(sev, "Unknown"),
            "device": a.get("device_id", "Medical device"),
            "message": _RISK_PLAIN.get(risk, risk),
            "device_status": _DEVICE_STATUS_MAP.get(
                a.get("device_action", "none"), "Operating normally",
            ),
            "safety_guidance": (
                "Monitor device output. Contact clinical IT if readings appear abnormal."
                if sev >= 3 else ""
            ),
            "expected_resolution": (
                f"IT security responding within "
                f"{a.get('response_time_minutes', 60)} minutes"
            ),
        },
        "ciso": {},
        "biomed_engineer": {
            "device_type": a.get("device_id", "unknown"),
            "diagnostic_summary": a.get("clinical_rationale", ""),
            "anomaly_indicators": explanation.get("top_features", []),
            "temporal_pattern": explanation.get("timestep_importance", []),
            "device_action_taken": a.get("device_action", "none"),
            "recommended_checks": _default_checks(a.get("device_action", "none")),
        },
    }
    return a


@st.cache_data(ttl=30)
def _load_risk_report() -> Dict[str, Any]:
    """Load Phase 4 risk report."""
    if not RISK_REPORT_PATH.exists():
        return {}
    with open(RISK_REPORT_PATH) as f:
        return json.load(f)


def _get_assessments(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract sample assessments from risk report."""
    return report.get("sample_assessments", report.get("risk_results", []))


# ═══════════════════════════════════════════════════════════════════
# SOC Analyst View
# ═══════════════════════════════════════════════════════════════════

def _render_soc(assessments: List[Dict[str, Any]]) -> None:
    """SOC Analyst: Threat matrix and forensic indicators."""
    st.markdown("### Threat Intelligence Overview")

    actionable = [a for a in assessments if a.get("clinical_severity", 1) >= 2]
    if not actionable:
        st.success("No actionable threats detected.")
        return

    # Summary metrics
    c1, c2, c3, c4 = st.columns(4)
    n_novel = sum(1 for a in actionable if a.get("attention_flag", False))
    cia_dims = [a.get("cia_max_dimension", "") for a in actionable if a.get("cia_max_dimension")]
    top_cia = max(set(cia_dims), key=cia_dims.count) if cia_dims else "N/A"

    c1.metric("Active Threats", len(actionable))
    c2.metric("Novel/Zero-Day", n_novel)
    c3.metric("Primary CIA Impact", {"C": "Confidentiality", "I": "Integrity", "A": "Availability"}.get(top_cia, top_cia))
    c4.metric("Alerts Emitted", sum(1 for a in actionable if a.get("alert_emit", True)))

    # Threat table
    st.markdown("#### Threat Details")
    rows = []
    for a in actionable[:50]:
        view = (a.get("stakeholder_views") or {}).get("soc_analyst", {})
        rows.append({
            "Sample": a.get("sample_index", ""),
            "Severity": a.get("clinical_severity", 0),
            "Threat Type": view.get("threat_type", a.get("attack_category", "unknown")),
            "CIA Impact": view.get("primary_cia_impact", a.get("cia_max_dimension", "")),
            "Scenario": a.get("scenario", ""),
            "Attention": "NOVEL" if a.get("attention_flag") else "",
            "Action": a.get("device_action", "none"),
        })
    st.dataframe(rows, width="stretch")

    # Top features driving detections
    st.markdown("#### Top Anomaly Indicators")
    feature_counts: Dict[str, int] = {}
    for a in actionable:
        explanation = a.get("explanation", {})
        for feat in explanation.get("top_features", []):
            name = feat.get("feature", "")
            if name:
                feature_counts[name] = feature_counts.get(name, 0) + 1
    if feature_counts:
        sorted_feats = sorted(feature_counts.items(), key=lambda x: -x[1])[:10]
        for feat, count in sorted_feats:
            st.progress(count / max(1, len(actionable)), text=f"{feat}: {count} alerts")


# ═══════════════════════════════════════════════════════════════════
# Clinician View
# ═══════════════════════════════════════════════════════════════════

def _render_clinician(assessments: List[Dict[str, Any]]) -> None:
    """Clinician: Patient safety status in plain language."""
    st.markdown("### Patient Safety Status")

    safety_events = [a for a in assessments if a.get("patient_safety_flag", False)]
    active_threats = [a for a in assessments if a.get("clinical_severity", 1) >= 3]

    if not safety_events and not active_threats:
        st.success("All monitored devices operating normally. No patient safety concerns.")
        return

    if safety_events:
        st.error(f"**{len(safety_events)} patient safety event(s) detected**")
        for event in safety_events[:5]:
            view = (event.get("stakeholder_views") or {}).get("clinician", {})
            with st.container():
                st.markdown(f"**{view.get('urgency', 'Alert')}**")
                st.markdown(f"Device: **{view.get('device', 'Medical device')}**")
                st.markdown(f"Status: {view.get('device_status', '')}")
                if view.get("safety_guidance"):
                    st.warning(view["safety_guidance"])
                if view.get("expected_resolution"):
                    st.info(view["expected_resolution"])
                st.markdown("---")
    else:
        st.warning(f"{len(active_threats)} device(s) under investigation — no patient safety impact confirmed.")
        for threat in active_threats[:5]:
            view = (threat.get("stakeholder_views") or {}).get("clinician", {})
            msg = view.get("message", "")
            device = view.get("device", "")
            if msg or device:
                st.info(f"{msg} | Device: {device}")


# ═══════════════════════════════════════════════════════════════════
# CISO View
# ═══════════════════════════════════════════════════════════════════

def _render_ciso(assessments: List[Dict[str, Any]]) -> None:
    """CISO: Compliance dashboard and incident metrics."""
    st.markdown("### Security Posture & Compliance")

    total = len(assessments)
    by_severity: Dict[str, int] = {}
    for a in assessments:
        sev = a.get("clinical_severity_name", "ROUTINE")
        by_severity[sev] = by_severity.get(sev, 0) + 1

    # Posture summary
    c1, c2, c3 = st.columns(3)
    n_incidents = sum(1 for a in assessments if a.get("clinical_severity", 1) >= 3)
    n_safety = sum(1 for a in assessments if a.get("patient_safety_flag", False))
    n_novel = sum(1 for a in assessments if a.get("attention_flag", False))

    c1.metric("Security Incidents", n_incidents, help="Severity >= URGENT")
    c2.metric("Patient Safety Events", n_safety)
    c3.metric("Novel Threats", n_novel)

    # Severity distribution
    st.markdown("#### Incident Severity Distribution")
    for sev_name in ["CRITICAL", "EMERGENT", "URGENT", "ADVISORY", "ROUTINE"]:
        count = by_severity.get(sev_name, 0)
        pct = count / max(total, 1)
        st.progress(pct, text=f"{sev_name}: {count} ({pct:.1%})")

    # Compliance impacts
    st.markdown("#### Regulatory Impact Assessment")
    compliance_impacts: Dict[str, int] = {}
    reporting_required: List[str] = []
    for a in assessments:
        view = (a.get("stakeholder_views") or {}).get("ciso", {})
        for impact in view.get("compliance_impacts", []):
            compliance_impacts[impact] = compliance_impacts.get(impact, 0) + 1
        for report in view.get("recommended_reporting", []):
            if report not in reporting_required:
                reporting_required.append(report)

    if compliance_impacts:
        for impact, count in sorted(compliance_impacts.items(), key=lambda x: -x[1]):
            st.markdown(f"- **{impact}** ({count} events)")
    else:
        # Generate inline compliance assessment from raw fields
        high_crit = [a for a in assessments if a.get("risk_level") in ("HIGH", "CRITICAL")]
        safety = [a for a in assessments if a.get("patient_safety_flag")]
        integrity = [a for a in assessments if a.get("cia_max_dimension") == "I"]
        conf = [a for a in assessments if a.get("cia_max_dimension") == "C"]

        if conf:
            st.warning(f"**HIPAA:** {len(conf)} confidentiality alerts — breach risk assessment required")
        if integrity:
            st.warning(f"**FDA 21 CFR Part 11:** {len(integrity)} integrity alerts — data validation recommended")
        if safety:
            st.error(f"**Joint Commission:** {len(safety)} patient safety events — incident report required")
        if not conf and not integrity and not safety:
            st.success("No compliance impacts identified.")

    if reporting_required:
        st.markdown("#### Required Reporting Actions")
        for action in reporting_required:
            st.markdown(f"- {action}")

    # Alert fatigue metrics
    st.markdown("#### Alert Management")
    emitted = sum(1 for a in assessments if a.get("alert_emit", True))
    suppressed = sum(1 for a in assessments if not a.get("alert_emit", True))
    st.metric("Alerts Emitted", emitted)
    st.metric("Alerts Suppressed (fatigue mitigation)", suppressed)
    if total > 0:
        st.metric("Suppression Rate", f"{suppressed / total:.1%}")


# ═══════════════════════════════════════════════════════════════════
# Biomedical Engineer View
# ═══════════════════════════════════════════════════════════════════

def _render_biomed(assessments: List[Dict[str, Any]]) -> None:
    """Biomedical Engineer: Device diagnostics and remediation."""
    st.markdown("### Device Security Diagnostics")

    affected = [a for a in assessments if a.get("clinical_severity", 1) >= 2]
    if not affected:
        st.success("All devices operating within normal parameters.")
        return

    # Device action summary
    actions: Dict[str, int] = {}
    for a in affected:
        action = a.get("device_action", "none")
        actions[action] = actions.get(action, 0) + 1

    c1, c2, c3 = st.columns(3)
    c1.metric("Devices Affected", len(affected))
    c2.metric("Network Isolated", actions.get("isolate_network", 0))
    c3.metric("Traffic Restricted", actions.get("restrict_network", 0))

    # Device details
    st.markdown("#### Affected Device Details")
    for a in affected[:20]:
        view = (a.get("stakeholder_views") or {}).get("biomed_engineer", {})
        if not view:
            continue

        severity = a.get("clinical_severity", 1)
        with st.expander(
            f"Sample {a.get('sample_index', '?')} | "
            f"Severity {severity} | "
            f"{view.get('device_type', 'unknown')} | "
            f"Action: {view.get('device_action_taken', 'none')}"
        ):
            st.markdown(f"**Diagnostic Summary:** {view.get('diagnostic_summary', 'N/A')}")

            if view.get("anomaly_indicators"):
                st.markdown("**Anomaly Indicators:**")
                for ind in view["anomaly_indicators"]:
                    imp = ind.get("importance", ind.get("shap_value", 0))
                    st.markdown(f"- {ind.get('feature', '')}: importance={float(imp):.4f}")

            if view.get("temporal_pattern"):
                st.markdown("**Temporal Pattern (attention weights):**")
                st.bar_chart(view["temporal_pattern"])

            if view.get("recommended_checks"):
                st.markdown("**Recommended Checks:**")
                for check in view["recommended_checks"]:
                    st.checkbox(check, key=f"check_{a.get('sample_index')}_{check[:20]}")


# ═══════════════════════════════════════════════════════════════════
# Main Render
# ═══════════════════════════════════════════════════════════════════

def render(
    gt: Dict[str, Any],
    role: str = "IT Security Analyst",
    live_alerts: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """Render stakeholder-specific view based on current role.

    Uses live buffer alerts during streaming, falls back to static
    Phase 4 risk_report.json for offline analysis.

    Args:
        gt: Ground truth data (may include Phase 4 risk data).
        role: Current user role from sidebar selector.
        live_alerts: Live alerts from buffer (preferred over static data).
    """
    st.markdown("## Stakeholder Intelligence View")

    # Prefer live data during streaming
    if live_alerts:
        assessments = live_alerts
    else:
        report = _load_risk_report()
        assessments = _get_assessments(report)

    if not assessments:
        st.info(
            "No risk data available. Start streaming or run Phase 4 "
            "pipeline to generate stakeholder views."
        )
        return

    # Ensure all assessments have stakeholder_views (generate inline if absent)
    assessments = [_ensure_views(a) for a in assessments]

    # Role-based rendering (fixed mapping)
    role_map = {
        "IT Security Analyst": _render_soc,
        "Clinical IT Administrator": _render_soc,
        "Attending Physician": _render_clinician,
        "Hospital Manager": _render_ciso,
        "Regulatory Auditor": _render_ciso,
    }

    renderer = role_map.get(role, _render_soc)
    renderer(assessments)

    # Biomed view visible to technical roles (not hidden in expander)
    if role in ("IT Security Analyst", "Clinical IT Administrator", "Hospital Manager"):
        st.markdown("---")
        _render_biomed(assessments)
