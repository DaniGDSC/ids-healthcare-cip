"""Panel — Stakeholder-Specific Views (Cognitive Translation).

Provides role-appropriate visualizations of Phase 4 risk assessments:
  SOC Analyst:       Threat matrix, CIA impact, forensic indicators
  Clinician:         Patient safety status, plain-language guidance
  CISO:              Compliance dashboard, incident metrics, reporting
  Biomed Engineer:   Device diagnostics, anomaly indicators, remediation

Reads from Phase 4 risk_report.json which now includes
stakeholder_views, clinical_severity, and alert_fatigue fields.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RISK_REPORT_PATH = PROJECT_ROOT / "data" / "phase4" / "risk_report.json"


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


def _severity_color(severity: int) -> str:
    """Map clinical severity to display color."""
    return {1: "green", 2: "blue", 3: "orange", 4: "red", 5: "red"}.get(severity, "gray")


def _severity_icon(severity: int) -> str:
    return {1: "ok", 2: "info", 3: "warn", 4: "alert", 5: "crit"}.get(severity, "?")


# ═══════════════════════════════════════════════════════════════════
# SOC Analyst View
# ═══════════════════════════════════════════════════════════════════

def _render_soc(assessments: List[Dict[str, Any]]) -> None:
    """SOC Analyst: Threat matrix and forensic indicators."""
    st.markdown("### Threat Intelligence Overview")

    actionable = [a for a in assessments if a.get("clinical_severity", 1) >= 3]
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
            "CIA Impact": view.get("primary_cia_impact", ""),
            "Scenario": a.get("scenario", ""),
            "Attention": "NOVEL" if a.get("attention_flag") else "",
            "Action": a.get("device_action", "none"),
        })
    st.dataframe(rows, use_container_width=True)

    # Top features driving detections
    st.markdown("#### Top Anomaly Indicators")
    feature_counts: Dict[str, int] = {}
    for a in actionable:
        explanation = a.get("explanation", {})
        for feat in explanation.get("top_features", []):
            name = feat.get("feature", "")
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
                if "safety_guidance" in view:
                    st.warning(view["safety_guidance"])
                if "expected_resolution" in view:
                    st.info(view["expected_resolution"])
                st.markdown("---")
    else:
        st.warning(f"{len(active_threats)} device(s) under investigation — no patient safety impact confirmed.")
        for threat in active_threats[:5]:
            view = (threat.get("stakeholder_views") or {}).get("clinician", {})
            st.info(f"{view.get('message', '')} | Device: {view.get('device', '')}")


# ═══════════════════════════════════════════════════════════════════
# CISO View
# ═══════════════════════════════════════════════════════════════════

def _render_ciso(assessments: List[Dict[str, Any]]) -> None:
    """CISO: Compliance dashboard and incident metrics."""
    st.markdown("### Security Posture & Compliance")

    total = len(assessments)
    by_severity = {}
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
                    st.markdown(f"- {ind.get('feature', '')}: importance={ind.get('importance', 0):.4f}")

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

def render(gt: Dict[str, Any], role: str = "IT Security Analyst") -> None:
    """Render stakeholder-specific view based on current role.

    Args:
        gt: Ground truth data (may include Phase 4 risk data).
        role: Current user role from sidebar selector.
    """
    st.markdown("## Stakeholder Intelligence View")

    report = _load_risk_report()
    assessments = _get_assessments(report)

    if not assessments:
        st.info(
            "Phase 4 risk report not available. "
            "Run Phase 4 pipeline to generate stakeholder views."
        )
        return

    # Check if cognitive translation was applied
    has_views = any(a.get("stakeholder_views") for a in assessments)
    if not has_views:
        st.warning(
            "Risk report does not include stakeholder views. "
            "Re-run Phase 4 with CIA enabled to generate cognitive translations."
        )

    # Role-based rendering
    role_map = {
        "IT Security Analyst": _render_soc,
        "Clinical IT Administrator": _render_ciso,
        "Attending Physician": _render_clinician,
        "Hospital Manager": _render_ciso,
        "Regulatory Auditor": _render_ciso,
    }

    renderer = role_map.get(role, _render_soc)
    renderer(assessments)

    # Biomed view available to IT Security and Clinical IT
    if role in ("IT Security Analyst", "Clinical IT Administrator"):
        with st.expander("Biomedical Engineering View"):
            _render_biomed(assessments)
