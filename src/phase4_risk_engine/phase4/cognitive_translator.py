"""Cognitive translator — stakeholder-specific alert views.

Translates raw risk assessments into role-appropriate language and
context for four hospital stakeholders:

  SOC Analyst:       Technical network forensics view
  Clinician:         Patient safety focus, plain language
  Hospital CISO:     Compliance, incident metrics, risk posture
  Biomedical Eng:    Device-specific diagnostics and remediation

Each view contains only the information relevant to that role's
decision-making, reducing cognitive load and improving response time.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from .base import BaseDetector
from .risk_level import RiskLevel

logger = logging.getLogger(__name__)


def _risk_to_plain(risk: str) -> str:
    """Translate risk level to plain clinical language."""
    return {
        "NORMAL": "No concerns",
        "LOW": "Minor anomaly detected — no action needed",
        "MEDIUM": "Unusual activity — under investigation",
        "HIGH": "Active threat detected — device being secured",
        "CRITICAL": "Immediate safety concern — medical team notified",
    }.get(risk, risk)


def _severity_to_urgency(severity: int) -> str:
    """Map clinical severity to urgency language."""
    return {
        1: "Routine",
        2: "Non-urgent",
        3: "Urgent — respond within 1 hour",
        4: "Emergent — respond within 15 minutes",
        5: "CRITICAL — respond within 5 minutes",
    }.get(severity, "Unknown")


class CognitiveTranslator(BaseDetector):
    """Generate stakeholder-specific views from risk assessments.

    Produces four parallel views of each alert, each containing
    only the fields and language relevant to that stakeholder's
    decision-making context.

    Args:
        organization: Hospital/organization name for CISO reports.
    """

    def __init__(self, organization: str = "Hospital") -> None:
        self._org = organization
        self._stats: Dict[str, int] = {
            "total_translated": 0,
            "actionable_alerts": 0,
        }

    def translate(
        self,
        risk_results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Add stakeholder views to each risk result.

        Adds a 'stakeholder_views' dict with keys:
          - soc_analyst
          - clinician
          - ciso
          - biomed_engineer

        Only generates views for non-ROUTINE alerts (severity > 1)
        to avoid cognitive noise.

        Args:
            risk_results: Risk results with clinical_severity, etc.

        Returns:
            Same list with 'stakeholder_views' added.
        """
        for result in risk_results:
            severity = result.get("clinical_severity", 1)

            if severity <= 1:
                result["stakeholder_views"] = None
                continue

            views = {
                "soc_analyst": self._view_soc(result),
                "clinician": self._view_clinician(result),
                "ciso": self._view_ciso(result),
                "biomed_engineer": self._view_biomed(result),
            }
            result["stakeholder_views"] = views
            self._stats["total_translated"] += 1
            if severity >= 3:
                self._stats["actionable_alerts"] += 1

        return risk_results

    def _view_soc(self, r: Dict[str, Any]) -> Dict[str, Any]:
        """SOC Analyst view — network forensics context."""
        risk = r.get("risk_level", "UNKNOWN")
        attack_cat = r.get("attack_category", "unknown")
        attn_flag = r.get("attention_flag", False)
        cia = r.get("cia_scores", {})
        cia_max = r.get("cia_max_dimension", "")
        scenario = r.get("scenario", "")
        explanation = r.get("explanation", {})

        threat_type = attack_cat
        if attn_flag and attack_cat == "unknown":
            threat_type = "NOVEL/ZERO-DAY (attention anomaly)"

        return {
            "alert_level": risk,
            "threat_type": threat_type,
            "primary_cia_impact": {
                "C": "Confidentiality", "I": "Integrity", "A": "Availability",
            }.get(cia_max, cia_max),
            "cia_scores": cia,
            "operational_scenario": scenario,
            "attention_anomaly": attn_flag,
            "top_features": explanation.get("top_features", []),
            "recommended_actions": self._soc_actions(risk, attn_flag),
        }

    def _view_clinician(self, r: Dict[str, Any]) -> Dict[str, Any]:
        """Clinician view — patient safety in plain language."""
        risk = r.get("risk_level", "UNKNOWN")
        severity = r.get("clinical_severity", 1)
        device = r.get("device_type", r.get("attack_category", "medical device"))
        safety_flag = r.get("patient_safety_flag", False)
        device_action = r.get("device_action", "none")
        response_time = r.get("response_time_minutes", 0)

        message = _risk_to_plain(risk)
        urgency = _severity_to_urgency(severity)

        view: Dict[str, Any] = {
            "message": message,
            "urgency": urgency,
            "device": device,
            "patient_safety_concern": safety_flag,
        }

        if safety_flag:
            view["safety_guidance"] = (
                "A potential security threat has been detected on a medical device "
                "connected to this patient. Device readings should be verified "
                "manually. DO NOT power off the device — maintain patient care "
                "while IT security investigates."
            )

        if device_action == "isolate_network":
            view["device_status"] = "Device network access restricted — manual readings recommended"
        elif device_action == "restrict_network":
            view["device_status"] = "Device traffic limited to essential functions"
        else:
            view["device_status"] = "Device operating normally"

        if response_time > 0:
            view["expected_resolution"] = f"IT security responding within {response_time} minutes"

        return view

    def _view_ciso(self, r: Dict[str, Any]) -> Dict[str, Any]:
        """CISO view — compliance and risk posture."""
        risk = r.get("risk_level", "UNKNOWN")
        severity = r.get("clinical_severity", 1)
        attack_cat = r.get("attack_category", "unknown")
        cia = r.get("cia_scores", {})
        cia_max = r.get("cia_max_dimension", "")
        attn_flag = r.get("attention_flag", False)
        safety_flag = r.get("patient_safety_flag", False)

        compliance_impacts = []
        if cia.get("C", 0) > 0.5:
            compliance_impacts.append("HIPAA — potential PHI exposure")
        if cia.get("I", 0) > 0.5:
            compliance_impacts.append("FDA 21 CFR Part 11 — data integrity")
        if cia.get("A", 0) > 0.5:
            compliance_impacts.append("Joint Commission — care continuity")
        if safety_flag:
            compliance_impacts.append("FDA MDR — potential reportable event")

        return {
            "incident_severity": severity,
            "risk_classification": risk,
            "attack_category": attack_cat,
            "novel_threat": attn_flag,
            "cia_impact": {
                "C": "Confidentiality", "I": "Integrity", "A": "Availability",
            }.get(cia_max, "Unknown"),
            "compliance_impacts": compliance_impacts,
            "patient_safety_event": safety_flag,
            "requires_disclosure": severity >= 4 and any(cia.get("C", 0) > 0.5 for _ in [1]),
            "recommended_reporting": self._ciso_reporting(severity, safety_flag),
        }

    def _view_biomed(self, r: Dict[str, Any]) -> Dict[str, Any]:
        """Biomedical Engineer view — device diagnostics."""
        risk = r.get("risk_level", "UNKNOWN")
        severity = r.get("clinical_severity", 1)
        device = r.get("device_type", "unknown")
        device_action = r.get("device_action", "none")
        explanation = r.get("explanation", {})
        rationale = r.get("clinical_rationale", "")

        return {
            "device_type": device,
            "alert_severity": severity,
            "anomaly_indicators": explanation.get("top_features", []),
            "temporal_pattern": explanation.get("timestep_importance", []),
            "device_action_taken": device_action,
            "diagnostic_summary": rationale,
            "recommended_checks": self._biomed_checks(risk, device, device_action),
        }

    @staticmethod
    def _soc_actions(risk: str, novel: bool) -> List[str]:
        """Recommended SOC analyst actions."""
        actions = ["Review alert in SIEM"]
        if novel:
            actions.append("Classify novel threat — update signature database")
        if risk in ("HIGH", "CRITICAL"):
            actions.extend([
                "Capture network traffic for forensic analysis",
                "Check lateral movement indicators",
                "Verify no other devices affected",
            ])
        if risk == "CRITICAL":
            actions.append("Activate incident response playbook")
        return actions

    @staticmethod
    def _ciso_reporting(severity: int, safety_flag: bool) -> List[str]:
        """Required reporting for CISO."""
        reports = []
        if severity >= 3:
            reports.append("Internal incident report within 24 hours")
        if severity >= 4:
            reports.append("Board notification within 48 hours")
        if safety_flag:
            reports.append("FDA MedWatch if patient impact confirmed")
        if severity >= 4:
            reports.append("HHS breach notification assessment (45-day clock)")
        return reports

    @staticmethod
    def _biomed_checks(risk: str, device: str, action: str) -> List[str]:
        """Recommended biomedical engineering checks."""
        checks = ["Verify device firmware version against known CVEs"]
        if action in ("restrict_network", "isolate_network"):
            checks.append("Confirm device functions correctly in restricted mode")
            checks.append("Prepare backup monitoring equipment")
        if risk in ("HIGH", "CRITICAL"):
            checks.extend([
                "Check device audit log for unauthorized configuration changes",
                "Verify calibration status",
                "Inspect physical access indicators (tamper seals)",
            ])
        return checks

    def get_summary(self) -> Dict[str, Any]:
        """Return translation summary."""
        return {
            "total_translated": self._stats["total_translated"],
            "actionable_alerts": self._stats["actionable_alerts"],
        }

    def get_config(self) -> Dict[str, Any]:
        return {"organization": self._org}
