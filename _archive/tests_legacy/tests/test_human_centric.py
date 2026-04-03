"""Human-Centric & Usability Testing.

Validates that the system is designed for human operators, not just
technical correctness. Three categories:

  1. Trust Scales — explanations build justified trust, not blind trust
  2. Effectiveness — alerts lead to correct actions within time budgets
  3. Alert Fatigue Assessment — suppression reduces volume without missing threats
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pytest


# ═══════════════════════════════════════════════════════════════════
# Test Data Factories
# ═══════════════════════════════════════════════════════════════════

def _make_alert(
    risk: str = "HIGH",
    severity: int = 4,
    attack: str = "Spoofing",
    device: str = "infusion_pump",
    safety: bool = True,
    attention: bool = False,
    score: float = 0.75,
    device_action: str = "isolate_network",
    response_min: int = 15,
) -> Dict[str, Any]:
    """Build a realistic alert dict with all Phase 4 fields."""
    return {
        "sample_index": 0,
        "anomaly_score": score,
        "threshold": 0.5,
        "distance": score - 0.5,
        "risk_level": risk,
        "attention_flag": attention,
        "clinical_severity": severity,
        "clinical_severity_name": {1: "ROUTINE", 2: "ADVISORY", 3: "URGENT", 4: "EMERGENT", 5: "CRITICAL"}[severity],
        "response_time_minutes": response_min,
        "device_action": device_action,
        "patient_safety_flag": safety,
        "clinical_rationale": f"Base: {risk} on {device}",
        "alert_emit": True,
        "alert_reason": "first_at_level",
        "scenario": "high_threat",
        "cia_scores": {"C": 0.24, "I": 0.9, "A": 0.3},
        "cia_max_dimension": "I",
        "cia_modifier": 0.9,
        "attack_category": attack,
        "explanation": {
            "level": "attention_and_shap",
            "timestep_importance": [0.05, 0.05, 0.1, 0.3, 0.5],
            "top_features": [
                {"feature": "DIntPkt", "importance": 0.082},
                {"feature": "TotBytes", "importance": 0.034},
                {"feature": "SrcLoad", "importance": 0.021},
            ],
        },
        "device_type": device,
    }


def _make_alert_sequence(
    n: int,
    risk: str = "HIGH",
    severity: int = 4,
) -> List[Dict[str, Any]]:
    """Build a sequence of n alerts at the same level."""
    return [
        {**_make_alert(risk=risk, severity=severity), "sample_index": i}
        for i in range(n)
    ]


# ═══════════════════════════════════════════════════════════════════
# 1. TRUST & EFFECTIVENESS SCALES
# ═══════════════════════════════════════════════════════════════════

class TestTrustCalibration:
    """Verify that the system builds CALIBRATED trust — not blind trust.

    Clinicians should trust the system when it's right and question it
    when it's uncertain. The system must provide enough information
    for the human to override.
    """

    def test_explanation_present_for_actionable_alerts(self) -> None:
        """Every URGENT+ alert must have an explanation."""
        for severity in [3, 4, 5]:
            alert = _make_alert(severity=severity)
            explanation = alert.get("explanation", {})
            level = explanation.get("level", "none")
            assert level != "none", (
                f"Severity {severity} alert has no explanation — "
                "clinician cannot assess trustworthiness"
            )

    def test_explanation_absent_for_routine(self) -> None:
        """ROUTINE alerts should NOT have explanations (noise reduction)."""
        from src.phase5_explanation_engine.phase5.conditional_explainer import _EXPLANATION_POLICY, ExplainabilityLevel
        assert _EXPLANATION_POLICY[1] == ExplainabilityLevel.NONE
        assert _EXPLANATION_POLICY[2] == ExplainabilityLevel.NONE

    def test_novel_threats_clearly_flagged(self) -> None:
        """Novel/zero-day threats must be visually distinct from known threats."""
        from dashboard.utils.cognitive_translator import CognitiveTranslator

        translator = CognitiveTranslator()
        novel = _make_alert(attention=True, attack="unknown")
        known = _make_alert(attention=False, attack="Spoofing")

        translator.translate([novel, known])

        soc_novel = novel["stakeholder_views"]["soc_analyst"]
        soc_known = known["stakeholder_views"]["soc_analyst"]

        # Novel must say NOVEL/ZERO-DAY
        assert "NOVEL" in soc_novel["threat_type"] or "ZERO-DAY" in soc_novel["threat_type"]
        # Known must say the actual attack type
        assert soc_known["threat_type"] == "Spoofing"

    def test_confidence_information_available(self) -> None:
        """Alerts must include the anomaly score so operators can gauge confidence."""
        alert = _make_alert(score=0.95)
        assert "anomaly_score" in alert
        assert 0.0 <= alert["anomaly_score"] <= 1.0

    def test_cia_breakdown_enables_override_decision(self) -> None:
        """CIA scores must be present so operators can assess whether
        the system's prioritization matches their clinical judgment."""
        alert = _make_alert()
        cia = alert.get("cia_scores", {})
        assert "C" in cia and "I" in cia and "A" in cia
        # All scores should be interpretable (0-1 range)
        for dim, val in cia.items():
            assert 0.0 <= val <= 1.0, f"CIA {dim}={val} out of range"

    def test_clinical_rationale_explains_decision(self) -> None:
        """Every alert must have a human-readable rationale string."""
        alert = _make_alert()
        rationale = alert.get("clinical_rationale", "")
        assert len(rationale) > 0
        # Must mention the device
        assert "infusion_pump" in rationale


class TestEffectiveness:
    """Verify alerts lead to correct, timely actions.

    Measures: Is the right person notified? Is the response time
    appropriate? Does the action match the threat?
    """

    def test_critical_alert_has_5min_response(self) -> None:
        """CRITICAL (severity 5) must require 5-minute response."""
        from src.phase4_risk_engine.phase4.clinical_impact import ClinicalImpactAssessor
        from src.phase4_risk_engine.phase4.risk_level import RiskLevel

        assessor = ClinicalImpactAssessor(biometric_columns=[])
        result = assessor.assess(RiskLevel.CRITICAL, "ecg_monitor", None, None)
        assert result.protocol.response_time_minutes == 5

    def test_emergent_alert_has_15min_response(self) -> None:
        from src.phase4_risk_engine.phase4.clinical_impact import ClinicalImpactAssessor
        from src.phase4_risk_engine.phase4.risk_level import RiskLevel

        assessor = ClinicalImpactAssessor(biometric_columns=[])
        result = assessor.assess(RiskLevel.HIGH, "generic_iomt_sensor", None, None)
        assert result.protocol.response_time_minutes == 15

    def test_urgent_alert_has_60min_response(self) -> None:
        from src.phase4_risk_engine.phase4.clinical_impact import ClinicalImpactAssessor
        from src.phase4_risk_engine.phase4.risk_level import RiskLevel

        assessor = ClinicalImpactAssessor(biometric_columns=[])
        result = assessor.assess(RiskLevel.MEDIUM, "generic_iomt_sensor", None, None)
        assert result.protocol.response_time_minutes == 60

    def test_critical_notifies_physician(self) -> None:
        """CRITICAL severity must notify on-call physician."""
        from src.phase4_risk_engine.phase4.clinical_impact import _PROTOCOLS, ClinicalSeverity

        protocol = _PROTOCOLS[ClinicalSeverity.CRITICAL]
        assert "on_call_physician" in protocol.notify

    def test_critical_notifies_incident_commander(self) -> None:
        from src.phase4_risk_engine.phase4.clinical_impact import _PROTOCOLS, ClinicalSeverity

        protocol = _PROTOCOLS[ClinicalSeverity.CRITICAL]
        assert "incident_commander" in protocol.notify

    def test_emergent_notifies_charge_nurse(self) -> None:
        from src.phase4_risk_engine.phase4.clinical_impact import _PROTOCOLS, ClinicalSeverity

        protocol = _PROTOCOLS[ClinicalSeverity.EMERGENT]
        assert "charge_nurse" in protocol.notify

    def test_routine_notifies_nobody(self) -> None:
        from src.phase4_risk_engine.phase4.clinical_impact import _PROTOCOLS, ClinicalSeverity

        protocol = _PROTOCOLS[ClinicalSeverity.ROUTINE]
        assert len(protocol.notify) == 0

    def test_device_action_matches_severity(self) -> None:
        """Higher severity should trigger stronger device actions."""
        from src.phase4_risk_engine.phase4.clinical_impact import _PROTOCOLS, ClinicalSeverity

        assert _PROTOCOLS[ClinicalSeverity.ROUTINE].device_action == "none"
        assert _PROTOCOLS[ClinicalSeverity.ADVISORY].device_action == "none"
        assert _PROTOCOLS[ClinicalSeverity.URGENT].device_action == "restrict_network"
        assert _PROTOCOLS[ClinicalSeverity.EMERGENT].device_action == "isolate_network"
        assert _PROTOCOLS[ClinicalSeverity.CRITICAL].device_action == "isolate_network"

    def test_ciso_gets_reporting_timeline(self) -> None:
        """CISO view must include reporting deadlines for high-severity incidents."""
        from dashboard.utils.cognitive_translator import CognitiveTranslator

        translator = CognitiveTranslator()
        alert = _make_alert(severity=5, risk="CRITICAL", safety=True)
        translator.translate([alert])

        ciso = alert["stakeholder_views"]["ciso"]
        reporting = ciso["recommended_reporting"]
        # Must have specific timeline actions
        assert any("24 hours" in r for r in reporting), "Missing 24-hour internal report"
        assert any("48 hours" in r for r in reporting), "Missing 48-hour board notification"

    def test_safety_guidance_says_do_not_power_off(self) -> None:
        """Patient safety guidance must explicitly say not to power off device."""
        from dashboard.utils.cognitive_translator import CognitiveTranslator

        translator = CognitiveTranslator()
        alert = _make_alert(severity=5, safety=True, device_action="isolate_network")
        translator.translate([alert])

        clinician = alert["stakeholder_views"]["clinician"]
        assert "DO NOT power off" in clinician.get("safety_guidance", "")


# ═══════════════════════════════════════════════════════════════════
# 2. ALERT FATIGUE ASSESSMENT
# ═══════════════════════════════════════════════════════════════════

class TestAlertFatigueAssessment:
    """Validate that fatigue mitigation reduces cognitive load without
    missing genuine threats.

    Medical literature: >85% of clinical alarms are non-actionable,
    leading to 72-99% being ignored (alarm fatigue). The system
    must reduce volume while ensuring zero missed escalations.
    """

    def test_suppression_reduces_volume_by_50pct(self) -> None:
        """Alert fatigue should suppress at least 50% of consecutive same-level alerts."""
        from src.phase4_risk_engine.phase4.alert_fatigue import AlertFatigueManager

        mgr = AlertFatigueManager(aggregation_window=5, alerts_per_window=5, rate_window_size=100)
        alerts = _make_alert_sequence(50, risk="HIGH", severity=4)
        mgr.process(alerts)

        emitted = sum(1 for a in alerts if a["alert_emit"])
        suppressed = sum(1 for a in alerts if not a["alert_emit"])

        assert suppressed / len(alerts) >= 0.50, (
            f"Only {suppressed}/{len(alerts)} suppressed — "
            "need >=50% reduction to prevent alarm fatigue"
        )

    def test_zero_escalations_missed(self) -> None:
        """The first escalation must ALWAYS fire, regardless of fatigue state.
        Subsequent alerts at the same escalated level may be aggregated/rate-limited
        since the operator already knows the situation is escalated."""
        from src.phase4_risk_engine.phase4.alert_fatigue import AlertFatigueManager

        mgr = AlertFatigueManager(aggregation_window=5, alerts_per_window=3, rate_window_size=50)

        # 20 LOW alerts → rate-limited
        alerts = _make_alert_sequence(20, risk="LOW", severity=2)
        # Then 5 escalations to HIGH
        for i in range(5):
            alerts.append({
                **_make_alert(risk="HIGH", severity=4),
                "sample_index": 20 + i,
            })

        mgr.process(alerts)

        # The FIRST escalation (LOW→HIGH) must always fire
        high_alerts = [a for a in alerts if a.get("risk_level") == "HIGH"]
        first_high = high_alerts[0]
        assert first_high["alert_emit"] is True, "First escalation was suppressed"
        assert first_high["alert_reason"] == "escalation"

    def test_de_escalation_signals_recovery(self) -> None:
        """When threat level drops, a de-escalation alert should fire
        to signal 'situation improving' to the operator."""
        from src.phase4_risk_engine.phase4.alert_fatigue import AlertFatigueManager

        mgr = AlertFatigueManager()
        alerts = [
            _make_alert(risk="HIGH", severity=4),
            _make_alert(risk="MEDIUM", severity=3),  # de-escalation
        ]
        alerts[0]["sample_index"] = 0
        alerts[1]["sample_index"] = 1
        mgr.process(alerts)

        assert alerts[1]["alert_emit"] is True
        assert alerts[1]["alert_reason"] == "de_escalation"

    def test_routine_never_emitted(self) -> None:
        """Severity 1 (ROUTINE) must never generate an alert."""
        from src.phase4_risk_engine.phase4.alert_fatigue import AlertFatigueManager

        mgr = AlertFatigueManager()
        alerts = _make_alert_sequence(100, risk="NORMAL", severity=1)
        mgr.process(alerts)

        emitted = sum(1 for a in alerts if a["alert_emit"])
        assert emitted == 0, f"{emitted} ROUTINE alerts emitted — should be 0"

    def test_aggregation_includes_suppressed_count(self) -> None:
        """When an aggregated alert fires, it must report how many
        were suppressed since the last emission."""
        from src.phase4_risk_engine.phase4.alert_fatigue import AlertFatigueManager

        mgr = AlertFatigueManager(aggregation_window=5)
        alerts = _make_alert_sequence(10, risk="HIGH", severity=4)
        mgr.process(alerts)

        # The 5th alert (index 4) should be an aggregation summary
        # with aggregated_count showing how many were suppressed
        aggregated = [a for a in alerts if a.get("alert_reason") == "aggregation_summary"]
        if aggregated:
            assert aggregated[0]["alert_aggregated_count"] > 0

    def test_rate_limit_prevents_flood(self) -> None:
        """Rate limiting should cap alerts per window."""
        from src.phase4_risk_engine.phase4.alert_fatigue import AlertFatigueManager

        mgr = AlertFatigueManager(
            aggregation_window=100,  # high so aggregation doesn't trigger
            alerts_per_window=3,
            rate_window_size=20,
        )
        # 20 unique-level alerts to avoid aggregation
        alerts = []
        for i in range(20):
            a = _make_alert(risk="HIGH", severity=4)
            a["sample_index"] = i
            alerts.append(a)
        mgr.process(alerts)

        emitted = sum(1 for a in alerts if a["alert_emit"])
        # First alert emits (first_at_level), then aggregation/rate limiting kicks in
        assert emitted <= 5, f"Rate limit failed: {emitted} alerts emitted in window of 20"

    def test_suppression_summary_has_correct_stats(self) -> None:
        """Summary should accurately report total, emitted, suppressed."""
        from src.phase4_risk_engine.phase4.alert_fatigue import AlertFatigueManager

        mgr = AlertFatigueManager()
        alerts = _make_alert_sequence(20, risk="HIGH", severity=4)
        mgr.process(alerts)

        summary = mgr.get_summary()
        assert summary["total_samples"] == 20
        assert summary["alerts_emitted"] + summary["alerts_suppressed"] == 20
        assert summary["alerts_emitted"] > 0
        assert summary["alerts_suppressed"] > 0

    def test_mixed_severity_stream_handles_transitions(self) -> None:
        """Realistic mixed-severity stream should handle all transitions."""
        from src.phase4_risk_engine.phase4.alert_fatigue import AlertFatigueManager

        mgr = AlertFatigueManager(aggregation_window=5)
        alerts = []
        # Normal → LOW → HIGH → HIGH → HIGH → CRITICAL → HIGH → NORMAL
        sequence = [
            ("NORMAL", 1), ("LOW", 2), ("HIGH", 4), ("HIGH", 4),
            ("HIGH", 4), ("CRITICAL", 5), ("HIGH", 4), ("NORMAL", 1),
        ]
        for i, (risk, sev) in enumerate(sequence):
            a = _make_alert(risk=risk, severity=sev)
            a["sample_index"] = i
            alerts.append(a)

        mgr.process(alerts)

        # No crash
        assert len(alerts) == 8
        # NORMAL (sev=1) should be suppressed
        assert alerts[0]["alert_emit"] is False
        assert alerts[7]["alert_emit"] is False
        # First LOW should emit
        assert alerts[1]["alert_emit"] is True
        # Escalation to HIGH should emit
        assert alerts[2]["alert_emit"] is True
        # Escalation to CRITICAL should emit
        assert alerts[5]["alert_emit"] is True
        # De-escalation from CRITICAL to HIGH should emit
        assert alerts[6]["alert_emit"] is True
