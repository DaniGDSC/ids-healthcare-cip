"""Data models for XAI-IDS-Healthcare prototype.

Defines all dataclasses matching research_spec.yaml schema exactly.
Build order step 1: all shared dataclasses used by Components 1, 2, and 3.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


# ── Component 2 output ──────────────────────────────────────────────────

@dataclass
class ScoredAlert:
    """Output of the Risk-Adaptive Scoring Engine (Component 2).

    Fields match research_spec.yaml component_2.output exactly.
    """

    adjusted_score: float
    """Anomaly score after risk multiplier applied."""

    threshold: float
    """Surfacing threshold for this device context."""

    should_surface: bool
    """True when adjusted_score > threshold."""

    risk_multiplier: float
    """Multiplier applied (>1.0 elevated risk, <1.0 suppressed)."""

    suppression_reason: Optional[str] = None
    """If suppressed, the human-readable reason."""


# ── Component 1 output ──────────────────────────────────────────────────

@dataclass
class MVEOutput:
    """Output of the MVE Generator (Component 1).

    Three-layer Minimum Viable Explanation:
      layer_1 — WHY anomalous: baseline vs deviation
      layer_2 — CLINICAL SEVERITY: patient-care impact
      layer_3 — RECOMMENDED ACTION: specific step + DO NOT + escalation

    Total output <= 150 words. No SHAP values. No CVSS.

    Layer fields are plain dicts so acceptance tests can call .get()
    directly, matching the pseudo-code in research_claims.yaml.
    """

    layer_1: dict
    """Keys: baseline_behavior, deviation_description, confidence_indicator.
    Max 60 words combined."""

    layer_2: dict
    """Keys: affected_system, patient_care_impact, phi_exposure,
    severity_label, severity_rationale. Max 50 words combined."""

    layer_3: dict
    """Keys: immediate_action, clinical_constraint, escalation_path,
    timeframe. Max 60 words combined."""

    alert_involves_clinical_system: bool = True
    """True for CRITICAL/HIGH/MEDIUM severity (clinical care systems).
    False for LOW (administrative/guest network). Controls DO NOT test."""

    # Field lists for word-count helpers
    _L1 = ["baseline_behavior", "deviation_description", "confidence_indicator"]
    _L2 = ["affected_system", "patient_care_impact", "phi_exposure",
            "severity_label", "severity_rationale"]
    _L3 = ["immediate_action", "clinical_constraint", "escalation_path", "timeframe"]

    @property
    def total_word_count(self) -> int:
        """Total word count across all 3 layers."""
        total = 0
        for fields, layer in [
            (self._L1, self.layer_1),
            (self._L2, self.layer_2),
            (self._L3, self.layer_3),
        ]:
            for f in fields:
                total += len(layer.get(f, "").split())
        return total

    def to_dict(self, alert_id: str = "") -> dict:
        """Flat dict for negative tests and alignment report.

        Includes alert_id so negative tests can cite the failing alert.
        Concatenates layer_1 fields into layer_1_why_anomalous for the
        test_no_rf_protocol_claims and test_no_model_internals_exposed checks.
        """
        return {
            "alert_id": alert_id,
            "layer_1_why_anomalous": " ".join(
                self.layer_1.get(f, "") for f in self._L1
            ),
            "layer_1": dict(self.layer_1),
            "layer_2": dict(self.layer_2),
            "layer_3": dict(self.layer_3),
            "total_word_count": self.total_word_count,
            "alert_involves_clinical_system": self.alert_involves_clinical_system,
        }


# ── Ground truth and pipeline record ───────────────────────────────────

@dataclass
class AlertGroundTruth:
    """Expert-assigned labels for a single alert (from fixtures)."""

    alert_id: str
    true_severity: str
    """CRITICAL / HIGH / MEDIUM / LOW (expert consensus)."""

    true_clinical_system: str
    """e.g., 'infusion pump network', 'EHR system', 'PACS archive'."""

    true_label: str
    """true_positive / false_positive / legitimate_rare."""

    device_patchable: bool
    device_criticality: str
    """CRITICAL / HIGH / MEDIUM / LOW."""


@dataclass
class AlertRecord:
    """Complete pipeline record for one alert."""

    alert_id: str
    raw_alert: dict
    device_context: dict
    behavioral_baseline: dict
    user_context: Optional[dict]
    ground_truth: AlertGroundTruth
    anomaly_score: float
    event_context: Optional[dict] = None
    scored: Optional[ScoredAlert] = None
    mve: Optional[MVEOutput] = None


# ── Harness output ──────────────────────────────────────────────────────

@dataclass
class TestReport:
    """Output of the Alert Simulation Harness (Component 3)."""

    metrics: List[dict]
    """One dict per automated test:
    {metric_id, metric_name, result_value, target, minimum, pass_fail, detail}."""

    negative_tests: List[dict]
    """One dict per negative test:
    {test_name, violations_found, pass_fail, violations}."""

    alignment: List[dict]
    """Claim-to-test mapping:
    {claim_id, claim_text, supported_by, all_tests_pass, verdict}."""
