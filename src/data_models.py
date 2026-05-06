"""Data models for XAI-IDS-Healthcare prototype.

Defines all dataclasses matching research_spec.yaml schema exactly.
Build order step 1: all shared dataclasses used by Components 1, 2, and 3.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Optional


class FusionClass(str, Enum):
    """Two-stage fusion outcome per ARCHITECTURE.md Step [7].

    DISAGREEMENT_ANOMALY (Enhancement 4) is a special case appended at
    fusion time when the three Track A models disagree above a tunable
    threshold (``diversity_score`` >= ``b_diversity``). It overrides
    KNOWN_ATTACK only — uncertain ensembles never confidently fire — but
    leaves NOVEL_ANOMALY / CONFIRMED_ANOMALY routing untouched (DAE has
    already had its say there).
    """

    KNOWN_ATTACK = "KNOWN_ATTACK"
    CONFIRMED_ANOMALY = "CONFIRMED_ANOMALY"
    NOVEL_ANOMALY = "NOVEL_ANOMALY"
    DISAGREEMENT_ANOMALY = "DISAGREEMENT_ANOMALY"  # Enhancement 4
    BENIGN = "BENIGN"


class DataQuality(str, Enum):
    """Per-alert sanitization outcome per ARCHITECTURE.md Step [5].

    Severity ladder:
      OK          — input clean OR nan_rate <= 5% (rare, isolated NaN/Inf).
      IMPUTED_NAN — row-level marker used by Module-3 batch path; equivalent
                    to OK at alert-severity level (kept for back-compat).
      DEGRADED    — nan_rate > 5%; likely sensor/capture issue or
                    NaN-injection attempt (EA-06). Operator should treat the
                    explanation as fragile.
      FAILED      — nan_rate >= 50%; input is essentially garbage; route to
                    biomed for device check.
    """

    OK = "OK"
    IMPUTED_NAN = "IMPUTED_NAN"
    DEGRADED = "DEGRADED"
    FAILED = "FAILED"


class OperatorRole(str, Enum):
    """Per ARCHITECTURE.md Step [13] — role-scoped view rendering.

    Each role sees an MVE tailored to its authority and decision space:
      IT_GENERALIST — primary audience; default MVE wording (network actions)
      BIOMED_ENGINEER — device-facing wording (verify/document/coordinate)
      NURSE_MANAGER — clinical-impact wording (verify backup/monitor/document)

    Closes GAP-A2. Routing primary/secondary already exists in
    module5_pipeline.py; the per-role *view rendering* lands here.
    """

    IT_GENERALIST = "IT_generalist"
    BIOMED_ENGINEER = "biomed_engineer"
    NURSE_MANAGER = "nurse_manager"


P_XGB_HIGH_CONF: float = 0.85
"""Track A high-confidence boundary above which we classify as KNOWN_ATTACK
without requiring DAE confirmation."""


# ── Multi-class Track A (cascade-contract refactor) ─────────────────────

# EHMS-2020 attack categories ordered with "normal" pinned to index 0 so
# every multi-class consumer can derive P(attack) = 1 - softmax[:, 0].
# Order of attack classes is lex-sorted for determinism. Update this
# constant if the dataset's Attack Category vocabulary changes.
MULTICLASS_LABEL_ORDER_EHMS: tuple[str, ...] = (
    "normal",
    "Data Alteration",
    "Spoofing",
)

# MedSec-25 categories — used by the LOCO experiment, not the EHMS pipeline.
MULTICLASS_LABEL_ORDER_MEDSEC: tuple[str, ...] = (
    "Benign",
    "Exfiltration",
    "Initial access",
    "Lateral movement",
    "Reconnaissance",
)


def normal_index(label_order: tuple[str, ...]) -> int:
    """Return the index of the benign class in ``label_order``.

    Convention: "normal" (EHMS) and "Benign" (MedSec) both denote the
    benign class and must sit at index 0 — but we look it up by name
    rather than hard-coding 0 so a future label-order change does not
    silently invert the cascade contract.
    """
    for cand in ("normal", "Benign"):
        if cand in label_order:
            return label_order.index(cand)
    raise ValueError(
        f"label_order {label_order!r} contains no recognised benign class "
        f"(expected 'normal' or 'Benign')"
    )


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

    fusion_class: FusionClass = FusionClass.BENIGN
    """Two-stage fusion outcome from Track A + Track B."""

    data_quality: DataQuality = DataQuality.OK
    """Per-row sanitization outcome: OK or IMPUTED_NAN (NaN/Inf replaced)."""


# ── Module 4 output (SHAP explanations) ─────────────────────────────────

@dataclass
class SHAPContext:
    """SHAP context passed from module4_online_explainer to mve_generator."""

    top_category: str
    """Label of the feature group with highest |SHAP| sum.
    One of the 8 clinical groupings in research_spec.yaml §2.module_4:
    timing_pattern, network_destination, data_volume, device_behavior,
    biometric, user_access_pattern, lateral_movement, exfiltration_signal."""

    top_features: List[str]
    """Top 3 feature names by |SHAP| value (from the winning group)."""

    shap_direction: str
    """'elevated' if SHAP pushed score toward attack class,
    'suppressed' if SHAP pushed toward benign."""

    confidence_from_shap: str
    """HIGH if top feature |SHAP| > 0.3, MEDIUM if |SHAP| in [0.1, 0.3],
    LOW if |SHAP| < 0.1."""

    stability_score: float = 1.0
    """Mean pairwise Jaccard similarity of top-k SHAP features across N
    bootstrap perturbations of the input. 1.0 = perfectly stable; 0.0 =
    every perturbation flips the ranking. Values < STABILITY_LOW (0.5)
    should be surfaced to the operator as a fragility warning."""


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

    layer_1: dict[str, str]
    """Keys: baseline_behavior, deviation_description, confidence_indicator.
    Max 60 words combined."""

    layer_2: dict[str, str]
    """Keys: affected_system, patient_care_impact, phi_exposure,
    severity_label, severity_rationale. Max 50 words combined."""

    layer_3: dict[str, str]
    """Keys: immediate_action, clinical_constraint, escalation_path,
    timeframe. Max 60 words combined."""

    alert_involves_clinical_system: bool = True
    """True for CRITICAL/HIGH/MEDIUM severity (clinical care systems).
    False for LOW (administrative/guest network). Controls DO NOT test."""

    # Field lists for word-count helpers
    # role_authorization_check is optional (T2/EHR-access only — IMP-02);
    # counted toward L1 budget when present, absent keys count as zero words.
    _L1 = ["baseline_behavior", "deviation_description", "confidence_indicator",
           "role_authorization_check"]
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

    def to_dict(self, alert_id: str = "") -> dict[str, Any]:
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
    raw_alert: dict[str, Any]
    device_context: dict[str, Any]
    behavioral_baseline: dict[str, Any]
    user_context: Optional[dict[str, Any]]
    ground_truth: AlertGroundTruth
    anomaly_score: float
    event_context: Optional[dict[str, Any]] = None
    scored: Optional[ScoredAlert] = None
    mve: Optional[MVEOutput] = None


# ── Harness output ──────────────────────────────────────────────────────

@dataclass
class TestReport:
    """Output of the Alert Simulation Harness (Component 3)."""

    metrics: List[dict[str, Any]]
    """One dict per automated test:
    {metric_id, metric_name, result_value, target, minimum, pass_fail, detail}."""

    negative_tests: List[dict[str, Any]]
    """One dict per negative test:
    {test_name, violations_found, pass_fail, violations}."""

    alignment: List[dict[str, Any]]
    """Claim-to-test mapping:
    {claim_id, claim_text, supported_by, all_tests_pass, verdict}."""


# ── Operator-decision audit (Step [16], closes GAP-A5) ──────────────────


@dataclass
class OperatorDecision:
    """Schema for one row of the operator-decision audit log (Step [16]).

    Promoted from the loose dict in module6_evaluation/module6_app.py to a
    formal dataclass so audit-log consumers can rely on a fixed shape.
    Required fields are positional; optional fields default sensibly.
    """

    alert_id: str
    operator_role: str            # "IT_generalist" | "biomed_engineer" | "nurse_manager"
    operator_action_taken: str    # free-form action label
    decision_time_seconds: float
    timestamp: str                # ISO 8601

    alert_type: str = ""
    recommended_action: str = ""
    operator_confidence: Optional[int] = None     # 1-5 Likert
    operator_rationale: str = ""

    def validate(self) -> None:
        """Raise ValueError on schema violations."""
        for field in ("alert_id", "operator_role", "operator_action_taken",
                      "timestamp"):
            if not getattr(self, field):
                raise ValueError(f"OperatorDecision.{field} must be non-empty")
        if self.decision_time_seconds < 0:
            raise ValueError("decision_time_seconds must be non-negative")
        if self.operator_confidence is not None and not (
            1 <= self.operator_confidence <= 5
        ):
            raise ValueError("operator_confidence must be in [1, 5] when present")
