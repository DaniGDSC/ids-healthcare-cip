"""Layer 6 v4.0 — tier-recommendation routing table for the 9-class
:class:`AlertType` typology.

This module is pure function: given an alert type, a confidence level,
and a few optional context flags (after-hours, clinical-active), it
returns a :class:`TierRecommendationV4` describing the recommended
tier, the rationale, hospital-realistic fallbacks, and escalation
options. It performs no I/O, no execution, and no logging.

It DOES NOT replace the existing
``module5_responses/module5_pipeline.py::PolicyEngine`` — that engine
is keyed by severity tier (LOW/MEDIUM/HIGH/CRITICAL) and remains the
source of truth for the recommended *action set*. The v4 routing table
adds a complementary mapping keyed by the *alert type* so the
operator-facing tier recommendation can differentiate, e.g.,
``DISAGREEMENT_ANOMALY`` (route to security specialist) from
``CONFIRMED_ANOMALY`` (route to senior IT) even when both produce a
HIGH-severity composite risk score.

Two policy YAMLs feed the recommendation:

* ``configs/tier_routing.yaml``        — routing rules per fusion_class
                                         × risk_tier (Step [14]).
* ``configs/hospital_capabilities.yaml`` — deployment sizing + tier
                                         availability fallbacks
                                         (Step [14]).

Both are loaded lazily via :func:`load_tier_routing_yaml` and
:func:`load_hospital_capabilities`. The in-source ``_ROUTING`` table
remains the canonical source for the 9-class AlertType mapping; the
YAMLs add the fusion-class × risk-tier overlay on top.
"""
from __future__ import annotations

import functools
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from src.data_models import AlertType, Confidence


# ── Configuration loaders (ARCHITECTURE.md Step [14]) ─────────────────


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_TIER_ROUTING_YAML = _PROJECT_ROOT / "configs" / "tier_routing.yaml"
_HOSPITAL_CAPABILITIES_YAML = _PROJECT_ROOT / "configs" / "hospital_capabilities.yaml"


@functools.lru_cache(maxsize=1)
def load_tier_routing_yaml() -> dict[str, Any]:
    """Cached load of ``configs/tier_routing.yaml``.

    Returns ``{"routing_rules": [...]}`` or ``{}`` when the YAML is
    absent. The dict is the source of truth for the fusion_class ×
    risk_tier overlay; downstream callers iterate ``routing_rules`` in
    order and pick the first matching condition.
    """
    if not _TIER_ROUTING_YAML.exists():
        return {}
    import yaml
    body = yaml.safe_load(_TIER_ROUTING_YAML.read_text(encoding="utf-8")) or {}
    return body if isinstance(body, dict) else {}


@functools.lru_cache(maxsize=1)
def load_hospital_capabilities() -> dict[str, Any]:
    """Cached load of ``configs/hospital_capabilities.yaml``.

    Returns the deployment-sizing dict (``deployment_size``,
    ``available_tiers``, ``presets``, ``fallback_routing``) or ``{}``
    when the YAML is absent. Used by the M6 dashboard to grey out
    unavailable tiers and surface the doc-mandated small-hospital
    fallback ("document_for_external_consultant_review")."""
    if not _HOSPITAL_CAPABILITIES_YAML.exists():
        return {}
    import yaml
    body = yaml.safe_load(_HOSPITAL_CAPABILITIES_YAML.read_text(encoding="utf-8")) or {}
    return body if isinstance(body, dict) else {}


def get_available_tiers() -> list[str]:
    """Return the list of tiers staffed at this deployment.

    Reads ``deployment_size`` from ``configs/hospital_capabilities.yaml``
    and pulls the matching preset's ``available_tiers``. Falls back to
    a permissive list (``["L1", "L2_specialist", "senior_engineer"]``)
    when the YAML is absent.
    """
    cfg = load_hospital_capabilities()
    presets = (cfg.get("presets") or {})
    size = cfg.get("deployment_size", "medium")
    preset = presets.get(size) or {}
    avail = preset.get("available_tiers") or cfg.get("available_tiers")
    if avail:
        return list(avail)
    return ["L1", "L2_specialist", "senior_engineer"]


class TierLevel(str, Enum):
    """v4.0 tier-recommendation levels.

    These are *recommendations*, not enforcement targets — INVARIANT 3
    guarantees nothing in this codebase auto-routes alerts to a queue.
    The dashboard surfaces the recommended tier and the operator
    decides what to do.
    """

    L1_IMMEDIATE = "L1_immediate"
    L1_WITH_REVIEW = "L1_with_review"
    L1 = "L1"
    L1_WITH_SENIOR = "L1_with_senior"
    L2_SPECIALIST = "L2_specialist"
    L2_SECURITY_SPECIALIST = "L2_security_specialist"
    AUDIT_LOG = "AUDIT_LOG"
    SUPPRESSED = "SUPPRESSED"


@dataclass
class TierRecommendationV4:
    """Output of :func:`recommend_tier_v4`."""
    recommended_tier: TierLevel
    rationale: str
    fallback_options: list[str] = field(default_factory=list)
    escalation_options: list[str] = field(default_factory=list)
    requires_immediate_attention: bool = False
    adversarial_flag: bool = False
    requires_security_specialist: bool = False
    confidence_indicator: str = ""


# ── Base routing table per AlertType ────────────────────────────────────

_ROUTING: dict[AlertType, dict] = {
    AlertType.KNOWN_ATTACK: {
        "tier": TierLevel.L1_IMMEDIATE,
        "rationale": "Established attack signature with high confidence — immediate L1 response.",
        "fallback": ["Best-effort L1 response", "Document for forensics"],
        "escalation": ["L2 specialist", "Senior IT", "External SOC"],
        "requires_immediate": True,
    },
    AlertType.KNOWN_ATTACK_UNCERTAIN: {
        "tier": TierLevel.L1_WITH_REVIEW,
        "rationale": "High Track A probability but ensemble disagreement — L1 with peer review.",
        "fallback": ["L1 documents and pages senior", "Peer review at handoff"],
        "escalation": ["Senior IT", "L2 specialist"],
        "requires_immediate": True,
    },
    AlertType.DISAGREEMENT_ANOMALY: {
        "tier": TierLevel.L2_SECURITY_SPECIALIST,
        "rationale": (
            "Model disagreement combined with DAE elevation — "
            "potential adversarial input. Security specialist required."
        ),
        "fallback": [
            "L1 documents and routes to security review queue",
            "External CISO consultation",
        ],
        "escalation": ["External security consultant", "CISO"],
        "requires_immediate": False,
        "adversarial_flag": True,
        "requires_security_specialist": True,
    },
    AlertType.STRONG_NOVEL_ANOMALY: {
        "tier": TierLevel.L2_SPECIALIST,
        "rationale": "Strong novelty signal in the silent regime — specialist investigation needed.",
        "fallback": ["L1 documents for L2 review", "Vendor support consultation"],
        "escalation": ["Senior IT", "External consultant", "CISO"],
        "requires_immediate": False,
    },
    AlertType.NOVEL_ANOMALY: {
        "tier": TierLevel.L2_SPECIALIST,
        "rationale": "Moderate novelty signal — specialist review.",
        "fallback": ["L1 documents for L2 review", "Vendor consultation"],
        "escalation": ["Senior IT", "External consultant"],
        "requires_immediate": False,
    },
    AlertType.CONFIRMED_ANOMALY: {
        "tier": TierLevel.L1_WITH_SENIOR,
        "rationale": "Multi-signal corroboration — senior input recommended.",
        "fallback": ["L1 best-effort with senior on-call"],
        "escalation": ["L2 specialist", "External consultant"],
        "requires_immediate": True,
    },
    AlertType.SUSPICIOUS_PATTERN: {
        "tier": TierLevel.L1,
        "rationale": "Track A moderate, DAE benign — standard L1 review.",
        "fallback": ["L1 standard handling"],
        "escalation": ["Senior IT", "Coordinate with biomed"],
        "requires_immediate": False,
    },
    AlertType.BENIGN_WATCH: {
        "tier": TierLevel.AUDIT_LOG,
        "rationale": "Marginal anomaly — logged for pattern analysis only.",
        "fallback": ["Audit log only"],
        "escalation": [],
        "requires_immediate": False,
    },
    AlertType.BENIGN: {
        "tier": TierLevel.SUPPRESSED,
        "rationale": "All filters indicate benign — no display.",
        "fallback": [],
        "escalation": [],
        "requires_immediate": False,
    },
}


def _adjust_for_confidence(base: TierLevel, confidence: Confidence) -> TierLevel:
    """LOW confidence demotes urgent tiers; everything else stays."""
    if confidence == Confidence.LOW:
        if base == TierLevel.L1_IMMEDIATE:
            return TierLevel.L1
        if base == TierLevel.L1:
            return TierLevel.AUDIT_LOG
    return base


def _hospital_fallbacks(
    base: list[str],
    *,
    is_after_hours: bool,
    clinical_active: bool,
) -> list[str]:
    out = list(base)
    if is_after_hours:
        out.append("On-call rotation activation")
    if clinical_active:
        out.insert(0, "Coordinate with clinical staff first (active care)")
    return out


def recommend_tier_v4(
    alert_type: AlertType | str,
    confidence: Confidence | str = Confidence.MEDIUM,
    *,
    is_after_hours: bool = False,
    clinical_active: bool = False,
) -> TierRecommendationV4:
    """Return the v4 tier recommendation for an alert.

    Total over the v4 typology — an unrecognised alert-type string
    falls through to the ``NOVEL_ANOMALY`` policy (the most cautious
    "investigate" route) so the dashboard never sees a missing tier.

    Args:
        alert_type: One of :class:`AlertType` or its string value.
        confidence: One of :class:`Confidence`. Defaults to MEDIUM if
            the caller omits it; LOW demotes urgent tiers.
        is_after_hours: Adds an on-call rotation fallback option.
        clinical_active: Prepends a "coordinate with clinical staff
            first" fallback so the operator knows to talk to the unit
            before any device action.

    Returns:
        :class:`TierRecommendationV4` ready to be surfaced on the
        alert card. Every field is populated; the lists may be empty
        for BENIGN/BENIGN_WATCH.
    """
    if isinstance(alert_type, str):
        try:
            alert_type = AlertType(alert_type)
        except ValueError:
            alert_type = AlertType.NOVEL_ANOMALY
    if isinstance(confidence, str):
        try:
            confidence = Confidence(confidence)
        except ValueError:
            confidence = Confidence.MEDIUM

    routing = _ROUTING[alert_type]
    tier = _adjust_for_confidence(routing["tier"], confidence)
    fallbacks = _hospital_fallbacks(
        routing["fallback"],
        is_after_hours=is_after_hours,
        clinical_active=clinical_active,
    )
    return TierRecommendationV4(
        recommended_tier=tier,
        rationale=routing["rationale"],
        fallback_options=fallbacks,
        escalation_options=list(routing["escalation"]),
        requires_immediate_attention=routing.get("requires_immediate", False),
        adversarial_flag=routing.get("adversarial_flag", False),
        requires_security_specialist=routing.get("requires_security_specialist", False),
        confidence_indicator=confidence.value,
    )


# ── Operator-followed-recommendation tracking ──────────────────────────


def operator_followed_recommendation(
    operator_action: str,
    recommended_actions: list[str],
    *,
    case_insensitive: bool = True,
) -> bool:
    """Decision-quality helper: did the operator pick an action that
    matches one of the system's recommendations?

    Used by Layer 6 to populate the ``operator_followed_recommendation``
    boolean on each :class:`OperatorDecision` audit record so the RQ3
    evaluation can compute ``followed_recommendation_pct``.

    Returns False on empty/whitespace-only operator actions and on
    placeholder selections like ``"— Select action —"`` so the metric
    isn't inflated by no-ops.

    Args:
        operator_action: The label the operator picked from the
            decision form (Layer 5).
        recommended_actions: All actions the system recommended for
            the alert + role. The operator only needs to match ONE.
        case_insensitive: Match case-insensitively (default).
    """
    if not operator_action or not operator_action.strip():
        return False
    op = operator_action.strip()
    if op.startswith("—") or op.startswith("--"):
        return False
    if case_insensitive:
        op = op.lower()
        recs = [r.strip().lower() for r in recommended_actions if r and r.strip()]
    else:
        recs = [r.strip() for r in recommended_actions if r and r.strip()]
    return op in recs


__all__ = [
    "TierLevel",
    "TierRecommendationV4",
    "recommend_tier_v4",
    "operator_followed_recommendation",
]
