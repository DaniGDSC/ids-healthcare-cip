"""Adaptive response selection + per-alert audit record builder."""
from __future__ import annotations

import hashlib
from datetime import datetime, timedelta

from .config import (
    ACTION_CATALOGUE,
    ATTACK_ROUTING,
    DEFAULT_DEVICE_TIER,
    DEFAULT_ROUTING,
    DEVICE_TIERS,
    TIER_POLICIES,
)


def select_adaptive_response(
    risk_level: str,
    risk_score: float,
    attack_category: str,
    device_tier: str = DEFAULT_DEVICE_TIER,
    biometric_in_top_features: bool = False,
) -> dict:
    """Select proportional response adapting to context beyond risk level.

    Pulls from the unified :mod:`module5_responses.config` taxonomy.
    """
    base = TIER_POLICIES.get(risk_level, TIER_POLICIES["NORMAL"])
    actions = list(base["default_actions"])
    rationale_parts = [f"Base response for {risk_level} risk level"]

    # 1. Magnitude scaling
    if risk_score >= 0.70 and risk_level != "CRITICAL":
        if "isolate_device" not in actions:
            actions.append("isolate_device")
        if "forensic_snapshot" not in actions:
            actions.append("forensic_snapshot")
        rationale_parts.append(
            f"Escalated: R={risk_score:.2f} exceeds 0.70 magnitude threshold"
        )
    elif risk_score < 0.30 and risk_level in ("MEDIUM", "HIGH"):
        if "isolate_device" in actions:
            actions.remove("isolate_device")
            if "restrict_traffic" not in actions:
                actions.append("restrict_traffic")
            rationale_parts.append(
                f"Demoted: R={risk_score:.2f} below 0.30, restrict instead of isolate"
            )

    # 2. Attack-category-specific actions
    routing = ATTACK_ROUTING.get(attack_category, DEFAULT_ROUTING)
    for action in routing["attack_specific_actions"]:
        if action not in actions:
            actions.append(action)
    if routing["attack_specific_actions"]:
        rationale_parts.append(
            f"Attack-specific ({attack_category}): added {routing['attack_specific_actions']}"
        )

    # 3. Device constraints
    tier_info = DEVICE_TIERS.get(device_tier, DEVICE_TIERS["vital_monitoring"])
    max_action_cost = ACTION_CATALOGUE[tier_info["max_action"]]["cost"]
    constrained_actions = []
    device_note = None
    for a in actions:
        if ACTION_CATALOGUE[a]["cost"] <= max_action_cost:
            constrained_actions.append(a)
        else:
            if tier_info["max_action"] not in constrained_actions:
                constrained_actions.append(tier_info["max_action"])
            device_note = (
                f"Device constraint ({device_tier}): {a} downgraded to "
                f"{tier_info['max_action']} — {tier_info['examples']}"
            )
    if device_note:
        rationale_parts.append(device_note)
    if tier_info["fallback_required"] and "isolate_device" in constrained_actions:
        rationale_parts.append("Fallback monitoring required before isolation")
    actions = constrained_actions

    # 4. Clinical escalation for biometric-involved alerts
    if biometric_in_top_features and "escalate_clinical" not in actions:
        actions.append("escalate_clinical")
        rationale_parts.append(
            "Biometric features in top SHAP contributors — clinical escalation added"
        )

    if "log_event" not in actions:
        actions.insert(0, "log_event")

    actions = sorted(set(actions), key=lambda a: ACTION_CATALOGUE[a]["cost"])

    # Phase 1.3 — surface per-action operational metadata so downstream
    # views (clinician summary, admin dashboard) can render reversibility
    # / cost / disruption badges without re-walking ACTION_CATALOGUE.
    actions_metadata = [
        {
            "name": a,
            "cost": float(ACTION_CATALOGUE[a]["cost"]),
            "reversible": bool(ACTION_CATALOGUE[a]["reversible"]),
            "requires_approval": bool(ACTION_CATALOGUE[a]["requires_approval"]),
            "expected_disruption": ACTION_CATALOGUE[a].get("expected_disruption", ""),
        }
        for a in actions
    ]

    return {
        "actions": actions,
        "action_descriptions": [ACTION_CATALOGUE[a]["description"] for a in actions],
        "actions_metadata": actions_metadata,
        "escalation_chain": {
            "primary": routing["primary"],
            "secondary": routing["secondary"],
            "tertiary": routing["tertiary"],
        },
        "escalation_rationale": routing["rationale"],
        "max_response_min": base["max_response_min"],
        "priority": base["priority"],
        "rationale": "; ".join(rationale_parts),
        "device_tier": device_tier,
        "device_constraint_applied": device_note is not None,
    }


def build_audit_record(
    idx: int,
    risk_score: float,
    risk_level: str,
    attack_category: str,
    ground_truth: str,
    response: dict,
    explanation_summary: str,
) -> dict:
    """Build FDA-style audit record with simulated outcome."""
    timestamp = datetime(2026, 4, 3, 12, 0, 0) + timedelta(seconds=idx)

    has_isolate = (
        "isolate_device" in response["actions"]
        or "restrict_traffic" in response["actions"]
    )
    is_true_attack = ground_truth == "attack"

    if is_true_attack and has_isolate:
        sim_outcome = "threat_contained"
        sim_effective = True
        sim_tte_sec = int(response["max_response_min"] * 60 * 0.6)
    elif is_true_attack and not has_isolate:
        sim_outcome = "threat_logged_not_mitigated"
        sim_effective = False
        sim_tte_sec = None
    elif not is_true_attack and has_isolate:
        sim_outcome = "false_positive_isolated"
        sim_effective = False
        sim_tte_sec = int(response["max_response_min"] * 60 * 0.3)
    else:
        sim_outcome = "benign_logged"
        sim_effective = True
        sim_tte_sec = None

    # Path B · commit 5 — per-record fingerprint, renamed from
    # ``integrity_hash`` to ``record_fingerprint``. The 16-char digest
    # is a per-row content checksum (not a chained signed hash); the
    # previous name collided with the 64-char ECDSA-signed chained hash
    # in ``audit_log.jsonl`` which keeps the ``integrity_hash`` name.
    # M5-2: f-string avoids dict construction + json.dumps + encode per record.
    record_fingerprint = hashlib.sha256(
        f"{idx}:{risk_score:.4f}:{risk_level}:{sim_outcome}".encode()
    ).hexdigest()[:16]

    return {
        "alert_id": f"ALERT-{idx:05d}",
        "timestamp": timestamp.isoformat(),
        "device_tier": response["device_tier"],
        "attack_category": attack_category,
        "risk_score": round(risk_score, 4),
        "risk_level": risk_level,
        "recommended_actions": response["actions"],
        "action_rationale": response["rationale"],
        "escalation_chain": response["escalation_chain"],
        "explanation_summary": explanation_summary[:200] if explanation_summary else "",
        "simulated_outcome": {
            "outcome": sim_outcome,
            "action_effective": sim_effective,
            "time_to_effectiveness_sec": sim_tte_sec,
            "ground_truth": ground_truth,
        },
        "record_fingerprint": record_fingerprint,
    }


__all__ = ["select_adaptive_response", "build_audit_record"]
