"""Module 5 — PolicyEngine + clinical safety override.

Replaces the old ``module5_pipeline.RESPONSE_POLICY``-driven engine; now
reads from the unified :mod:`module5_responses.config`.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from .config import (
    ACTION_CATALOGUE,
    ACUITY_OVERRIDES,
    ATTACK_ROUTING,
    DEVICE_TIERS,
    TIER_POLICIES,
    export_response_policy_dict,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "results/reports"


def export_response_policy(path: Path | None = None) -> Path:
    """Task 5.1: Export standalone response policy config.

    Writes the legacy 1.x dict shape so external readers (dashboard,
    integration tests) don't break with the 2.0 schema bump.
    """
    out = Path(path) if path else OUTPUT_DIR / "response_policy.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(export_response_policy_dict(), indent=2),
        encoding="utf-8",
    )
    logger.info("5.1 Saved: %s", out.name)
    return out


class PolicyEngine:
    """Rule-based engine: reads policy config, returns recommended actions."""

    def __init__(self, policy: dict | None = None):
        # Accept the legacy RESPONSE_POLICY shape for back-compat (callers
        # in tests sometimes inject a custom policy). When None, derive
        # from the unified config so the engine and the artifact agree.
        self.policy = policy if policy is not None else export_response_policy_dict()
        self.catalogue = self.policy.get("action_catalogue") or {
            n: {"cost": s["cost"], "reversible": s["reversible"],
                "requires_approval": s["requires_approval"]}
            for n, s in ACTION_CATALOGUE.items()
        }

    def recommend(
        self,
        alert_tier: str,
        device_tier: str = "vital_monitoring",
        attack_category: str = "unknown",
        patient_acuity: float = 0.0,
    ) -> dict:
        tier_policies = self.policy.get("tier_policies") or {
            t: {"default_actions": list(p["default_actions"]),
                "max_response_min": p["max_response_min"],
                "auto_execute": p["auto_execute"]}
            for t, p in TIER_POLICIES.items() if t != "NORMAL"
        }
        tier_policy = tier_policies.get(alert_tier, tier_policies["LOW"])
        actions = list(tier_policy["default_actions"])

        routing_table = self.policy.get("attack_routing") or {
            cat: {"add_actions": list(r["add_actions"]),
                  "primary_notify": r["primary"],
                  "secondary_notify": r["secondary"]}
            for cat, r in ATTACK_ROUTING.items() if cat != "normal"
        }
        routing = routing_table.get(attack_category, {})
        for a in routing.get("add_actions", []):
            if a not in actions:
                actions.append(a)

        device_constraints = self.policy.get("device_constraints") or {
            t: {"max_action_cost": s["max_action_cost"],
                "isolation_blocked": s["isolation_blocked"],
                "clinical_approval_required": s["clinical_approval_required"]}
            for t, s in DEVICE_TIERS.items()
        }
        constraint = device_constraints.get(device_tier, {})
        max_cost = constraint.get("max_action_cost", 1.0)
        if constraint.get("isolation_blocked") and "isolate_device" in actions:
            actions.remove("isolate_device")
            if "restrict_traffic" not in actions:
                actions.append("restrict_traffic")

        actions = [
            a for a in actions
            if self.catalogue.get(a, {}).get("cost", 0) <= max_cost
            or a in ("log_event", "escalate_clinical")
        ]

        override = clinical_safety_check(
            alert_tier, device_tier, patient_acuity, actions,
        )

        actions = sorted(
            set(actions), key=lambda a: self.catalogue.get(a, {}).get("cost", 0)
        )

        return {
            "actions": actions,
            "max_response_min": tier_policy["max_response_min"],
            "auto_execute": tier_policy["auto_execute"],
            "primary_notify": routing.get("primary_notify", "IT Security"),
            "secondary_notify": routing.get("secondary_notify"),
            "clinical_override": override,
            "requires_approval": any(
                self.catalogue.get(a, {}).get("requires_approval", False)
                for a in actions
            ),
        }


def clinical_safety_check(
    alert_tier: str,
    device_tier: str,
    patient_acuity: float,
    actions: list,
) -> dict:
    """Check if device is safety-critical AND patient acuity elevated → override.

    Mutates ``actions`` in-place when a downgrade fires (isolate → restrict)
    and returns a structured override descriptor for the audit trail.
    """
    override = {
        "triggered": False,
        "reason": None,
        "original_actions": list(actions),
        "clinical_confirmation_required": False,
    }

    is_critical_device = device_tier in ("life_sustaining", "vital_monitoring")
    acuity_elevated = patient_acuity >= ACUITY_OVERRIDES["elevated_acuity_threshold"]

    if is_critical_device and acuity_elevated:
        override["triggered"] = True
        override["clinical_confirmation_required"] = True

        if "isolate_device" in actions:
            override["reason"] = (
                f"Device tier '{device_tier}' with elevated patient acuity "
                f"({patient_acuity:.2f}) — isolation downgraded to restrict_traffic. "
                "Clinical confirmation required before any disruptive action."
            )
            actions.remove("isolate_device")
            if "restrict_traffic" not in actions:
                actions.append("restrict_traffic")
        else:
            override["reason"] = (
                f"Device tier '{device_tier}' with elevated acuity "
                f"({patient_acuity:.2f}) — clinical confirmation required."
            )

        if "escalate_clinical" not in actions:
            actions.append("escalate_clinical")

    return override


__all__ = ["PolicyEngine", "clinical_safety_check", "export_response_policy"]
