"""Module 5 — unified response-policy taxonomy.

Single source of truth for the closed-loop response engine. Replaces the
two parallel taxonomies that previously lived side-by-side in
``module5_pipeline.RESPONSE_POLICY`` and ``module5_responses.{MITIGATION_ACTIONS,
DEVICE_TIERS, BASE_PROTOCOL, ESCALATION_ROUTING}``.

Schema bumped 1.0 → 2.0 to flag the unified shape. The old top-level
``RESPONSE_POLICY`` dict shape is reconstructed by
:func:`export_response_policy_dict` for the legacy ``response_policy.json``
artifact so external readers don't break.
"""
from __future__ import annotations

import os

RESPONSE_POLICY_VERSION = "2.0"

# ── Mitigation action catalogue ────────────────────────────────────────
# Unifies the two prior shapes:
#   MITIGATION_ACTIONS (responses): severity_floor + cost + reversible + description
#   RESPONSE_POLICY.action_catalogue (pipeline): cost + reversible + requires_approval
# All four fields are kept per action so neither engine loses information.

ACTION_CATALOGUE: dict[str, dict] = {
    "log_event": {
        "severity_floor": "LOW",
        "cost": 0.1,
        "reversible": True,
        "requires_approval": False,
        "description": "Log event to SIEM for audit trail",
    },
    "enhanced_monitoring": {
        "severity_floor": "LOW",
        "cost": 0.2,
        "reversible": True,
        "requires_approval": False,
        "description": "Enable enhanced logging and monitoring on device",
    },
    "re_authenticate": {
        "severity_floor": "MEDIUM",
        "cost": 0.3,
        "reversible": True,
        "requires_approval": False,
        "description": "Force device re-authentication and credential verification",
    },
    "forensic_snapshot": {
        "severity_floor": "HIGH",
        "cost": 0.4,
        "reversible": True,
        "requires_approval": False,
        "description": "Capture full packet capture and device state for forensics",
    },
    "restrict_traffic": {
        "severity_floor": "MEDIUM",
        "cost": 0.5,
        "reversible": True,
        "requires_approval": False,
        "description": "Restrict device to essential clinical traffic only (whitelist mode)",
    },
    "escalate_clinical": {
        "severity_floor": "HIGH",
        "cost": 0.7,
        "reversible": False,
        "requires_approval": False,
        "description": "Escalate to clinical staff — verify patient vitals independently",
    },
    "isolate_device": {
        "severity_floor": "HIGH",
        "cost": 0.8,
        "reversible": True,
        "requires_approval": True,
        "description": "Isolate device from network segment via VLAN quarantine",
    },
    "escalate_incident": {
        "severity_floor": "CRITICAL",
        "cost": 1.0,
        "reversible": False,
        "requires_approval": False,
        "description": "Initiate full incident response — page CISO + on-call physician",
    },
}

# ── Device constraint tiers ────────────────────────────────────────────
# Unifies the two prior shapes:
#   DEVICE_TIERS (responses): max_action, fallback_required,
#       clinical_escalation_mandatory, examples
#   RESPONSE_POLICY.device_constraints (pipeline): max_action_cost,
#       isolation_blocked, clinical_approval_required
# `clinical_escalation_mandatory` is renamed `clinical_approval_required`
# (the pipeline name) since it is the one used by external consumers.
# `max_action_cost` is derived from ACTION_CATALOGUE so the two stay in
# lockstep.

def _cost_of(action: str) -> float:
    return ACTION_CATALOGUE[action]["cost"]


DEVICE_TIERS: dict[str, dict] = {
    "life_sustaining": {
        "max_action": "restrict_traffic",
        "max_action_cost": _cost_of("restrict_traffic"),
        "isolation_blocked": True,
        "fallback_required": True,
        "clinical_approval_required": True,
        "examples": "infusion pump, ventilator",
    },
    "vital_monitoring": {
        "max_action": "isolate_device",
        "max_action_cost": _cost_of("isolate_device"),
        "isolation_blocked": False,
        "fallback_required": True,
        "clinical_approval_required": True,
        "examples": "ECG monitor, pulse oximeter",
    },
    "diagnostic": {
        "max_action": "isolate_device",
        "max_action_cost": _cost_of("isolate_device"),
        "isolation_blocked": False,
        "fallback_required": False,
        "clinical_approval_required": False,
        "examples": "blood pressure monitor, thermometer",
    },
    "auxiliary": {
        "max_action": "isolate_device",
        "max_action_cost": _cost_of("isolate_device"),
        "isolation_blocked": False,
        "fallback_required": False,
        "clinical_approval_required": False,
        "examples": "environmental sensor, room monitor",
    },
}
DEFAULT_DEVICE_TIER = "vital_monitoring"

# ── Tier policies ──────────────────────────────────────────────────────
# Unifies:
#   tier_policies (pipeline): default_actions + max_response_min + auto_execute
#   BASE_PROTOCOL (responses):  priority + base_actions + max_response_min
# Canonicalises action order so `log_event` is always first.

TIER_POLICIES: dict[str, dict] = {
    "CRITICAL": {
        "priority": 1,
        "default_actions": [
            "log_event",
            "isolate_device",
            "forensic_snapshot",
            "escalate_incident",
            "escalate_clinical",
        ],
        "max_response_min": 5,
        "auto_execute": True,
    },
    "HIGH": {
        "priority": 2,
        "default_actions": [
            "log_event",
            "isolate_device",
            "forensic_snapshot",
            "enhanced_monitoring",
        ],
        "max_response_min": 15,
        "auto_execute": True,
    },
    "MEDIUM": {
        "priority": 3,
        "default_actions": [
            "log_event",
            "restrict_traffic",
            "enhanced_monitoring",
        ],
        "max_response_min": 60,
        "auto_execute": False,
    },
    "LOW": {
        "priority": 4,
        "default_actions": ["log_event", "enhanced_monitoring"],
        "max_response_min": 480,
        "auto_execute": False,
    },
    "NORMAL": {
        "priority": 5,
        "default_actions": ["log_event"],
        "max_response_min": 0,
        "auto_execute": False,
    },
}

# ── Attack-category routing ────────────────────────────────────────────
# Unifies ATTACK_ROUTING (pipeline: add_actions + primary_notify +
# secondary_notify) and ESCALATION_ROUTING (responses: primary +
# secondary + tertiary + rationale + attack_specific_actions). Single
# dict carries all fields; legacy ``primary_notify`` / ``secondary_notify``
# names are dropped in favour of ``primary`` / ``secondary``.

ATTACK_ROUTING: dict[str, dict] = {
    "Spoofing": {
        "primary": "IT Security",
        "secondary": "Biomedical Engineering",
        "tertiary": None,
        "rationale": (
            "Spoofing targets device identity — biomed must verify "
            "physical device integrity"
        ),
        "attack_specific_actions": ["re_authenticate", "restrict_traffic"],
        "add_actions": ["re_authenticate"],
    },
    "Data Alteration": {
        "primary": "IT Security",
        "secondary": "Charge Nurse",
        "tertiary": "On-call Physician",
        "rationale": (
            "Data alteration may corrupt biometric readings — clinical "
            "verification required"
        ),
        "attack_specific_actions": [
            "isolate_device",
            "forensic_snapshot",
            "escalate_clinical",
        ],
        "add_actions": ["forensic_snapshot", "escalate_clinical"],
    },
    "normal": {
        "primary": None,
        "secondary": None,
        "tertiary": None,
        "rationale": "No attack detected",
        "attack_specific_actions": [],
        "add_actions": [],
    },
}

DEFAULT_ROUTING: dict = {
    "primary": "IT Security",
    "secondary": "Incident Commander",
    "tertiary": None,
    "rationale": (
        "Unknown attack type — follow general incident response protocol"
    ),
    "attack_specific_actions": ["restrict_traffic", "forensic_snapshot"],
    "add_actions": [],
}

# ── Acuity overrides ───────────────────────────────────────────────────

ACUITY_OVERRIDES: dict = {
    "elevated_acuity_threshold": 0.25,
    "action_on_elevated": (
        "Add escalate_clinical if not present; require clinical "
        "confirmation before isolation"
    ),
}

# ── MVE LLM tripwire ───────────────────────────────────────────────────
# Y8 fix: was a magic literal inside build_all_records.
# Operator override via IOMT_MVE_TRIPWIRE env var.

MVE_LLM_FAIL_STREAK_MAX = int(os.environ.get("IOMT_MVE_TRIPWIRE", "5"))


# ── Legacy dict view (for back-compat artifact writer) ─────────────────


def export_response_policy_dict() -> dict:
    """Reconstruct the legacy RESPONSE_POLICY shape from the unified config.

    Used by ``policy.export_response_policy`` to write the historical
    ``response_policy.json`` artifact shape — external readers (dashboard,
    integration tests) still see the 1.x keys.
    """
    return {
        "version": RESPONSE_POLICY_VERSION,
        "description": (
            "Maps (alert_tier, device_tier, patient_acuity_level) to "
            "response action sets"
        ),
        "action_catalogue": {
            name: {
                "cost": spec["cost"],
                "reversible": spec["reversible"],
                "requires_approval": spec["requires_approval"],
            }
            for name, spec in ACTION_CATALOGUE.items()
        },
        "tier_policies": {
            tier: {
                "default_actions": list(p["default_actions"]),
                "max_response_min": p["max_response_min"],
                "auto_execute": p["auto_execute"],
            }
            for tier, p in TIER_POLICIES.items()
            if tier != "NORMAL"
        },
        "device_constraints": {
            tier: {
                "max_action_cost": spec["max_action_cost"],
                "isolation_blocked": spec["isolation_blocked"],
                "clinical_approval_required": spec["clinical_approval_required"],
            }
            for tier, spec in DEVICE_TIERS.items()
        },
        "acuity_overrides": dict(ACUITY_OVERRIDES),
        "attack_routing": {
            cat: {
                "add_actions": list(r["add_actions"]),
                "primary_notify": r["primary"],
                "secondary_notify": r["secondary"],
            }
            for cat, r in ATTACK_ROUTING.items()
            if cat != "normal"
        },
    }


__all__ = [
    "ACTION_CATALOGUE",
    "DEVICE_TIERS",
    "TIER_POLICIES",
    "ATTACK_ROUTING",
    "ACUITY_OVERRIDES",
    "DEFAULT_DEVICE_TIER",
    "DEFAULT_ROUTING",
    "MVE_LLM_FAIL_STREAK_MAX",
    "RESPONSE_POLICY_VERSION",
    "export_response_policy_dict",
]
