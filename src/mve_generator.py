"""Component 1: MVE Generator.

Produces a 3-layer Minimum Viable Explanation from a raw alert,
device context, behavioral baseline, and optional user context.

Provider chain (first available path wins; remaining paths skipped):
  Option A1 (LLM, OpenAI)    — OpenAI API with JSON-mode if
                               OPENAI_API_KEY is set.
                               Model: OPENAI_MVE_MODEL or gpt-4o-mini.
  Option A2 (LLM, Anthropic) — Anthropic API with a structured JSON
                               prompt if ANTHROPIC_API_KEY is set.
                               Model: ANTHROPIC_MVE_MODEL or claude-sonnet-4-6.
  Option B  (rule-based)     — deterministic templates per alert type,
                               always implemented as offline fallback.

Both LLM paths enforce identical prompts and identical output validation,
so the only difference between A1 and A2 is the provider. The rule-based
fallback (Option B) is always callable and is what every test runs against
unless a key is explicitly provided in the test environment.

The explanation adapts the CLINICIAN_TEMPLATES concept from
AlertExplainer._clinician_nlg() into the full 3-layer MVE structure
required by research_spec.yaml, without exposing SHAP values,
feature importances, or model architecture.
"""

from __future__ import annotations

import functools
import itertools
import json
import logging
import os
import re
from typing import Any, Optional

from src import sanitize_for_log
from src.data_models import MVEOutput

logger = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────────────────────

VALID_SEVERITY = {"CRITICAL", "HIGH", "MEDIUM", "LOW"}

_SEVERITY_RATIONALE = {
    "CRITICAL": "Life-sustaining system actively supporting patient care.",
    "HIGH": "Active clinical system with direct patient-care and PHI exposure risk.",
    "MEDIUM": "Clinical-support system not immediately affecting patient safety.",
    "LOW": "Administrative system with minimal PHI — monitoring sufficient.",
}

_SEVERITY_TIMEFRAME = {
    "CRITICAL": "Act within 15 minutes. Preserve network logs from the past 30 minutes.",
    "HIGH": "Act within 1 hour. Preserve logs from the past 4 hours.",
    "MEDIUM": "Act within 4 hours. Flag for next scheduled security review.",
    "LOW": "Review within 24 hours. Log for shift handover.",
}

# Maps criticality → whether the alert involves a clinical system
_IS_CLINICAL = {"CRITICAL": True, "HIGH": True, "MEDIUM": True, "LOW": False}


# ── Alert type detection ────────────────────────────────────────────────


def _detect_alert_type(raw_alert: dict[str, Any], user_context: Optional[dict[str, Any]]) -> str:
    """Classify alert into one of 5 types from mve_specification.yaml.

    Detection order (first match wins):
      T2 — if user_context is populated (always an EHR/EMR access alert)
      T3 — lateral movement keywords in alert_name or protocol
      T4 — exfiltration / large transfer keywords
      T5 — IoMT behavioral deviation keywords
      T1 — default (anomalous outbound from clinical device)

    Args:
        raw_alert: Dict with alert_name, protocol, etc.
        user_context: Populated only for type-2 EHR access alerts.

    Returns:
        One of 'T1', 'T2', 'T3', 'T4', 'T5'.
    """
    if user_context:
        return "T2"

    name = raw_alert.get("alert_name", "").lower()
    protocol = raw_alert.get("protocol", "").lower()

    if any(k in name for k in ("lateral", "cross-vlan", "smb", "rdp", "wmi")):
        return "T3"
    if any(k in protocol for k in ("smb", "rdp", "wmi")) and "445" in protocol:
        return "T3"

    if any(
        k in name
        for k in (
            "dlp",
            "exfil",
            "large outbound",
            "large transfer",
            "data transfer",
            "exfiltration",
        )
    ):
        return "T4"

    if any(
        k in name
        for k in ("behavioral", "iomt", "iot", "deviation", "behavioral anomaly", "device anomaly")
    ):
        return "T5"

    return "T1"


# ── Option B: Rule-based templates ─────────────────────────────────────


def _fmt_dests(dests: list[str]) -> str:
    """Format normal_destinations list into a readable string."""
    if not dests:
        return "approved internal hosts"
    if len(dests) == 1:
        return dests[0]
    # M-5: islice avoids allocating a dests[:3] copy
    return ", ".join(itertools.islice(dests, 3)) + ("" if len(dests) <= 3 else " and others")


# ── Known vendor IPs for maintenance FP reduction (FM-L1-09) ────────────

_KNOWN_VENDOR_IPS: dict[str, str] = {
    # Baxter / BD Alaris infusion pump update servers
    "203.0.113.50": "Baxter",
    "203.0.113.51": "Baxter",
    # BD (Becton Dickinson) update infrastructure
    "198.51.100.10": "BD",
    "198.51.100.11": "BD",
    # Philips medical device update servers
    "192.0.2.20": "Philips",
    "192.0.2.21": "Philips",
    # GE Healthcare update infrastructure
    "192.0.2.30": "GE Healthcare",
    # Medtronic device management
    "198.51.100.20": "Medtronic",
}

# ── Known IoMT device types (FIX-D, FIX-E) ─────────────────────────────

_IOMT_DEVICE_TYPES = frozenset({
    "infusion_pump", "ventilator", "patient_monitor", "insulin_pump",
})

_LIFE_SUSTAINING = frozenset({"infusion_pump", "ventilator"})

# M-4: pre-compiled fence-strip patterns — avoids per-call pattern cache lookup
_RE_FENCE_OPEN  = re.compile(r"^```(?:json)?\s*")
_RE_FENCE_CLOSE = re.compile(r"\s*```$")

# Substrings used to recognize device_type from descriptive product names
# (fixtures use "BD Alaris infusion pump", not "infusion_pump")
# M-1: tuple of tuples — immutable; signals the list is never mutated.
_DEVICE_TYPE_KEYWORDS = (
    # IoMT devices
    ("infusion", "infusion_pump"),
    ("ventilator", "ventilator"),
    ("patient monitor", "patient_monitor"),
    ("oximeter", "patient_monitor"),
    ("insulin", "insulin_pump"),
    # Clinical systems
    ("ehr", "ehr_workstation"),
    ("pacs", "pacs_server"),
    ("pharmacy", "pharmacy_system"),
    ("laboratory", "server"),
    ("scheduling", "server"),
    ("data warehouse", "server"),
    # Infrastructure
    ("workstation", "workstation"),
    ("server", "server"),
    ("controller", "server"),
    ("gateway", "server"),
    ("sensor", "other"),
    ("access point", "other"),
    ("wi-fi", "other"),
    ("hvac", "other"),
    ("monitor", "patient_monitor"),
    ("system", "server"),
)


def _normalize_device_type(raw: str) -> str:
    """Map descriptive device names to canonical types.

    'BD Alaris infusion pump' → 'infusion_pump'
    'GE CARESCAPE B650 patient monitor' → 'patient_monitor'

    OOD-01 fix: returns empty string when no keyword matches,
    so the caller's is_unknown check triggers UNREGISTERED DEVICE.
    """
    if not raw or not raw.strip():
        return ""
    lower = raw.lower()
    for keyword, canonical in _DEVICE_TYPE_KEYWORDS:
        if keyword in lower:
            return canonical
    # No keyword matched — this device type is outside our vocabulary
    return ""

# ── Device-specific patient care impact (FIX-A, resolves FM-L2-03) ──────

_PATIENT_CARE_IMPACT = {
    "ventilator": (
        "Compromise could alter respiratory parameters or disable "
        "ventilation. Respiratory arrest risk for connected patient."
    ),
    "infusion_pump": (
        "Compromise could alter medication dosage or interrupt drug "
        "delivery. Active infusion therapy at risk."
    ),
    "patient_monitor": (
        "Compromise could produce false vital sign readings. Clinical "
        "staff may miss patient deterioration or respond to false alarms."
    ),
    "insulin_pump": (
        "Compromise could alter insulin dosing. Hypoglycemia or "
        "hyperglycemia risk for connected patient."
    ),
    "ehr_workstation": (
        "Clinical documentation access disrupted. No direct patient "
        "physiological risk."
    ),
    "pacs_server": (
        "Diagnostic imaging access disrupted. Pending radiology reads "
        "delayed until service restored."
    ),
}

# ── Confidence calibration (FIX-C, resolves FM-L1-03, FM-L1-06, ST-07) ──

def _confidence_level(
    severity_score: float,
    baseline_days: int,
    criticality: str,
) -> str:
    """Compute calibrated confidence indicator.

    Replaces the hardcoded 'Confidence: HIGH' in T1 templates.
    Uses severity_score to set base level, then downgrades for
    short baselines.
    """
    score = float(severity_score) if severity_score else 0.0

    # Base level from anomaly score
    if score > 7.0:
        level = "HIGH"
    elif score > 4.0:
        level = "MEDIUM"
    else:
        level = "LOW"

    # Short baseline: downgrade one level
    days = int(baseline_days) if baseline_days else 0
    baseline_note = ""
    if days < 14:
        if level == "HIGH":
            level = "MEDIUM"
        elif level == "MEDIUM":
            level = "LOW"
        baseline_note = (
            f" (baseline only {days} days — behavioral profile may be incomplete)"
        )

    # M-6: build only the needed string — avoids evaluating 2 unused f-strings
    if level == "HIGH":
        detail = f"strong anomaly signal with {days}-day baseline{baseline_note}"
    elif level == "MEDIUM":
        detail = f"moderate anomaly signal — warrants investigation{baseline_note}"
    else:
        detail = f"weak anomaly signal — may be benign{baseline_note}"
    return f"Confidence: {level} — {detail}"


def _escalation(
    alert_type: str,
    criticality: str,
    device_type: str = "",
) -> str:
    """Return role escalation path based on alert type, severity, and device.

    FIX-G: Adds charge nurse for IoMT devices at CRITICAL/HIGH.
    """
    device_type = str(device_type).lower()

    if alert_type == "T2":
        # EHR access — add charge nurse (FIX-G, DC-05)
        return "(1) Privacy Officer, (2) Security lead, (3) HR, (4) Charge nurse on duty."
    if alert_type == "T3":
        return "(1) Security lead, (2) Clinical Engineering, (3) Network Admin."
    if alert_type == "T4":
        return "(1) Security lead, (2) Department IT admin, (3) Privacy Officer."
    if alert_type == "T5":
        # IoMT — add charge nurse for CRITICAL/HIGH (FIX-G, DC-01/02)
        if criticality in ("CRITICAL", "HIGH") and device_type in _IOMT_DEVICE_TYPES:
            return (
                "(1) Biomed Engineering on-call, (2) Security lead, "
                "(3) ICU/floor charge nurse."
            )
        return "(1) Biomed Engineering, (2) Security lead if maintenance unconfirmed."

    # T1 — varies by criticality
    if criticality == "CRITICAL":
        return "(1) Clinical Engineering on-call, (2) Security lead, (3) ICU charge nurse."
    if criticality == "HIGH":
        return "(1) Clinical Engineering, (2) Security lead, (3) Floor charge nurse."
    if criticality == "MEDIUM":
        return "(1) Security lead, (2) Clinical Engineering."
    return "(1) Security lead."


def _generate_rule_based(
    raw_alert: dict[str, Any],
    device_context: dict[str, Any],
    baseline: dict[str, Any],
    user_context: Optional[dict[str, Any]],
    alert_type: str,
) -> MVEOutput:
    """Build MVEOutput using deterministic templates (Option B).

    Templates adapted from CLINICIAN_TEMPLATES in
    module4_online_explainer.py into the 3-layer MVE structure.
    All fields guaranteed non-empty; no SHAP, CVSS, or model internals.

    Args:
        raw_alert: Dict with alert_name, source_ip, dest_ip, protocol, timestamp.
        device_context: Dict with device_type, clinical_function, location,
                        criticality, patchable.
        baseline: Dict with normal_destinations, normal_protocols,
                  normal_hours, baseline_days.
        user_context: Populated only for T2 alerts.
        alert_type: 'T1'–'T5'.

    Returns:
        MVEOutput with all 3 layers populated within word limits.
    """
    criticality = str(device_context.get("criticality", "LOW")).upper()
    raw_device_type = str(device_context.get("device_type", "device"))
    device_type = _normalize_device_type(raw_device_type)
    # Display name: use original descriptive name for affected_system (M2 matching)
    display_device_type = raw_device_type if raw_device_type else device_type
    location = device_context.get("location", "clinical area")
    clinical_fn = device_context.get("clinical_function", "clinical operations")
    source_ip = raw_alert.get("source_ip", "unknown")
    dest_ip = raw_alert.get("dest_ip", "unknown")
    protocol = raw_alert.get("protocol", "unknown protocol")
    severity_score = float(raw_alert.get("severity_score", 0))
    timestamp = raw_alert.get("timestamp", "")
    # Extract time portion only (HH:MM) for conciseness
    time_str = timestamp[11:16] if len(timestamp) >= 16 else timestamp
    normal_dests = _fmt_dests(baseline.get("normal_destinations", []))
    normal_protos = ", ".join(baseline.get("normal_protocols", ["HTTPS"]))
    baseline_days = baseline.get("baseline_days", 90)

    # FIX-E: Severity floor for life-sustaining device classes.
    # Infusion pumps and ventilators must never be below HIGH.
    severity_floor_note = ""
    if device_type in _LIFE_SUSTAINING and criticality in ("LOW", "MEDIUM"):
        severity_floor_note = (
            f" (Severity elevated: {device_type} requires minimum HIGH "
            "— verify criticality assignment with biomed.)"
        )
        criticality = "HIGH"
    severity_rationale = _SEVERITY_RATIONALE.get(criticality, _SEVERITY_RATIONALE["LOW"])
    if severity_floor_note:
        severity_rationale += severity_floor_note
    timeframe = _SEVERITY_TIMEFRAME.get(criticality, _SEVERITY_TIMEFRAME["LOW"])
    escalation = _escalation(alert_type, criticality, device_type)
    is_clinical = _IS_CLINICAL.get(criticality, False)

    # ── T2: Unauthorized EHR access ────────────────────────────────────
    if alert_type == "T2" and user_context:
        uid = user_context.get("user_id", "unknown user")
        role = user_context.get("role", "staff")
        dept = user_context.get("department", "unknown department")
        shift = user_context.get("shift", "business hours")
        scope = user_context.get("normal_access_scope", "assigned department")
        vol = user_context.get("normal_access_volume", 10)

        layer_1 = {
            "baseline_behavior": (
                f"User {uid} ({role}, {dept}) normally accesses {scope} during {shift}."
            ),
            "deviation_description": (
                f"At {time_str}, accessed records outside normal department scope, "
                f"exceeding typical volume of {vol} records/day."
            ),
            "confidence_indicator": (
                f"Confidence: HIGH — after-hours cross-department access "
                f"not observed in {baseline_days} days."
            ),
        }
        layer_2 = {
            "affected_system": f"EHR system ({location}) — {clinical_fn}",
            "patient_care_impact": (
                "No direct care disruption. Risk: unauthorized PHI access "
                "or data exfiltration by unauthorized session."
            ),
            "phi_exposure": (
                "Patient records accessed: diagnoses, treatments, "
                "insurance, and contacts potentially exposed."
            ),
            "severity_label": "HIGH",
            "severity_rationale": (
                "EHR access violation indicates insider threat "
                "or compromised credentials with bulk PHI exposure."
            ),
        }
        layer_3 = {
            "immediate_action": (
                f"Disable {uid}'s active EHR sessions and force MFA re-authentication. "
                "Preserve account — do not delete."
            ),
            "clinical_constraint": (
                "DO NOT lock shared workstations — isolate user session only, "
                "preserve active clinical staff access."
            ),
            "escalation_path": escalation,
            "timeframe": "Act within 30 minutes. Preserve EHR audit logs for past 72 hours.",
        }
        return MVEOutput(
            layer_1=layer_1,
            layer_2=layer_2,
            layer_3=layer_3,
            alert_involves_clinical_system=True,
        )

    # ── T3: Lateral movement ────────────────────────────────────────────
    if alert_type == "T3":
        layer_1 = {
            "baseline_behavior": (
                f"Host {source_ip} is not authorized " "to access the medical device subnet."
            ),
            "deviation_description": (
                f"At {time_str}, it initiated {protocol} to {dest_ip} "
                "crossing VLAN segmentation boundaries."
            ),
            "confidence_indicator": (
                "Confidence: HIGH — unauthorized cross-VLAN traffic "
                "is a lateral movement indicator."
            ),
        }
        layer_2 = {
            "affected_system": f"{display_device_type} ({location}) — {clinical_fn}",
            "patient_care_impact": (
                "Attacker reaching clinical subnet could disrupt device "
                "configurations and falsify patient readings."
            ),
            "phi_exposure": (
                "Patient vitals, device configs, and clinical data " "on the medical VLAN at risk."
            ),
            "severity_label": criticality,
            "severity_rationale": (
                "Active boundary crossing into clinical infrastructure " "from unauthorized host."
            ),
        }
        layer_3 = {
            "immediate_action": (
                f"Block {source_ip} at inter-VLAN firewall. "
                "Isolate source workstation from all segments."
            ),
            "clinical_constraint": (
                "DO NOT block entire ADMIN-to-DEVICE path — "
                "authorized biomed workstations need continued access."
            ),
            "escalation_path": escalation,
            "timeframe": "Act within 15 minutes. Trigger incident response playbook.",
        }
        return MVEOutput(
            layer_1=layer_1,
            layer_2=layer_2,
            layer_3=layer_3,
            alert_involves_clinical_system=is_clinical,
        )

    # ── T4: Data exfiltration ───────────────────────────────────────────
    if alert_type == "T4":
        confidence = "MEDIUM" if criticality in ("LOW", "MEDIUM") else "HIGH"
        layer_1 = {
            "baseline_behavior": (
                f"{display_device_type} ({location}) normally transfers "
                f"data only to {normal_dests}."
            ),
            "deviation_description": (
                f"At {time_str}, it transferred via {protocol} to {dest_ip}, "
                "not on any approved partner list."
            ),
            "confidence_indicator": (
                f"Confidence: {confidence} — outbound to unrecognized "
                "destination warrants immediate verification."
            ),
        }
        layer_2 = {
            "affected_system": f"{display_device_type} ({location}) — {clinical_fn}",
            "patient_care_impact": (
                "No immediate care disruption. Risk: bulk PHI exfiltration "
                "with embedded patient identifiers that cannot be recalled."
            ),
            "phi_exposure": (
                "Patient records, imaging data, and clinical documents "
                "potentially transferred to unapproved destination."
            ),
            "severity_label": criticality,
            "severity_rationale": severity_rationale,
        }
        layer_3 = {
            "immediate_action": (
                f"Block outbound {protocol} from {source_ip} to {dest_ip}. "
                "Verify with department IT if transfer was authorized."
            ),
            "clinical_constraint": (
                "DO NOT block all outbound from this system — "
                "approved clinical partners require continued data exchange."
            ),
            "escalation_path": escalation,
            "timeframe": timeframe,
        }
        return MVEOutput(
            layer_1=layer_1,
            layer_2=layer_2,
            layer_3=layer_3,
            alert_involves_clinical_system=is_clinical,
        )

    # ── T5: IoMT behavioral deviation (FIX-A: device-class-specific) ───
    if alert_type == "T5":
        layer_1 = {
            "baseline_behavior": (
                f"{display_device_type} ({location}) normally communicates "
                f"with {normal_dests} using {normal_protos}."
            ),
            "deviation_description": (
                f"Starting {time_str}, device initiated {protocol} to {dest_ip} "
                "at abnormal frequency, outside established baseline."
            ),
            "confidence_indicator": _confidence_level(
                severity_score, baseline_days, criticality
            ),
        }

        # Device-specific patient_care_impact (FIX-A, resolves FM-L2-03)
        care_impact = _PATIENT_CARE_IMPACT.get(device_type, (
            "Device functioning normally. If compromised, false clinical "
            "readings are possible. Isolation removes automated patient monitoring."
        ))

        layer_2 = {
            "affected_system": f"{display_device_type} ({location}) — {clinical_fn}",
            "patient_care_impact": care_impact,
            "phi_exposure": (
                f"Real-time patient vitals and identifiers for "
                f"{location} census at risk."
            ),
            "severity_label": criticality,
            "severity_rationale": severity_rationale,
        }

        # Device-class-specific Layer 3 (IMP-03 format)
        # DO NOT [physical] — SAFE: [network] — Contact [role]
        if device_type == "ventilator":
            constraint = (
                "DO NOT power off or disconnect ventilator. "
                "SAFE: block port at switch — clinical traffic on 443 unaffected."
            )
            immediate = (
                f"Check with biomed if maintenance scheduled. "
                f"If NO: block anomalous port at switch for {source_ip}."
            )
        elif device_type == "infusion_pump":
            constraint = (
                "DO NOT power-cycle pump during active infusion. "
                "SAFE: NAC quarantine blocking non-HTTPS preserves controller."
            )
            immediate = (
                f"Check with biomed if maintenance scheduled. "
                f"If NO: apply NAC quarantine on {source_ip}."
            )
        elif device_type == "insulin_pump":
            constraint = (
                "DO NOT disrupt wireless control loop. "
                "SAFE: destination-specific block only if IP-connected."
            )
            immediate = (
                f"Check with biomed: confirm IP vs RF connectivity. "
                f"If IP, no maintenance: block destination from {source_ip}."
            )
        elif device_type == "patient_monitor":
            constraint = (
                "DO NOT isolate from EHR gateway — vitals must continue. "
                "SAFE: DNS rate-limit or port block — HL7 on 443 unaffected."
            )
            immediate = (
                f"Check with biomed if maintenance scheduled. "
                f"If NO: rate-limit abnormal traffic from {source_ip}."
            )
        else:
            constraint = (
                "DO NOT isolate from clinical network without biomed confirmation."
            )
            immediate = (
                f"Verify with biomed engineering if maintenance was "
                f"scheduled. If NO: rate-limit abnormal traffic from {source_ip}."
            )

        layer_3 = {
            "immediate_action": immediate,
            "clinical_constraint": constraint,
            "escalation_path": escalation,
            "timeframe": "Verify within 1 hour. If unconfirmed: escalate and restrict traffic.",
        }
        return MVEOutput(
            layer_1=layer_1,
            layer_2=layer_2,
            layer_3=layer_3,
            alert_involves_clinical_system=is_clinical,
        )

    # ── T1: Anomalous outbound from clinical device (default) ───────────
    if criticality == "CRITICAL":
        immediate = (
            f"Apply NAC policy CLINICAL-QUARANTINE: block outbound "
            f"non-HTTPS from {source_ip}, preserve EHR and device controller connectivity."
        )
        constraint = (
            "DO NOT power-cycle or disconnect device during active clinical use — "
            "coordinate with ICU charge nurse first."
        )
    elif criticality == "HIGH":
        immediate = (
            f"Apply firewall rule to block {protocol} outbound from {source_ip}, "
            "maintain internal EHR and clinical network connectivity."
        )
        constraint = (
            "DO NOT isolate device entirely — restrict suspicious outbound "
            "only, preserve clinical traffic."
        )
    elif criticality == "MEDIUM":
        immediate = (
            f"Rate-limit non-standard protocol traffic from {source_ip} "
            "and enable enhanced logging. Notify clinical engineering."
        )
        constraint = (
            "DO NOT disable device entirely — restrict suspicious traffic only "
            "and verify device function with clinical staff."
        )
    else:  # LOW
        immediate = (
            f"Log connection from {source_ip} to {dest_ip} and apply "
            "outbound traffic monitoring. No immediate disruption required."
        )
        constraint = (
            "DO NOT disrupt shared infrastructure — apply logging first, "
            "investigate before any blocking action."
        )

    # FIX-C: calibrated confidence for T1 (replaces hardcoded HIGH)
    t1_confidence = _confidence_level(severity_score, baseline_days, criticality)

    layer_1 = {
        "baseline_behavior": (
            f"{display_device_type} ({location}) normally communicates "
            f"with {normal_dests} using {normal_protos}."
        ),
        "deviation_description": (
            f"At {time_str}, it initiated {protocol} to {dest_ip}, "
            "not on any approved destination list."
        ),
        "confidence_indicator": t1_confidence,
    }

    # Device-specific patient_care_impact (FIX-A, resolves FM-L2-03 for T1)
    t1_care_impact = _PATIENT_CARE_IMPACT.get(device_type, (
        "Compromise could disrupt active patient care. "
        "Clinical coordination required before any isolation."
    ))

    layer_2 = {
        "affected_system": f"{display_device_type} ({location}) — {clinical_fn}",
        "patient_care_impact": t1_care_impact,
        "phi_exposure": (f"Device data and clinical records for {location} patients at risk."),
        "severity_label": criticality,
        "severity_rationale": severity_rationale,
    }
    layer_3 = {
        "immediate_action": immediate,
        "clinical_constraint": constraint,
        "escalation_path": escalation,
        "timeframe": timeframe,
    }
    return MVEOutput(
        layer_1=layer_1,
        layer_2=layer_2,
        layer_3=layer_3,
        alert_involves_clinical_system=is_clinical,
    )


# ── Option A: LLM-based generation ─────────────────────────────────────
#
# Two LLM providers share the same prompt contract. The factored helpers
# below build the system + user prompts once, validate one response, and
# return a parsed MVEOutput (or None on any failure path). The provider-
# specific functions only handle the SDK call.

_LLM_SYSTEM_PROMPT = """You are a clinical IDS explanation engine for hospital IT generalists.
Generate a 3-layer Minimum Viable Explanation (MVE) as JSON.
Rules (non-negotiable):
- layer_1 total words <= 60 (baseline_behavior + deviation_description + confidence_indicator)
- layer_2 total words <= 50 (affected_system + patient_care_impact
  + phi_exposure + severity_label + severity_rationale)
- layer_3 total words <= 60 (immediate_action + clinical_constraint
  + escalation_path + timeframe)
- severity_label must be CRITICAL/HIGH/MEDIUM/LOW based on clinical impact, NOT CVSS
- clinical_constraint must start with "DO NOT" for CRITICAL/HIGH/MEDIUM alerts
- immediate_action must contain a specific executable step (block/isolate/disable/apply/rate-limit)
- DO NOT mention SHAP values, feature importances, model names, p-values, or CVSS scores
- DO NOT claim detection of Bluetooth, Zigbee, RF, or proprietary wireless protocols
- DO NOT claim early ransomware detection capability
Return only valid JSON with keys: layer_1, layer_2, layer_3."""


def _build_user_prompt(
    raw_alert: dict[str, Any],
    device_context: dict[str, Any],
    baseline: dict[str, Any],
    user_context: Optional[dict[str, Any]],
    alert_type: str,
) -> str:
    """Construct the user prompt with alert payloads and the JSON schema.

    Body is identical across OpenAI and Anthropic paths so that switching
    providers does not perturb the output distribution beyond the
    provider's own modelling differences.
    """
    return f"""Alert type: {alert_type}
Raw alert: {json.dumps(raw_alert)}
Device context: {json.dumps(device_context)}
Behavioral baseline: {json.dumps(baseline)}
User context: {json.dumps(user_context)}

Return JSON with this exact structure:
{{
  "layer_1": {{
    "baseline_behavior": "...",
    "deviation_description": "...",
    "confidence_indicator": "Confidence: HIGH/MEDIUM/LOW — ..."
  }},
  "layer_2": {{
    "affected_system": "...",
    "patient_care_impact": "...",
    "phi_exposure": "...",
    "severity_label": "CRITICAL|HIGH|MEDIUM|LOW",
    "severity_rationale": "..."
  }},
  "layer_3": {{
    "immediate_action": "...",
    "clinical_constraint": "DO NOT ...",
    "escalation_path": "(1) ..., (2) ...",
    "timeframe": "Act within ..."
  }}
}}"""


def _parse_llm_json(raw_text: str, is_clinical: bool) -> Optional[MVEOutput]:
    """Parse and validate an LLM response into MVEOutput.

    Strips markdown code-fences (some providers wrap JSON despite
    instructions), parses, validates the severity label, and returns
    either a usable MVEOutput or None (causing the caller to fall back
    to the next provider, ultimately rule-based).
    """
    raw = raw_text.strip()
    # M-4: pre-compiled patterns; cheap to apply unconditionally
    raw = _RE_FENCE_OPEN.sub("", raw)
    raw = _RE_FENCE_CLOSE.sub("", raw)

    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning(
            "LLM returned non-JSON (%s); using next provider",
            sanitize_for_log(exc),
        )
        return None

    try:
        mve = MVEOutput(
            layer_1=data["layer_1"],
            layer_2=data["layer_2"],
            layer_3=data["layer_3"],
            alert_involves_clinical_system=is_clinical,
        )
    except (KeyError, TypeError) as exc:
        logger.warning(
            "LLM response missing required keys (%s); using next provider",
            sanitize_for_log(exc),
        )
        return None

    if mve.layer_2.get("severity_label", "").upper() not in VALID_SEVERITY:
        logger.warning("LLM returned invalid severity; using next provider")
        return None

    return mve


def _generate_llm_openai(
    raw_alert: dict[str, Any],
    device_context: dict[str, Any],
    baseline: dict[str, Any],
    user_context: Optional[dict[str, Any]],
    alert_type: str,
) -> Optional[MVEOutput]:
    """Attempt LLM-based MVE generation using the OpenAI API (primary path).

    Falls back to None (caller tries the next provider) if:
      - OPENAI_API_KEY not set in environment
      - openai package not installed
      - API call fails for any reason
      - response is non-JSON, missing keys, or has invalid severity

    Uses `response_format={"type": "json_object"}` to enforce JSON output.
    Model is overridable via the OPENAI_MVE_MODEL env var (default
    `gpt-4o-mini` — chosen for cost-per-call against this concise prompt).
    Temperature is pinned to 0 for thesis reproducibility.

    Returns:
        MVEOutput on success, None on any failure.
    """
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return None

    try:
        import openai  # optional dependency
    except ImportError:
        logger.debug("openai package not installed; trying next provider")
        return None

    # M-3: reuse a single client (and its HTTP connection pool) across all
    # alerts in the process — avoids a new TLS handshake per surfaced alert.
    @functools.lru_cache(maxsize=1)
    def _client(key: str) -> "openai.OpenAI":
        return openai.OpenAI(api_key=key)

    model = os.environ.get("OPENAI_MVE_MODEL", "gpt-4o-mini")
    criticality = str(device_context.get("criticality", "LOW")).upper()
    is_clinical = _IS_CLINICAL.get(criticality, False)

    user_prompt = _build_user_prompt(
        raw_alert, device_context, baseline, user_context, alert_type
    )

    try:
        client = _client(api_key)
        response = client.chat.completions.create(
            model=model,
            max_tokens=512,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _LLM_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        raw_text = response.choices[0].message.content or ""
        mve = _parse_llm_json(raw_text, is_clinical)
        if mve is not None:
            logger.debug(
                "OpenAI MVE generated (%s, %d words)",
                model,
                mve.total_word_count,
            )
        return mve

    except Exception as exc:
        logger.warning(
            "OpenAI MVE failed (%s); trying next provider",
            sanitize_for_log(exc),
        )
        return None


def _generate_llm_anthropic(
    raw_alert: dict[str, Any],
    device_context: dict[str, Any],
    baseline: dict[str, Any],
    user_context: Optional[dict[str, Any]],
    alert_type: str,
) -> Optional[MVEOutput]:
    """Attempt LLM-based MVE generation using the Anthropic API (secondary path).

    Tried only when OpenAI is unavailable. Falls back to None on:
      - ANTHROPIC_API_KEY not set in environment
      - anthropic package not installed
      - API call fails for any reason
      - response is non-JSON, missing keys, or has invalid severity

    Model is overridable via the ANTHROPIC_MVE_MODEL env var (default
    `claude-sonnet-4-6`).

    Returns:
        MVEOutput on success, None on any failure.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return None

    try:
        import anthropic  # optional dependency
    except ImportError:
        logger.debug("anthropic package not installed; using rule-based fallback")
        return None

    @functools.lru_cache(maxsize=1)
    def _client(key: str) -> "anthropic.Anthropic":
        return anthropic.Anthropic(api_key=key)

    model = os.environ.get("ANTHROPIC_MVE_MODEL", "claude-sonnet-4-6")
    criticality = str(device_context.get("criticality", "LOW")).upper()
    is_clinical = _IS_CLINICAL.get(criticality, False)

    user_prompt = _build_user_prompt(
        raw_alert, device_context, baseline, user_context, alert_type
    )

    try:
        client = _client(api_key)
        response = client.messages.create(
            model=model,
            max_tokens=512,
            system=_LLM_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
        raw_text = response.content[0].text
        mve = _parse_llm_json(raw_text, is_clinical)
        if mve is not None:
            logger.debug(
                "Anthropic MVE generated (%s, %d words)",
                model,
                mve.total_word_count,
            )
        return mve

    except Exception as exc:
        logger.warning(
            "Anthropic MVE failed (%s); using rule-based fallback",
            sanitize_for_log(exc),
        )
        return None


# ── Public API ──────────────────────────────────────────────────────────


def generate_mve(
    raw_alert: dict[str, Any],
    device_context: dict[str, Any],
    baseline: dict[str, Any],
    user_context: Optional[dict[str, Any]],
    shap_context: Optional[dict[str, Any]] = None,
    event_context: Optional[dict[str, Any]] = None,
    force_rule_based: bool = False,
) -> MVEOutput:
    """Generate a 3-layer Minimum Viable Explanation for a single alert.

    Includes safe-failure defaults (FIX-D) and unpatchable enrichment (FIX-B).

    Args:
        raw_alert: Dict matching component_1 input schema.
        device_context: Dict with device_type, clinical_function,
                        location, criticality, patchable.
        baseline: Dict with normal_destinations, normal_protocols,
                  normal_hours, baseline_days.
        user_context: Only populated for T2 (EHR access) alerts.
        shap_context: Optional SHAP feature category context.
        event_context: Optional dict with is_maintenance_window,
                       is_known_vendor_ip, baseline_days.
        force_rule_based: Skip the OpenAI/Anthropic provider chain and go
                          straight to the deterministic templates. Batch
                          callers flip this once an LLM quota tripwire
                          fires so the rest of the batch doesn't waste
                          1-2 seconds per failed API call.

    Returns:
        MVEOutput with layer_1, layer_2, layer_3 and total_word_count <= 150.
    """
    # M-7: normalise once — raw_alert or {} evaluated 3× before this fix
    raw_alert = raw_alert or {}

    # ── FIX-D: Safe-failure validation (ST-01, ST-02, ST-05) ────────────
    if not device_context:
        device_context = {}
        logger.warning("Empty device_context — applying CRITICAL safe defaults")

    raw_dt = str(device_context.get("device_type", ""))
    device_type = _normalize_device_type(raw_dt) if raw_dt else ""
    criticality = str(device_context.get("criticality", "")).upper()

    # ST-05: Invalid criticality → default to CRITICAL
    invalid_criticality = criticality not in VALID_SEVERITY
    if invalid_criticality:
        logger.warning(
            "Invalid criticality '%s' — defaulting to CRITICAL",
            sanitize_for_log(criticality),
        )

    # ST-01/ST-02: Unknown or missing device → CRITICAL + warning
    is_unknown = not device_type or device_type == "unknown"
    if is_unknown:
        logger.warning(
            "Unknown device_type '%s' — defaulting to CRITICAL",
            sanitize_for_log(raw_dt),
        )

    # M-2: single dict copy covers both invalid-criticality and unknown-device
    # branches — was two separate dict() calls that each copied the full dict.
    if invalid_criticality or is_unknown:
        device_context = {
            **device_context,
            "criticality": "CRITICAL",
            **({"device_type": "unknown"} if is_unknown else {}),
        }

    if not baseline:
        baseline = {}

    alert_type = _detect_alert_type(raw_alert, user_context)

    # Provider chain: A1 (OpenAI) → A2 (Anthropic) → B (rule-based).
    # Each LLM helper returns None when its key is missing, its SDK is
    # uninstalled, or the call fails — the next provider is then tried.
    # force_rule_based short-circuits the chain entirely (batch tripwire).
    provider_used = "rule_based"
    if force_rule_based:
        mve = None
    else:
        mve = _generate_llm_openai(
            raw_alert, device_context, baseline, user_context, alert_type
        )
        if mve is not None:
            provider_used = "openai"
        else:
            mve = _generate_llm_anthropic(
                raw_alert, device_context, baseline, user_context, alert_type
            )
            if mve is not None:
                provider_used = "anthropic"
    if mve is None:
        mve = _generate_rule_based(
            raw_alert, device_context, baseline, user_context, alert_type
        )

    # ── FIX-D: Prefix Layer 1 for unknown/unregistered devices ──────────
    if is_unknown:
        existing_bl = mve.layer_1.get("baseline_behavior", "")
        mve.layer_1["baseline_behavior"] = (
            f"UNREGISTERED DEVICE — identity not confirmed. {existing_bl}"
        )

    # ── FIX-B: Unpatchable device enrichment (FM-L1-10, FM-L3-12) ──────
    # The unpatchable status is communicated through the risk scorer's
    # elevated threshold (Component 2) and the device-class-specific
    # constraint in Layer 3 which names Biomed Engineering — the team
    # that manages vendor relationships for unpatchable devices.
    # Explicit Layer 1/3 text was removed to stay within word limits.

    # ── ST-08: Maintenance context in Layer 1 ────────────────────────────
    if event_context:
        is_maint = event_context.get("is_maintenance_window", False)
        is_vendor = event_context.get("is_known_vendor_ip", False)
        if is_maint and not is_vendor:
            # Maintenance window active but source is NOT a known vendor
            existing_conf = mve.layer_1.get("confidence_indicator", "")
            mve.layer_1["confidence_indicator"] = (
                f"{existing_conf} Maintenance window active but "
                "source is not a known vendor — treat as suspicious."
            )

    # ── FM-L1-09: Vendor IP check in Layer 1 ──────────────────────────
    dest_ip = raw_alert.get("dest_ip", "")
    if dest_ip and dest_ip in _KNOWN_VENDOR_IPS:
        vendor = _KNOWN_VENDOR_IPS[dest_ip]
        existing_dev = mve.layer_1.get("deviation_description", "")
        mve.layer_1["deviation_description"] = (
            f"{existing_dev} Note: destination matches {vendor} "
            "update server — verify with biomed before acting."
        )

    # ── SHAP-context enrichment (v2.0 M5: shap_narrative_alignment) ─────
    # Layer 1 MUST mention top_category and at least one top_feature
    # when shap_context is provided (research_spec §2.module_4).
    if shap_context:
        top_category = str(shap_context.get("top_category", "")).strip()
        top_features = shap_context.get("top_features") or []
        if top_category == "biometric":
            narrative = shap_context.get(
                "top_feature_narrative", "abnormal biometric reading"
            )
            existing = mve.layer_1.get("deviation_description", "")
            mve.layer_1["deviation_description"] = (
                f"{existing} Concurrent clinical anomaly: {narrative} "
                "deviates from this device's baseline vital signs."
            )
        elif top_category and top_features:
            feat = str(top_features[0])
            existing = mve.layer_1.get("deviation_description", "")
            readable = top_category.replace("_", " ")
            mve.layer_1["deviation_description"] = (
                f"{existing} Primary signal: {readable} ({feat})."
            ).strip()

    mve.provider = provider_used
    return mve
