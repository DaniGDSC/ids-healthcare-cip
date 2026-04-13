"""Component 1: MVE Generator.

Produces a 3-layer Minimum Viable Explanation from a raw alert,
device context, behavioral baseline, and optional user context.

Design mirrors the two-track approach in
pipeline/module4_explanations/module4_online_explainer.py:
  Option A (LLM) — Anthropic API with a structured JSON prompt if
                   ANTHROPIC_API_KEY is set in the environment.
  Option B (rule-based) — deterministic templates per alert type,
                          always implemented as offline fallback.

The explanation adapts the CLINICIAN_TEMPLATES concept from
AlertExplainer._clinician_nlg() into the full 3-layer MVE structure
required by research_spec.yaml, without exposing SHAP values,
feature importances, or model architecture.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Optional

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


def _detect_alert_type(raw_alert: dict, user_context: Optional[dict]) -> str:
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


def _fmt_dests(dests: list) -> str:
    """Format normal_destinations list into a readable string."""
    if not dests:
        return "approved internal hosts"
    if len(dests) == 1:
        return dests[0]
    return ", ".join(dests[:3]) + ("" if len(dests) <= 3 else " and others")


def _escalation(alert_type: str, criticality: str) -> str:
    """Return role escalation path based on alert type and severity."""
    if alert_type == "T2":
        return "(1) Privacy Officer, (2) Security lead, (3) HR."
    if alert_type == "T3":
        return "(1) Security lead, (2) Clinical Engineering, (3) Network Admin."
    if alert_type == "T4":
        return "(1) Security lead, (2) Department IT admin, (3) Privacy Officer."
    if alert_type == "T5":
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
    raw_alert: dict,
    device_context: dict,
    baseline: dict,
    user_context: Optional[dict],
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
    device_type = device_context.get("device_type", "device")
    location = device_context.get("location", "clinical area")
    clinical_fn = device_context.get("clinical_function", "clinical operations")
    source_ip = raw_alert.get("source_ip", "unknown")
    dest_ip = raw_alert.get("dest_ip", "unknown")
    protocol = raw_alert.get("protocol", "unknown protocol")
    timestamp = raw_alert.get("timestamp", "")
    # Extract time portion only (HH:MM) for conciseness
    time_str = timestamp[11:16] if len(timestamp) >= 16 else timestamp
    normal_dests = _fmt_dests(baseline.get("normal_destinations", []))
    normal_protos = ", ".join(baseline.get("normal_protocols", ["HTTPS"]))
    baseline_days = baseline.get("baseline_days", 90)
    severity_rationale = _SEVERITY_RATIONALE.get(criticality, _SEVERITY_RATIONALE["LOW"])
    timeframe = _SEVERITY_TIMEFRAME.get(criticality, _SEVERITY_TIMEFRAME["LOW"])
    escalation = _escalation(alert_type, criticality)
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
                f"Confidence: HIGH — unauthorized cross-VLAN traffic "
                "is a lateral movement indicator."
            ),
        }
        layer_2 = {
            "affected_system": f"{device_type} ({location}) — {clinical_fn}",
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
                f"{device_type} ({location}) normally transfers data " f"only to {normal_dests}."
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
            "affected_system": f"{device_type} ({location}) — {clinical_fn}",
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

    # ── T5: IoMT behavioral deviation ──────────────────────────────────
    if alert_type == "T5":
        layer_1 = {
            "baseline_behavior": (
                f"{device_type} ({location}) normally communicates "
                f"with {normal_dests} using {normal_protos}."
            ),
            "deviation_description": (
                f"Starting {time_str}, device initiated {protocol} to {dest_ip} "
                "at abnormal frequency, outside established baseline."
            ),
            "confidence_indicator": (
                "Confidence: MEDIUM — may indicate firmware update, "
                "vendor maintenance, or behavioral compromise."
            ),
        }
        layer_2 = {
            "affected_system": f"{device_type} ({location}) — {clinical_fn}",
            "patient_care_impact": (
                "Device functioning normally. If compromised, false clinical "
                "readings are possible. Isolation removes automated patient monitoring."
            ),
            "phi_exposure": (
                f"Real-time patient vitals and identifiers for " f"{location} census at risk."
            ),
            "severity_label": criticality,
            "severity_rationale": severity_rationale,
        }
        layer_3 = {
            "immediate_action": (
                f"Verify with biomed engineering if maintenance was scheduled. "
                f"If NO: rate-limit abnormal traffic from {source_ip}."
            ),
            "clinical_constraint": (
                "DO NOT isolate device from EHR gateway — "
                "vital sign reporting to nursing station must continue."
            ),
            "escalation_path": escalation,
            "timeframe": ("Verify within 1 hour. If unconfirmed: escalate and restrict traffic."),
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

    layer_1 = {
        "baseline_behavior": (
            f"{device_type} ({location}) normally communicates "
            f"with {normal_dests} using {normal_protos}."
        ),
        "deviation_description": (
            f"At {time_str}, it initiated {protocol} to {dest_ip}, "
            "not on any approved destination list."
        ),
        "confidence_indicator": (
            f"Confidence: HIGH — not observed in {baseline_days} days "
            "of baseline for this device class."
        ),
    }
    layer_2 = {
        "affected_system": f"{device_type} ({location}) — {clinical_fn}",
        "patient_care_impact": (
            "Compromise could disrupt active patient care. "
            "Clinical coordination required before any isolation."
        ),
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


def _generate_llm(
    raw_alert: dict,
    device_context: dict,
    baseline: dict,
    user_context: Optional[dict],
    alert_type: str,
) -> Optional[MVEOutput]:
    """Attempt LLM-based MVE generation using the Anthropic API.

    Falls back to None (caller uses rule-based) if:
      - ANTHROPIC_API_KEY not set in environment
      - anthropic package not installed
      - API call fails for any reason

    Prompt enforces 3-layer structure, word limits, clinical framing,
    and explicitly prohibits SHAP values, CVSS scores, model internals,
    and RF protocol claims.

    Args:
        raw_alert: Raw alert dict.
        device_context: Device context dict.
        baseline: Behavioral baseline dict.
        user_context: User context or None.
        alert_type: Alert type string (T1–T5).

    Returns:
        MVEOutput if successful, None if unavailable or failed.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return None

    try:
        import anthropic  # optional dependency
    except ImportError:
        logger.debug("anthropic package not installed; using rule-based fallback")
        return None

    criticality = str(device_context.get("criticality", "LOW")).upper()
    is_clinical = _IS_CLINICAL.get(criticality, False)

    system_prompt = """You are a clinical IDS explanation engine for hospital IT generalists.
Generate a 3-layer Minimum Viable Explanation (MVE) as JSON.
Rules (non-negotiable):
- layer_1 total words <= 60 (baseline_behavior + deviation_description + confidence_indicator)
- layer_2 total words <= 50 (affected_system + patient_care_impact + phi_exposure + severity_label + severity_rationale)
- layer_3 total words <= 60 (immediate_action + clinical_constraint + escalation_path + timeframe)
- severity_label must be CRITICAL/HIGH/MEDIUM/LOW based on clinical impact, NOT CVSS
- clinical_constraint must start with "DO NOT" for CRITICAL/HIGH/MEDIUM alerts
- immediate_action must contain a specific executable step (block/isolate/disable/apply/rate-limit)
- DO NOT mention SHAP values, feature importances, model names, p-values, or CVSS scores
- DO NOT claim detection of Bluetooth, Zigbee, RF, or proprietary wireless protocols
- DO NOT claim early ransomware detection capability
Return only valid JSON with keys: layer_1, layer_2, layer_3."""

    user_prompt = f"""Alert type: {alert_type}
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

    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=512,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        raw = response.content[0].text.strip()
        # Strip markdown code fences if present
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        data = json.loads(raw)

        mve = MVEOutput(
            layer_1=data["layer_1"],
            layer_2=data["layer_2"],
            layer_3=data["layer_3"],
            alert_involves_clinical_system=is_clinical,
        )
        # Validate severity label
        if mve.layer_2.get("severity_label", "").upper() not in VALID_SEVERITY:
            logger.warning("LLM returned invalid severity; using rule-based fallback")
            return None

        logger.debug("LLM MVE generated, %d words", mve.total_word_count)
        return mve

    except Exception as exc:
        logger.warning("LLM MVE failed (%s); using rule-based fallback", exc)
        return None


# ── Public API ──────────────────────────────────────────────────────────


def generate_mve(
    raw_alert: dict,
    device_context: dict,
    baseline: dict,
    user_context: Optional[dict],
    shap_context: Optional[dict] = None,
) -> MVEOutput:
    """Generate a 3-layer Minimum Viable Explanation for a single alert.

    Option A (LLM) is used when ANTHROPIC_API_KEY is set and the
    anthropic package is installed.  Falls back to Option B (rule-based)
    automatically on any failure.

    The rule-based path adapts CLINICIAN_TEMPLATES from
    AlertExplainer._clinician_nlg() (module4_online_explainer.py) into
    the full 3-layer MVE structure required by research_spec.yaml,
    without exposing SHAP values or model internals.

    Args:
        raw_alert: Dict matching component_1 input schema:
                   alert_name, source_ip, dest_ip, protocol,
                   timestamp, severity_score.
        device_context: Dict with device_type, clinical_function,
                        location, criticality, patchable.
        baseline: Dict with normal_destinations, normal_protocols,
                  normal_hours, baseline_days.
        user_context: Dict with user_id, department, role, shift,
                      normal_access_volume, normal_access_scope.
                      Only populated for T2 (EHR access) alerts.
        shap_context: Optional dict with top feature categories from
                      SHAP analysis. Used to add biometric context to
                      Layer 1 when biometric features dominate the
                      model's decision. Keys:
                        top_category: str — "biometric" | "network_timing" | etc.
                        top_feature_narrative: str — e.g., "abnormal temperature"
                      When None, Layer 1 uses network-only framing.

    Returns:
        MVEOutput with layer_1, layer_2, layer_3 and total_word_count <= 150.
    """
    alert_type = _detect_alert_type(raw_alert, user_context)

    # Try LLM first (Option A), fall back to rule-based (Option B)
    mve = _generate_llm(raw_alert, device_context, baseline, user_context, alert_type)
    if mve is None:
        mve = _generate_rule_based(raw_alert, device_context, baseline, user_context, alert_type)

    # Enrich Layer 1 with biometric context when SHAP indicates
    # biometric features dominate the model's decision.
    if shap_context and shap_context.get("top_category") == "biometric":
        narrative = shap_context.get("top_feature_narrative", "abnormal biometric reading")
        existing = mve.layer_1.get("deviation_description", "")
        mve.layer_1["deviation_description"] = (
            f"{existing} Concurrent clinical anomaly: {narrative} "
            "deviates from this device's baseline vital signs."
        )

    return mve
