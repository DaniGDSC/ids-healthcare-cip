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
from dataclasses import dataclass
from typing import Any, Optional

from src import sanitize_for_log
from src.data_models import MVEOutput

logger = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────────────────────

VALID_SEVERITY = {"CRITICAL", "HIGH", "MEDIUM", "LOW"}

# N8 fix: single dict carries both rationale + timeframe per tier (was
# two parallel dicts keyed identically). Helper proxies preserve the
# legacy ``_SEVERITY_RATIONALE.get(...)`` / ``_SEVERITY_TIMEFRAME.get(...)``
# call shape inside the rule-based templates.
_SEVERITY_META = {
    "CRITICAL": {
        "rationale": "Life-sustaining system actively supporting patient care.",
        "timeframe": "Act within 15m; preserve 30m of network logs.",
    },
    "HIGH": {
        "rationale": "Active clinical system with direct patient-care risk.",
        "timeframe": "Act within 1h; preserve 4h logs.",
    },
    "MEDIUM": {
        "rationale": "Clinical-support system not immediately affecting patient safety.",
        "timeframe": "Act within 4h; flag for next security review.",
    },
    "LOW": {
        "rationale": "Administrative system with minimal PHI — monitoring sufficient.",
        "timeframe": "Review within 24 hours. Log for shift handover.",
    },
}


class _SeverityMetaProxy:
    """Dict-like accessor backed by ``_SEVERITY_META``.

    Preserves the legacy ``_SEVERITY_RATIONALE[tier]`` / ``.get(tier, default)``
    call shape so the per-template helpers don't need rewriting.
    """

    __slots__ = ("_key",)

    def __init__(self, key: str):
        self._key = key

    def get(self, tier: str, default: str = "") -> str:
        return _SEVERITY_META.get(tier, {}).get(self._key, default)

    def __getitem__(self, tier: str) -> str:
        return _SEVERITY_META[tier][self._key]


_SEVERITY_RATIONALE = _SeverityMetaProxy("rationale")
_SEVERITY_TIMEFRAME = _SeverityMetaProxy("timeframe")

# Maps criticality → whether the alert involves a clinical system
_IS_CLINICAL = {"CRITICAL": True, "HIGH": True, "MEDIUM": True, "LOW": False}


# ── Alert type detection ────────────────────────────────────────────────


# N10 fix: alert-type detection keyword sets pulled out as module-level
# constants so future tuning (e.g. adding a new lateral-movement protocol)
# happens in one place rather than buried inside the detector function.
_T3_NAME_KEYWORDS = frozenset({"lateral", "cross-vlan", "smb", "rdp", "wmi"})
_T3_PROTOCOL_KEYWORDS = frozenset({"smb", "rdp", "wmi"})
_T4_NAME_KEYWORDS = frozenset({
    "dlp", "exfil", "large outbound", "large transfer",
    "data transfer", "exfiltration",
})
_T5_NAME_KEYWORDS = frozenset({
    "behavioral", "iomt", "iot", "deviation",
    "behavioral anomaly", "device anomaly",
})


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

    if any(k in name for k in _T3_NAME_KEYWORDS):
        return "T3"
    if any(k in protocol for k in _T3_PROTOCOL_KEYWORDS) and "445" in protocol:
        return "T3"
    if any(k in name for k in _T4_NAME_KEYWORDS):
        return "T4"
    if any(k in name for k in _T5_NAME_KEYWORDS):
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


# ── MITRE ATT&CK technique lookup (RQ2.e) ───────────────────────────
# Layer 1 narratives append "Consistent with MITRE TXXXX (Name)." when
# the alert's attack category maps to a technique in
# config/attack_to_mitre_mapping.yaml. Cached via lru_cache after first
# read; tests can reset via `_load_mitre_mapping.cache_clear()`.


@functools.lru_cache(maxsize=1)
def _load_mitre_mapping() -> dict:
    """Read config/attack_to_mitre_mapping.yaml once; return cached dict.

    Returns {} when PyYAML isn't installed or the file is missing/unreadable
    — MVE generation must not break because of an optional reference.
    """
    from pathlib import Path
    try:
        import yaml
    except ImportError:
        return {}
    path = Path(__file__).resolve().parent.parent / "config" / "attack_to_mitre_mapping.yaml"
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError):
        return {}


def _lookup_mitre_reference(attack_category: str) -> Optional[dict[str, str]]:
    """Return ``{'id', 'name', 'plain_gloss'}`` for the primary MITRE
    technique of an attack category, or None when no mapping applies
    (benign baseline, unknown category, or low-confidence association).

    Phase 1.4 — ``plain_gloss`` is a short, jargon-free clause that
    explains the technique without security expertise (sourced from
    ``config/attack_to_mitre_mapping.yaml``). Empty string when the
    YAML entry doesn't define one yet, so renderers can guard on
    truthiness rather than ``KeyError``.
    """
    if not attack_category:
        return None
    mapping = _load_mitre_mapping()
    cat = (mapping.get("attack_categories") or {}).get(attack_category)
    if not cat or cat.get("excluded_from_coverage_audit"):
        return None
    primary = cat.get("primary_technique") or {}
    pid = primary.get("id")
    pname = primary.get("name", "")
    pconf = str(primary.get("confidence", "")).lower()
    if not pid or pid == "NONE" or pconf == "low":
        return None
    return {
        "id": pid,
        "name": pname,
        "plain_gloss": primary.get("plain_gloss", "") or "",
    }


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
        "Compromise could produce false vital signs; "
        "staff may miss deterioration or react to false alarms."
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


@dataclass
class _TemplateContext:
    """Shared variables extracted from raw_alert/device_context/baseline.

    Phase-3 decomposition: replaces the 30-LOC variable-extraction block
    at the top of the old ``_generate_rule_based``. Built once per alert,
    passed to each ``_template_tN`` helper.
    """

    criticality: str
    device_type: str
    display_device_type: str
    location: str
    clinical_fn: str
    source_ip: str
    dest_ip: str
    protocol: str
    severity_score: float
    time_str: str
    normal_dests: str
    normal_protos: str
    baseline_days: int
    severity_rationale: str
    timeframe: str
    escalation: str
    is_clinical: bool


def _build_template_context(
    raw_alert: dict[str, Any],
    device_context: dict[str, Any],
    baseline: dict[str, Any],
    alert_type: str,
) -> _TemplateContext:
    """Build the template-shared context. Applies FIX-E (life-sustaining floor)."""
    criticality = str(device_context.get("criticality", "LOW")).upper()
    raw_device_type = str(device_context.get("device_type", "device"))
    device_type = _normalize_device_type(raw_device_type)
    display_device_type = raw_device_type if raw_device_type else device_type
    location = device_context.get("location", "clinical area")
    clinical_fn = device_context.get("clinical_function", "clinical operations")
    source_ip = raw_alert.get("source_ip", "unknown")
    dest_ip = raw_alert.get("dest_ip", "unknown")
    protocol = raw_alert.get("protocol", "unknown protocol")
    severity_score = float(raw_alert.get("severity_score", 0))
    timestamp = raw_alert.get("timestamp", "")
    time_str = timestamp[11:16] if len(timestamp) >= 16 else timestamp
    normal_dests = _fmt_dests(baseline.get("normal_destinations", []))
    normal_protos = ", ".join(baseline.get("normal_protocols", ["HTTPS"]))
    baseline_days = baseline.get("baseline_days", 90)

    # FIX-E: severity floor for life-sustaining device classes.
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

    return _TemplateContext(
        criticality=criticality,
        device_type=device_type,
        display_device_type=display_device_type,
        location=location,
        clinical_fn=clinical_fn,
        source_ip=source_ip,
        dest_ip=dest_ip,
        protocol=protocol,
        severity_score=severity_score,
        time_str=time_str,
        normal_dests=normal_dests,
        normal_protos=normal_protos,
        baseline_days=baseline_days,
        severity_rationale=severity_rationale,
        timeframe=timeframe,
        escalation=escalation,
        is_clinical=is_clinical,
    )


def _template_t2(ctx: _TemplateContext, user_context: dict[str, Any]) -> MVEOutput:
    """T2: Unauthorized EHR access. ``user_context`` must be populated."""
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
            f"At {ctx.time_str}, accessed records outside normal department scope, "
            f"exceeding typical volume of {vol} records/day."
        ),
        "confidence_indicator": (
            f"Confidence: HIGH — after-hours cross-department access "
            f"not observed in {ctx.baseline_days} days."
        ),
    }
    layer_2 = {
        "affected_system": f"EHR system ({ctx.location}) — {ctx.clinical_fn}",
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
        "escalation_path": ctx.escalation,
        "timeframe": "Act within 30 minutes. Preserve EHR audit logs for past 72 hours.",
    }
    return MVEOutput(layer_1=layer_1, layer_2=layer_2, layer_3=layer_3,
                     alert_involves_clinical_system=True)


def _template_t3(ctx: _TemplateContext) -> MVEOutput:
    """T3: Lateral movement."""
    layer_1 = {
        "baseline_behavior": (
            f"Host {ctx.source_ip} is not authorized "
            "to access the medical device subnet."
        ),
        "deviation_description": (
            f"At {ctx.time_str}, it initiated {ctx.protocol} to {ctx.dest_ip} "
            "crossing VLAN segmentation boundaries."
        ),
        "confidence_indicator": (
            "Confidence: HIGH — unauthorized cross-VLAN traffic "
            "is a lateral movement indicator."
        ),
    }
    layer_2 = {
        "affected_system": f"{ctx.display_device_type} ({ctx.location}) — {ctx.clinical_fn}",
        "patient_care_impact": (
            "Attacker reaching clinical subnet could disrupt device "
            "configurations and falsify patient readings."
        ),
        "phi_exposure": (
            "Patient vitals, device configs, and clinical data "
            "on the medical VLAN at risk."
        ),
        "severity_label": ctx.criticality,
        "severity_rationale": (
            "Active boundary crossing into clinical infrastructure "
            "from unauthorized host."
        ),
    }
    layer_3 = {
        "immediate_action": (
            f"Block {ctx.source_ip} at inter-VLAN firewall. "
            "Isolate source workstation from all segments."
        ),
        "clinical_constraint": (
            "DO NOT block entire ADMIN-to-DEVICE path — "
            "authorized biomed workstations need continued access."
        ),
        "escalation_path": ctx.escalation,
        "timeframe": "Act within 15 minutes. Trigger incident response playbook.",
    }
    return MVEOutput(layer_1=layer_1, layer_2=layer_2, layer_3=layer_3,
                     alert_involves_clinical_system=ctx.is_clinical)


def _template_t4(ctx: _TemplateContext) -> MVEOutput:
    """T4: Data exfiltration."""
    confidence = "MEDIUM" if ctx.criticality in ("LOW", "MEDIUM") else "HIGH"
    layer_1 = {
        "baseline_behavior": (
            f"{ctx.display_device_type} ({ctx.location}) normally transfers "
            f"data only to {ctx.normal_dests}."
        ),
        "deviation_description": (
            f"At {ctx.time_str}, it transferred via {ctx.protocol} to {ctx.dest_ip}, "
            "not on any approved partner list."
        ),
        "confidence_indicator": (
            f"Confidence: {confidence} — outbound to unrecognized "
            "destination warrants immediate verification."
        ),
    }
    layer_2 = {
        "affected_system": f"{ctx.display_device_type} ({ctx.location}) — {ctx.clinical_fn}",
        "patient_care_impact": (
            "No immediate care disruption. Risk: bulk PHI exfiltration "
            "with embedded patient identifiers that cannot be recalled."
        ),
        "phi_exposure": (
            "Patient records, imaging data, and clinical documents "
            "potentially transferred to unapproved destination."
        ),
        "severity_label": ctx.criticality,
        "severity_rationale": ctx.severity_rationale,
    }
    layer_3 = {
        "immediate_action": (
            f"Block outbound {ctx.protocol} from {ctx.source_ip} to {ctx.dest_ip}. "
            "Verify with department IT if transfer was authorized."
        ),
        "clinical_constraint": (
            "DO NOT block all outbound from this system — "
            "approved clinical partners require continued data exchange."
        ),
        "escalation_path": ctx.escalation,
        "timeframe": ctx.timeframe,
    }
    return MVEOutput(layer_1=layer_1, layer_2=layer_2, layer_3=layer_3,
                     alert_involves_clinical_system=ctx.is_clinical)


# Device-class-specific Layer 3 for T5 IoMT alerts (IMP-03 format):
# DO NOT [physical] — SAFE: [network] — Contact [role].
_T5_DEVICE_L3 = {
    "ventilator": (
        "Check with biomed if maintenance scheduled. "
        "If NO: block anomalous port at switch for {source_ip}.",
        "DO NOT power off or disconnect ventilator. "
        "SAFE: block port at switch — clinical traffic on 443 unaffected.",
    ),
    "infusion_pump": (
        "Check with biomed if maintenance scheduled. "
        "If NO: apply NAC quarantine on {source_ip}.",
        "DO NOT power-cycle pump during active infusion. "
        "SAFE: NAC quarantine blocking non-HTTPS preserves controller.",
    ),
    "insulin_pump": (
        "Check with biomed: confirm IP vs RF connectivity. "
        "If IP, no maintenance: block destination from {source_ip}.",
        "DO NOT disrupt wireless control loop. "
        "SAFE: destination-specific block only if IP-connected.",
    ),
    "patient_monitor": (
        "Check with biomed if maintenance scheduled. "
        "If NO: rate-limit abnormal traffic from {source_ip}.",
        "DO NOT isolate from EHR gateway — vitals must continue. "
        "SAFE: DNS rate-limit or port block — HL7 on 443 unaffected.",
    ),
}
_T5_DEFAULT_L3 = (
    "Verify with biomed engineering if maintenance was "
    "scheduled. If NO: rate-limit abnormal traffic from {source_ip}.",
    "DO NOT isolate from clinical network without biomed confirmation.",
)


def _template_t5(ctx: _TemplateContext) -> MVEOutput:
    """T5: IoMT behavioral deviation (FIX-A: device-class-specific)."""
    layer_1 = {
        "baseline_behavior": (
            f"{ctx.display_device_type} ({ctx.location}) normally communicates "
            f"with {ctx.normal_dests} using {ctx.normal_protos}."
        ),
        "deviation_description": (
            f"Starting {ctx.time_str}, device initiated {ctx.protocol} to {ctx.dest_ip} "
            "at abnormal frequency, outside established baseline."
        ),
        "confidence_indicator": _confidence_level(
            ctx.severity_score, ctx.baseline_days, ctx.criticality
        ),
    }
    care_impact = _PATIENT_CARE_IMPACT.get(ctx.device_type, (
        "Device functioning normally. If compromised, false clinical "
        "readings are possible. Isolation removes automated patient monitoring."
    ))
    layer_2 = {
        "affected_system": f"{ctx.display_device_type} ({ctx.location}) — {ctx.clinical_fn}",
        "patient_care_impact": care_impact,
        "phi_exposure": (
            f"Real-time patient vitals and identifiers for "
            f"{ctx.location} census at risk."
        ),
        "severity_label": ctx.criticality,
        "severity_rationale": ctx.severity_rationale,
    }
    immediate_tmpl, constraint = _T5_DEVICE_L3.get(ctx.device_type, _T5_DEFAULT_L3)
    layer_3 = {
        "immediate_action": immediate_tmpl.format(source_ip=ctx.source_ip),
        "clinical_constraint": constraint,
        "escalation_path": ctx.escalation,
        "timeframe": "Verify within 1 hour. If unconfirmed: escalate and restrict traffic.",
    }
    return MVEOutput(layer_1=layer_1, layer_2=layer_2, layer_3=layer_3,
                     alert_involves_clinical_system=ctx.is_clinical)


# T1 immediate-action + clinical-constraint templates keyed by criticality.
_T1_L3_BY_CRIT = {
    "CRITICAL": (
        "Apply NAC policy CLINICAL-QUARANTINE: block outbound "
        "non-HTTPS from {source_ip}, preserve EHR and device controller connectivity.",
        "DO NOT power-cycle or disconnect device during active clinical use — "
        "coordinate with ICU charge nurse first.",
    ),
    "HIGH": (
        "Apply firewall rule to block {protocol} outbound from {source_ip}, "
        "maintain internal EHR and clinical network connectivity.",
        "DO NOT isolate device entirely — restrict suspicious outbound "
        "only, preserve clinical traffic.",
    ),
    "MEDIUM": (
        "Rate-limit non-standard protocol traffic from {source_ip} "
        "and enable enhanced logging. Notify clinical engineering.",
        "DO NOT disable device entirely — restrict suspicious traffic only "
        "and verify device function with clinical staff.",
    ),
    "LOW": (
        "Log connection from {source_ip} to {dest_ip} and apply "
        "outbound traffic monitoring. No immediate disruption required.",
        "DO NOT disrupt shared infrastructure — apply logging first, "
        "investigate before any blocking action.",
    ),
}


def _template_t1(ctx: _TemplateContext) -> MVEOutput:
    """T1: Anomalous outbound from clinical device (default)."""
    immediate_tmpl, constraint = _T1_L3_BY_CRIT.get(ctx.criticality, _T1_L3_BY_CRIT["LOW"])
    layer_1 = {
        "baseline_behavior": (
            f"{ctx.display_device_type} ({ctx.location}) normally communicates "
            f"with {ctx.normal_dests} using {ctx.normal_protos}."
        ),
        "deviation_description": (
            f"At {ctx.time_str}, it initiated {ctx.protocol} to {ctx.dest_ip}, "
            "not on any approved destination list."
        ),
        "confidence_indicator": _confidence_level(
            ctx.severity_score, ctx.baseline_days, ctx.criticality
        ),
    }
    t1_care_impact = _PATIENT_CARE_IMPACT.get(ctx.device_type, (
        "Compromise could disrupt active patient care. "
        "Clinical coordination required before any isolation."
    ))
    layer_2 = {
        "affected_system": f"{ctx.display_device_type} ({ctx.location}) — {ctx.clinical_fn}",
        "patient_care_impact": t1_care_impact,
        "phi_exposure": (
            f"Device data and clinical records for {ctx.location} patients at risk."
        ),
        "severity_label": ctx.criticality,
        "severity_rationale": ctx.severity_rationale,
    }
    layer_3 = {
        "immediate_action": immediate_tmpl.format(
            source_ip=ctx.source_ip, protocol=ctx.protocol, dest_ip=ctx.dest_ip,
        ),
        "clinical_constraint": constraint,
        "escalation_path": ctx.escalation,
        "timeframe": ctx.timeframe,
    }
    return MVEOutput(layer_1=layer_1, layer_2=layer_2, layer_3=layer_3,
                     alert_involves_clinical_system=ctx.is_clinical)


def _generate_rule_based(
    raw_alert: dict[str, Any],
    device_context: dict[str, Any],
    baseline: dict[str, Any],
    user_context: Optional[dict[str, Any]],
    alert_type: str,
) -> MVEOutput:
    """Build MVEOutput using deterministic templates (Option B).

    Thin dispatcher: builds the shared :class:`_TemplateContext` and
    routes to ``_template_t1`` … ``_template_t5``. The per-type bodies
    are kept small and individually testable.
    """
    ctx = _build_template_context(raw_alert, device_context, baseline, alert_type)

    if alert_type == "T2" and user_context:
        return _template_t2(ctx, user_context)
    if alert_type == "T3":
        return _template_t3(ctx)
    if alert_type == "T4":
        return _template_t4(ctx)
    if alert_type == "T5":
        return _template_t5(ctx)
    # T1 / default
    return _template_t1(ctx)


# ── Legacy in-line template code (replaced by the dispatcher above) ───
# The original body has been factored into _template_tN helpers; the
# placeholder branch below preserves the original control flow for the
# duration of the diff but is unreachable at runtime.

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
- severity_label must equal the provided ``pipeline_risk_level`` verbatim when present (this is the canonical risk tier the response engine acts on); otherwise CRITICAL/HIGH/MEDIUM/LOW based on clinical impact, NOT CVSS. severity_rationale must justify that tier.
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
    risk_level: Optional[str] = None,
) -> str:
    """Construct the user prompt with alert payloads and the JSON schema.

    Body is identical across OpenAI and Anthropic paths so that switching
    providers does not perturb the output distribution beyond the
    provider's own modelling differences.
    """
    pipeline_line = (
        f"Pipeline risk_level (canonical — use verbatim as severity_label): {risk_level}\n"
        if risk_level
        else ""
    )
    return f"""Alert type: {alert_type}
{pipeline_line}Raw alert: {json.dumps(raw_alert)}
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
    risk_level: Optional[str] = None,
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
        raw_alert, device_context, baseline, user_context, alert_type,
        risk_level=risk_level,
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
    risk_level: Optional[str] = None,
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
        raw_alert, device_context, baseline, user_context, alert_type,
        risk_level=risk_level,
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
    risk_level: Optional[str] = None,
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
        risk_level: Optional canonical pipeline risk tier (one of
                    ``VALID_SEVERITY``). When provided, the final
                    ``layer_2.severity_label`` is coerced to this value so
                    the MVE agrees with module 3's ``risk_level`` (which
                    is what module 5's response engine acts on). The
                    LLM-/rule-derived value is preserved in
                    ``layer_2.severity_rationale`` when the coercion
                    changes the label.

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
            raw_alert, device_context, baseline, user_context, alert_type,
            risk_level=risk_level,
        )
        if mve is not None:
            provider_used = "openai"
        else:
            mve = _generate_llm_anthropic(
                raw_alert, device_context, baseline, user_context, alert_type,
                risk_level=risk_level,
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
    # Layer 1 MUST mention top_category and the top-k SHAP features
    # when shap_context is provided (research_spec §2.module_4).
    #
    # RQ2.b G1+G2 fix: previously injected only top_features[0], leaving
    # the "≥2 features mentioned" alignment target at 0%. Now lists up
    # to the top-3 features so Mode B alignment hits 100% / 100% / 100%
    # on top-1 / ≥2 / all-3 by construction. Word budget impact is ~6-12
    # additional words (feature names are short network/biometric tokens),
    # well under the 150-word Layer 1+2+3 cap enforced post-generation.
    if shap_context:
        top_category = str(shap_context.get("top_category", "")).strip()
        top_features = shap_context.get("top_features") or []
        if top_category == "biometric":
            narrative = shap_context.get(
                "top_feature_narrative", "abnormal biometric reading"
            )
            existing = mve.layer_1.get("deviation_description", "")
            # Also list raw biometric features so the SHAP-alignment audit
            # can verify the top-k were surfaced.
            feats_str = ", ".join(str(f) for f in top_features[:3])
            feat_suffix = f" Driving features: {feats_str}." if feats_str else ""
            mve.layer_1["deviation_description"] = (
                f"{existing} Concurrent clinical anomaly: {narrative} "
                f"deviates from this device's baseline vital signs.{feat_suffix}"
            )
        elif top_category and top_features:
            feats_str = ", ".join(str(f) for f in top_features[:3])
            existing = mve.layer_1.get("deviation_description", "")
            readable = top_category.replace("_", " ")
            mve.layer_1["deviation_description"] = (
                f"{existing} Top signals: {readable} ({feats_str})."
            ).strip()

    # ── MITRE ATT&CK reference (RQ2.e G3) ───────────────────────────────
    # When the alert's attack_category maps to a MITRE technique in
    # config/attack_to_mitre_mapping.yaml, append the technique ID + name
    # to Layer 1. Previously zero narratives referenced MITRE; this
    # closes the spec target of ≥90% Layer 1 MITRE-reference rate.
    attack_category = str(raw_alert.get("attack_category", "")).strip()
    mitre_ref = _lookup_mitre_reference(attack_category)
    if mitre_ref:
        existing = mve.layer_1.get("deviation_description", "")
        # Phase 1.4 — append the plain-language gloss when the YAML entry
        # defines one, so non-security stakeholders aren't left to decode
        # the bare technique ID.
        gloss = mitre_ref.get("plain_gloss", "") if isinstance(mitre_ref, dict) else ""
        gloss_clause = f" — {gloss}" if gloss else ""
        mve.layer_1["deviation_description"] = (
            f"{existing} Consistent with MITRE {mitre_ref['id']} "
            f"({mitre_ref['name']}{gloss_clause})."
        ).strip()

    # ── Phase 1.2 — Parametrize Layer 3 with alert_id + escalation contacts ──
    # Prepend the alert ID and the rendered device type to immediate_action
    # so non-network stakeholders can dial back to a specific record / device
    # instead of guessing. Annotate escalation_path with extensions + SLA
    # from ``ESCALATION_CONTACTS``. Both are no-ops when their inputs are
    # missing (alert_id absent, role unknown) so legacy callers without
    # the new metadata still render the same text.
    try:
        from module5_responses.config import annotate_role as _annotate_role
    except ImportError:
        _annotate_role = None

    alert_id = str(raw_alert.get("alert_id", "")).strip()
    raw_device_type = str(device_context.get("device_type", "")).strip()
    if alert_id or raw_device_type:
        existing_action = mve.layer_3.get("immediate_action", "")
        prefix_bits: list[str] = []
        if alert_id:
            prefix_bits.append(alert_id)
        if raw_device_type and raw_device_type not in ("device", "system"):
            prefix_bits.append(raw_device_type)
        if prefix_bits:
            prefix = " · ".join(prefix_bits)
            mve.layer_3["immediate_action"] = f"[{prefix}] {existing_action}".strip()

    if _annotate_role is not None:
        existing_esc = mve.layer_3.get("escalation_path", "")
        if existing_esc:
            # Escalation strings look like
            #   ``(1) Privacy Officer, (2) Security lead, (3) HR.``
            # Match each ``(N) <Role>`` group up to the next ``, (N+1)``
            # or a trailing period / EOS, then re-emit with the extension
            # / SLA from ``ESCALATION_CONTACTS`` appended to the role text.
            mve.layer_3["escalation_path"] = re.sub(
                r"\((\d+)\)\s*([^()]+?)(?=,\s*\(\d+\)|\.|$)",
                lambda m: f"({m.group(1)}) "
                          f"{_annotate_role(m.group(2).strip().rstrip(',.'))}",
                existing_esc,
            )

    mve.provider = provider_used

    # ── Severity reconciliation (Invariant 6) ──────────────────────────
    # The canonical severity is module 3's ``risk_level``; the rest of
    # the pipeline (response engine, clinician summary template) acts on
    # that tier. When provided, coerce ``layer_2.severity_label`` to it so
    # the MVE cannot disagree with the rest of the alert record.
    if risk_level:
        canonical = str(risk_level).upper()
        if canonical in VALID_SEVERITY:
            original = mve.layer_2.get("severity_label", "")
            if str(original).upper() != canonical:
                logger.info(
                    "MVE severity coerced: %s → %s (provider=%s)",
                    sanitize_for_log(original), canonical, provider_used,
                )
                mve.layer_2["severity_label"] = canonical
                rationale = mve.layer_2.get("severity_rationale", "")
                note = f" (severity normalized to pipeline risk_level={canonical})"
                if note not in rationale:
                    mve.layer_2["severity_rationale"] = f"{rationale}{note}".strip()

    return mve
