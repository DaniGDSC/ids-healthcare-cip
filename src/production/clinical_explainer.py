"""Clinical explainability — 7 methods for human-interpretable explanations.

Transforms raw model outputs (feature importance, timestep weights, risk
levels) into clinician-readable narratives. No LLM — template-based NLG
with device-specific context and attack pattern matching.

Methods:
  1. Counterfactual: "what would need to change for alert to clear"
  2. Natural language: feature importance → readable sentences
  3. Attack pattern library: match against known IoMT attack signatures
  4. Temporal narration: "anomaly began at timestep 14, peaked at 17"
  5. Confidence-aware: MC Dropout variance → confidence labels
  6. Device-specific templates: pump-specific vs sensor-specific guidance
  7. Comparative: "why HIGH not MEDIUM" — risk chain justification
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

_BIOMETRIC = {"Temp", "SpO2", "Pulse_Rate", "SYS", "DIA", "Heart_rate", "Resp_Rate", "ST"}

_BIOMETRIC_FRIENDLY = {
    "Temp": "body temperature",
    "SpO2": "blood oxygen",
    "Pulse_Rate": "pulse rate",
    "SYS": "systolic blood pressure",
    "DIA": "diastolic blood pressure",
    "Heart_rate": "heart rate",
    "Resp_Rate": "respiratory rate",
    "ST": "ST segment",
}

_NETWORK_FRIENDLY = {
    "SrcBytes": "outbound data volume",
    "DstBytes": "inbound data volume",
    "SrcLoad": "outbound traffic load",
    "DstLoad": "inbound traffic load",
    "SIntPkt": "send interval",
    "DIntPkt": "receive interval",
    "SIntPktAct": "active send interval",
    "sMaxPktSz": "max outbound packet size",
    "dMaxPktSz": "max inbound packet size",
    "sMinPktSz": "min outbound packet size",
    "Dur": "connection duration",
    "TotBytes": "total data volume",
    "Load": "total traffic load",
    "pSrcLoss": "outbound packet loss",
    "pDstLoss": "inbound packet loss",
    "Packet_num": "packet count",
}


def _friendly_name(feature: str) -> str:
    return _BIOMETRIC_FRIENDLY.get(feature, _NETWORK_FRIENDLY.get(feature, feature))


# ═══════════════════════════════════════════════════════════════════
# Method 2: Natural Language Generation
# ═══════════════════════════════════════════════════════════════════

def generate_narrative(
    top_features: List[Dict[str, Any]],
    risk_level: str,
    device_id: str = "",
) -> str:
    """Convert feature importance list into human-readable narrative."""
    if not top_features:
        return ""

    total_imp = sum(abs(f.get("importance", f.get("shap_value", 0)) or 0) for f in top_features)
    if total_imp == 0:
        return ""

    bio = [f for f in top_features if f.get("feature", "") in _BIOMETRIC]
    net = [f for f in top_features if f.get("feature", "") not in _BIOMETRIC]

    parts = []

    if bio and net:
        bio_names = ", ".join(_friendly_name(f["feature"]) for f in bio[:3])
        net_names = ", ".join(_friendly_name(f["feature"]) for f in net[:3])
        parts.append(
            f"Both vital sign readings ({bio_names}) and network traffic "
            f"({net_names}) show anomalies, suggesting coordinated device compromise."
        )
    elif bio:
        bio_names = ", ".join(_friendly_name(f["feature"]) for f in bio[:3])
        parts.append(
            f"Vital sign readings are abnormal ({bio_names}). "
            f"Check sensor connections and compare with manual readings."
        )
    elif net:
        net_names = ", ".join(_friendly_name(f["feature"]) for f in net[:3])
        parts.append(
            f"Network traffic is anomalous ({net_names}). "
            f"Device may be scanning the network or data may be exfiltrated."
        )

    # Add contribution percentages for top feature
    top = top_features[0]
    top_name = _friendly_name(top.get("feature", ""))
    top_pct = abs(top.get("importance", 0) or 0) / max(total_imp, 1e-9) * 100
    if top_pct > 30:
        parts.append(f"Primary driver: {top_name} ({top_pct:.0f}% of detection signal).")

    return " ".join(parts)


# ═══════════════════════════════════════════════════════════════════
# Method 3: Attack Pattern Library
# ═══════════════════════════════════════════════════════════════════

_ATTACK_SIGNATURES: Dict[str, Dict[str, tuple]] = {
    "Reconnaissance scan": {
        "Packet_num": (">", 1.5),
        "SrcBytes": ("<", 0.5),
        "Dur": ("<", 0.5),
    },
    "Data exfiltration": {
        "SrcBytes": (">", 2.0),
        "Dur": (">", 1.5),
        "DstBytes": ("<", 0.5),
    },
    "Biometric tampering": {
        "SpO2": (">", 1.5),
        "Heart_rate": (">", 1.5),
    },
    "DoS / flooding": {
        "Packet_num": (">", 2.5),
        "SrcBytes": (">", 2.0),
        "Load": (">", 2.0),
    },
    "Command injection": {
        "SrcBytes": (">", 1.0),
        "DstBytes": (">", 1.0),
        "Dur": (">", 2.0),
    },
}


def match_attack_patterns(
    features: np.ndarray,
    feature_names: List[str],
) -> List[Dict[str, Any]]:
    """Match current feature vector against known IoMT attack signatures.

    Returns list of matches sorted by similarity score.
    """
    matches = []
    for name, signature in _ATTACK_SIGNATURES.items():
        matched = 0
        total = len(signature)
        for feat, (op, threshold) in signature.items():
            if feat not in feature_names:
                continue
            idx = feature_names.index(feat)
            val = abs(float(features[idx]))
            if op == ">" and val > threshold:
                matched += 1
            elif op == "<" and val < threshold:
                matched += 1
        similarity = matched / max(total, 1)
        if similarity > 0.4:
            matches.append({
                "attack_type": name,
                "similarity": round(similarity, 2),
                "matched_features": matched,
                "total_features": total,
            })
    matches.sort(key=lambda x: -x["similarity"])
    return matches


# ═══════════════════════════════════════════════════════════════════
# Method 4: Temporal Narration
# ═══════════════════════════════════════════════════════════════════

def narrate_temporal(timestep_importance: List[float]) -> str:
    """Convert timestep importance array into temporal narrative."""
    if not timestep_importance or len(timestep_importance) < 2:
        return ""

    weights = np.array(timestep_importance)
    peak = int(np.argmax(weights))
    mean_w = float(np.mean(weights))

    # Find onset: first timestep above mean
    onset = 0
    for i, w in enumerate(weights):
        if w > mean_w:
            onset = i
            break

    n = len(weights)
    peak_val = float(weights[peak])

    if onset > n * 0.6:
        pattern = f"Anomaly began late in the window (timestep {onset}/{n}) — sudden event in recent flows."
    elif onset < n * 0.2:
        pattern = f"Anomaly present from the start of the window — persistent pattern across all {n} flows."
    else:
        pattern = f"Anomaly builds gradually from timestep {onset}, suggesting escalating activity."

    pattern += f" Peak intensity at timestep {peak} (weight {peak_val:.4f})."

    # Check if concentrated or distributed
    top_3 = np.sort(weights)[-3:]
    concentration = float(np.sum(top_3) / max(np.sum(weights), 1e-9))
    if concentration > 0.6:
        pattern += " Signal is concentrated in a few timesteps — likely a burst event."
    else:
        pattern += " Signal is spread across the window — likely sustained activity."

    return pattern


# ═══════════════════════════════════════════════════════════════════
# Method 5: Confidence-Aware Labels
# ═══════════════════════════════════════════════════════════════════

def compute_confidence(
    importances_runs: List[np.ndarray],
) -> List[Dict[str, Any]]:
    """Compute confidence from multiple MC Dropout runs.

    Args:
        importances_runs: List of feature importance arrays (one per run).

    Returns:
        Per-feature confidence: mean, std, confidence label.
    """
    if not importances_runs or len(importances_runs) < 2:
        return []

    stacked = np.array(importances_runs)
    mean_imp = np.mean(stacked, axis=0)
    std_imp = np.std(stacked, axis=0)

    results = []
    for i in range(len(mean_imp)):
        cv = float(std_imp[i] / max(abs(mean_imp[i]), 1e-9))
        if cv < 0.3:
            label = "HIGH"
        elif cv < 0.6:
            label = "MEDIUM"
        else:
            label = "LOW"
        results.append({
            "mean": round(float(mean_imp[i]), 6),
            "std": round(float(std_imp[i]), 6),
            "confidence": label,
        })
    return results


# ═══════════════════════════════════════════════════════════════════
# Method 6: Device-Specific Templates
# ═══════════════════════════════════════════════════════════════════

_DEVICE_TEMPLATES: Dict[str, Dict[str, str]] = {
    "infusion_pump": {
        "SrcBytes": "Pump sending more data than expected — possible dosage data exfiltration",
        "DstBytes": "Pump receiving unexpected commands — possible remote dosage manipulation",
        "Dur": "Unusually long connections — possible command injection attempt",
        "Packet_num": "Abnormal packet count — possible network scanning via pump",
        "SpO2": "SpO2 readings from pump patient are anomalous",
        "Heart_rate": "Heart rate from pump patient is anomalous",
        "action": "Verify current infusion rate matches prescription order. "
                  "Check pump display for error codes.",
    },
    "ecg_monitor": {
        "SrcBytes": "ECG data volume increased — possible waveform redirection",
        "DstBytes": "Monitor receiving unexpected data — possible spoofing",
        "Packet_num": "Unusual packet count — possible network scanning via monitor",
        "Heart_rate": "Heart rate readings are anomalous — verify against physical assessment",
        "ST": "ST segment data is abnormal — could be sensor issue or data tampering",
        "action": "Verify ECG waveform displays correctly on bedside monitor. "
                  "Compare with manual pulse check.",
    },
    "pulse_oximeter": {
        "SpO2": "Blood oxygen readings are anomalous — check probe placement",
        "Pulse_Rate": "Pulse rate readings are abnormal — compare with manual count",
        "SrcBytes": "Oximeter sending unusual data volume",
        "action": "Check probe placement. Verify SpO2 reading against patient appearance "
                  "(cyanosis, respiratory effort).",
    },
    "temperature_sensor": {
        "Temp": "Temperature readings are anomalous — check sensor placement",
        "SrcBytes": "Sensor sending unusual data volume",
        "action": "Check sensor placement. Verify with manual thermometer.",
    },
}


def get_device_explanation(
    device_id: str,
    top_features: List[Dict[str, Any]],
) -> Dict[str, str]:
    """Get device-specific explanation for top features.

    Returns dict with 'details' (per-feature explanations) and 'action' (recommended check).
    """
    # Match device_id prefix to template
    template = None
    for dev_type, tmpl in _DEVICE_TEMPLATES.items():
        if dev_type in device_id.lower():
            template = tmpl
            break
    if template is None:
        template = {}

    details = []
    for f in top_features[:5]:
        feat = f.get("feature", "")
        if feat in template:
            details.append(template[feat])
        else:
            friendly = _friendly_name(feat)
            details.append(f"{friendly} is anomalous")

    return {
        "details": details,
        "action": template.get("action", "Monitor device output. Contact clinical IT if readings appear abnormal."),
    }


# ═══════════════════════════════════════════════════════════════════
# Method 7: Comparative Explanation ("Why THIS level?")
# ═══════════════════════════════════════════════════════════════════

def explain_risk_chain(
    result: Dict[str, Any],
) -> List[str]:
    """Explain why the system chose this risk level, step by step.

    Returns list of human-readable justification steps.
    """
    steps = []
    risk = result.get("risk_level", "NORMAL")
    score = result.get("anomaly_score", 0)
    threshold = result.get("threshold", 0)
    distance = result.get("distance", 0)
    attn = result.get("attention_flag", False)
    cia_max = result.get("cia_max_dimension", "")
    cia_scores = result.get("cia_scores", {})
    severity = result.get("clinical_severity", 1)
    safety = result.get("patient_safety_flag", False)
    device_action = result.get("device_action", "none")
    percentile = result.get("percentile")

    # Step 1: Model output
    if percentile is not None:
        steps.append(f"Model score: {score:.4f} (percentile: {percentile:.1f})")
    else:
        steps.append(f"Model score: {score:.4f} (threshold: {threshold:.4f}, distance: {distance:.4f})")

    # Step 2: Risk classification
    if risk == "NORMAL":
        steps.append("Risk: NORMAL — score below detection threshold.")
    elif risk == "LOW":
        steps.append("Risk: LOW — score slightly elevated but below investigation threshold.")
    elif risk == "MEDIUM":
        if attn:
            steps.append("Risk: MEDIUM — attention anomaly detected (potential novel threat).")
        else:
            steps.append("Risk: MEDIUM — score above detection boundary. Investigation recommended.")
    elif risk == "HIGH":
        steps.append("Risk: HIGH — score significantly elevated. Active investigation required.")
    elif risk == "CRITICAL":
        steps.append("Risk: CRITICAL — both network and biometric anomalies confirmed.")

    # Step 3: CIA escalation
    if cia_scores:
        max_dim = {"C": "Confidentiality", "I": "Integrity", "A": "Availability"}.get(cia_max, cia_max)
        max_val = max(cia_scores.values()) if cia_scores else 0
        if max_val >= 0.7:
            steps.append(f"CIA escalation: {max_dim} score ({max_val:.2f}) triggered risk escalation.")
        else:
            steps.append(f"CIA: Primary concern is {max_dim} ({max_val:.2f}). No escalation.")

    # Step 4: Clinical severity
    sev_name = result.get("clinical_severity_name", "")
    if severity >= 4:
        steps.append(f"Clinical severity: {sev_name} — immediate response required.")
    elif severity >= 3:
        steps.append(f"Clinical severity: {sev_name} — investigation within 60 minutes.")
    else:
        steps.append(f"Clinical severity: {sev_name}.")

    # Step 5: Safety flag
    if safety:
        steps.append("PATIENT SAFETY FLAG: Safety-critical device + elevated risk → clinical team notified.")

    # Step 6: Device action
    if device_action == "isolate_network":
        steps.append("Device has been isolated from network. Manual monitoring required.")
    elif device_action == "restrict_network":
        steps.append("Device network access restricted. Monitor for degraded readings.")

    return steps


# ═══════════════════════════════════════════════════════════════════
# Method 1: Counterfactual Explanation
# ═══════════════════════════════════════════════════════════════════

def generate_counterfactual(
    top_features: List[Dict[str, Any]],
    risk_level: str,
) -> str:
    """Generate counterfactual: what would need to change for alert to clear."""
    if not top_features or risk_level in ("NORMAL", "LOW"):
        return ""

    # The top feature with highest importance is the most "changeable"
    top = top_features[0]
    feat = _friendly_name(top.get("feature", ""))
    imp = abs(top.get("importance", 0) or 0)

    if imp > 0:
        return (
            f"To clear this alert, {feat} would need to return to normal levels. "
            f"This is the primary driver ({imp:.4f} importance). "
            f"If {feat} normalizes, risk would likely decrease to NORMAL/LOW."
        )
    return ""


# ═══════════════════════════════════════════════════════════════════
# Combined Clinical Explanation
# ═══════════════════════════════════════════════════════════════════

def build_clinical_explanation(
    result: Dict[str, Any],
    feature_names: Optional[List[str]] = None,
    raw_features: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Build comprehensive clinical explanation from all 7 methods.

    Args:
        result: Full prediction result dict from inference service.
        feature_names: List of 24 feature names.
        raw_features: Raw feature vector (24-dim) for pattern matching.

    Returns:
        Dict with all explanation components.
    """
    explanation = result.get("explanation") or {}
    if not isinstance(explanation, dict):
        explanation = {}
    top_features = explanation.get("top_features", [])
    timesteps = explanation.get("timestep_importance", [])
    risk = result.get("risk_level", "NORMAL")
    device_id = result.get("device_id", "generic_iomt_sensor")

    clinical = {
        # Method 2: Natural language
        "narrative": generate_narrative(top_features, risk, device_id),

        # Method 4: Temporal narration
        "temporal_narrative": narrate_temporal(timesteps),

        # Method 7: Comparative (risk chain)
        "risk_chain": explain_risk_chain(result),

        # Method 1: Counterfactual
        "counterfactual": generate_counterfactual(top_features, risk),

        # Method 6: Device-specific
        "device_explanation": get_device_explanation(device_id, top_features),
    }

    # Method 3: Attack pattern matching (requires raw features)
    if raw_features is not None and feature_names is not None:
        clinical["attack_patterns"] = match_attack_patterns(raw_features, feature_names)
    else:
        clinical["attack_patterns"] = []

    return clinical
