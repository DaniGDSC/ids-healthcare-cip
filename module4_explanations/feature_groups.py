"""Feature-group narrative mapping (SHAP stability defense).

Maps individual SHAP feature names to clinically-meaningful narrative
categories. This absorbs within-category feature swaps (e.g., DIntPkt ↔
Sport) that account for most SHAP instability, producing stable
narratives even when the exact top-1 feature changes.

Used by:
  - online ``AlertExplainer._clinician_nlg`` (per-alert NLG)
  - offline ``build_clinician_summaries`` (batch NLG)
  - both via ``module4_explanations.nlg``
"""

from __future__ import annotations


_FEATURE_GROUPS: dict[str, tuple[str, str]] = {
    # Network timing anomalies
    "DIntPkt":    ("unusual network packet timing",     "network_timing"),
    "SIntPkt":    ("unusual network packet timing",     "network_timing"),
    "SIntPktAct": ("unusual network packet timing",     "network_timing"),
    "Dur":        ("abnormal connection duration",      "network_timing"),
    # Network protocol anomalies
    "Sport":      ("unexpected network port activity",  "network_protocol"),
    "Flgs":       ("abnormal protocol flags",           "network_protocol"),
    # Transfer volume anomalies
    "SrcBytes":   ("unusual data transfer volume",      "network_volume"),
    "DstBytes":   ("unusual data transfer volume",      "network_volume"),
    "TotBytes":   ("unusual data transfer volume",      "network_volume"),
    "SrcLoad":    ("abnormal network load",             "network_volume"),
    "DstLoad":    ("abnormal network load",             "network_volume"),
    "Load":       ("abnormal network load",             "network_volume"),
    # Packet structure anomalies
    "sMaxPktSz":  ("unusual packet structure",          "network_packet"),
    "dMaxPktSz":  ("unusual packet structure",          "network_packet"),
    "sMinPktSz":  ("unusual packet structure",          "network_packet"),
    "pSrcLoss":   ("abnormal packet loss",              "network_loss"),
    "pDstLoss":   ("abnormal packet loss",              "network_loss"),
    # Biometric anomalies
    "Temp":       ("abnormal temperature reading",      "biometric"),
    "SpO2":       ("abnormal oxygen saturation",        "biometric"),
    "Pulse_Rate": ("abnormal pulse rate",               "biometric"),
    "SYS":        ("abnormal blood pressure",           "biometric"),
    "DIA":        ("abnormal blood pressure",           "biometric"),
    "Heart_rate": ("abnormal heart rate",               "biometric"),
    "Resp_Rate":  ("abnormal respiratory rate",         "biometric"),
    "ST":         ("abnormal ST segment",               "biometric"),
}


def _feature_to_narrative(feature_name: str) -> tuple[str, str]:
    """Map a feature name to ``(narrative_phrase, category)``.

    Unknown features fall back to ``(feature_name, "unknown")`` so the
    caller's narrative still renders something (the raw feature name
    instead of a clinical phrase).
    """
    return _FEATURE_GROUPS.get(feature_name, (feature_name, "unknown"))


__all__ = ["_FEATURE_GROUPS", "_feature_to_narrative"]
