"""Panel 5 — Device Inventory and Behavioral Profiling (Network View).

Expandable device cards with risk indicators, protocol detection,
behavioral profiling, and recommended actions.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional

import streamlit as st

from dashboard.utils.metrics import risk_color

HIPAA_HASH_PREFIX_LENGTH: int = 8

# Device type inference from traffic signature patterns
DEVICE_SIGNATURES = {
    "pulse_oximeter": {
        "pattern": "High-frequency, low-payload",
        "label": "Pulse Oximeter / SpO2",
        "protocol": "BLE",
        "confidence": "medium",
    },
    "ecg_monitor": {
        "pattern": "Continuous stream, periodic",
        "label": "ECG Monitor",
        "protocol": "Wi-Fi",
        "confidence": "medium",
    },
    "infusion_pump": {
        "pattern": "Burst pattern, MQTT",
        "label": "Infusion Pump Controller",
        "protocol": "MQTT",
        "confidence": "low",
    },
    "imaging_device": {
        "pattern": "Irregular, large payload",
        "label": "Imaging Device",
        "protocol": "Wi-Fi",
        "confidence": "low",
    },
}


def _infer_device_type(
    top_features: List[str],
    anomaly_score: float,
) -> Dict[str, str]:
    """Infer device type from traffic signature features.

    Args:
        top_features: Top contributing features for this device.
        anomaly_score: Device anomaly score.

    Returns:
        Dictionary with device type inference.
    """
    biometric = {"Temp", "SpO2", "Pulse_Rate", "Heart_rate",
                 "Resp_Rate", "SYS", "DIA", "ST"}

    bio_count = sum(1 for f in top_features if f in biometric)

    if bio_count >= 2:
        if "SpO2" in top_features or "Pulse_Rate" in top_features:
            return DEVICE_SIGNATURES["pulse_oximeter"]
        if "Heart_rate" in top_features or "ST" in top_features:
            return DEVICE_SIGNATURES["ecg_monitor"]

    if "TotBytes" in top_features and anomaly_score > 0.5:
        return DEVICE_SIGNATURES["imaging_device"]

    return {
        "pattern": "Standard IoMT traffic",
        "label": "Unknown IoMT Device",
        "protocol": "TCP/Wi-Fi",
        "confidence": "low",
    }


def _recommended_action(risk_level: str) -> str:
    """Get recommended action based on risk level."""
    actions = {
        "NORMAL": "Operating within baseline",
        "LOW": "Monitor — minor deviation detected",
        "MEDIUM": "Review device logs",
        "HIGH": "Verify physical device integrity",
        "CRITICAL": "Isolate device — contact IT Admin",
    }
    return actions.get(risk_level, "Unknown")


def render_device_card(
    device: Dict[str, Any],
    index: int,
) -> None:
    """Render a single device card as an expander.

    Args:
        device: Device information dictionary.
        index: Device index for unique keys.
    """
    risk = device.get("risk_level", "NORMAL")
    color = risk_color(risk)
    device_hash = hashlib.sha256(
        str(device.get("device_id", index)).encode(),
    ).hexdigest()[:HIPAA_HASH_PREFIX_LENGTH]

    protocol = device.get("protocol", "TCP/Wi-Fi")
    status = "ANOMALOUS" if risk in ("HIGH", "CRITICAL") else "NORMAL"

    header = (
        f"[{risk}] {device_hash}... | {protocol} | {status}"
    )

    with st.expander(header, expanded=(risk in ("CRITICAL",))):
        top_features = device.get("top_features", [])
        feat_names = [f.get("feature", "") if isinstance(f, dict) else str(f)
                      for f in top_features[:3]]
        score = device.get("anomaly_score", 0)

        dev_info = _infer_device_type(feat_names, score)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Device Type:** {dev_info['label']}")
            st.markdown(f"**Protocol:** {dev_info['protocol']}")
            st.markdown(f"**Confidence:** {dev_info['confidence']}")
            st.markdown(f"**Status:** {status}")
        with col2:
            st.markdown(f"**Anomaly Score:** {score:.4f}")
            st.markdown(f"**Risk Level:** {risk}")
            alert_count = device.get("alert_count", 0)
            st.markdown(f"**Alert Count:** {alert_count}")

        if feat_names:
            st.markdown("**Top SHAP Features:**")
            for i, f in enumerate(top_features[:3]):
                if isinstance(f, dict):
                    name = f.get("feature", "")
                    val = f.get("shap_value", 0)
                    st.text(f"  {i + 1}. {name}: {val:+.5f}")
                else:
                    st.text(f"  {i + 1}. {f}")

        action = _recommended_action(risk)
        if risk == "CRITICAL":
            st.error(f"Action: {action}")
        elif risk == "HIGH":
            st.warning(f"Action: {action}")
        else:
            st.success(f"Action: {action}")

        st.caption(
            "Device type inferred from traffic signature — "
            "not verified against asset registry"
        )


def render(gt: Dict[str, Any]) -> None:
    """Render the full Device Inventory panel.

    Args:
        gt: Ground truth data.
    """
    st.header("Device Inventory & Behavioral Profiling")

    explanation = gt.get("explanation", {})
    if explanation.get("status") not in ("VERIFIED", "PRESENT_UNVERIFIED"):
        st.info("Device profiling not available — explanation artifact not found")
        return

    from dashboard.utils.loader import load_explanation_report
    report = load_explanation_report()

    if not report or "explanations" not in report:
        st.info("No device data available")
        return

    alerts = report["explanations"]

    # Group by simulated device (use sample_index as device proxy)
    # Sort by risk level severity
    risk_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "NORMAL": 4}
    sorted_alerts = sorted(
        alerts,
        key=lambda a: risk_order.get(a.get("risk_level", "NORMAL"), 5),
    )

    st.caption(f"Showing {len(sorted_alerts)} device profiles, sorted by risk level")

    for i, alert in enumerate(sorted_alerts[:20]):
        render_device_card(alert, i)
