"""Performance and latency computation utilities.

Provides inference timing, engine health computation,
and compliance rate calculation for dashboard panels.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

# Latency thresholds (milliseconds)
LATENCY_GREEN: float = 80.0
LATENCY_AMBER: float = 100.0  # 80% of SLA

# SLA constants
CRITICAL_LATENCY_BUDGET: float = 1.0  # seconds
SHAP_TIMEOUT_SECONDS: float = 30.0
MAX_ALERT_DISPLAY: int = 50


def latency_color(latency_ms: float) -> str:
    """Return color hex for a latency value.

    Args:
        latency_ms: Latency in milliseconds.

    Returns:
        Color hex string.
    """
    if latency_ms < LATENCY_GREEN:
        return "#2ecc71"  # green
    elif latency_ms < LATENCY_AMBER:
        return "#f39c12"  # amber
    else:
        return "#e74c3c"  # red


def state_color(state: str) -> str:
    """Return color hex for an engine state.

    Args:
        state: Engine state string (UP/DEGRADED/DOWN/etc).

    Returns:
        Color hex string.
    """
    colors = {
        "UP": "#2ecc71",
        "STARTING": "#3498db",
        "DEGRADED": "#f39c12",
        "DOWN": "#e74c3c",
        "UNKNOWN": "#95a5a6",
    }
    return colors.get(state, "#95a5a6")


def risk_color(level: str) -> str:
    """Return color hex for a risk level.

    Args:
        level: Risk level string.

    Returns:
        Color hex string.
    """
    colors = {
        "NORMAL": "#2ecc71",
        "LOW": "#3498db",
        "MEDIUM": "#f39c12",
        "HIGH": "#e67e22",
        "CRITICAL": "#e74c3c",
    }
    return colors.get(level, "#95a5a6")


def severity_color(severity: int) -> str:
    """Return color hex for a clinical severity level (1–5).

    Args:
        severity: Clinical severity integer.

    Returns:
        Color hex string.
    """
    colors = {
        1: "#2ecc71",  # ROUTINE — green
        2: "#3498db",  # ADVISORY — blue
        3: "#f39c12",  # URGENT — amber
        4: "#e67e22",  # EMERGENT — orange
        5: "#e74c3c",  # CRITICAL — red
    }
    return colors.get(severity, "#95a5a6")


def severity_label(severity: int) -> str:
    """Return display label for a clinical severity level.

    Args:
        severity: Clinical severity integer.

    Returns:
        Human-readable severity label.
    """
    labels = {
        1: "ROUTINE",
        2: "ADVISORY",
        3: "URGENT",
        4: "EMERGENT",
        5: "CRITICAL",
    }
    return labels.get(severity, "UNKNOWN")


def compute_engine_health(
    monitoring_log: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Compute current health status for each engine.

    Args:
        monitoring_log: List of state transition entries.

    Returns:
        List of engine health dictionaries.
    """
    engine_names = {
        "phase2_detection": "Data Processing",
        "phase3_classification": "Classification",
        "phase4_risk_adaptive": "Risk-Adaptive",
        "phase5_explanation": "Explanation",
        "phase6_notification": "Notification",
    }

    # Extract latest state per engine
    latest: Dict[str, Dict[str, Any]] = {}
    latencies: Dict[str, List[float]] = {}

    for entry in monitoring_log:
        eid = entry.get("engine_id", "")
        latest[eid] = {
            "state": entry.get("new_state", "UNKNOWN"),
            "timestamp": entry.get("timestamp", ""),
            "reason": entry.get("reason", ""),
        }

        # Parse latency from reason
        reason = entry.get("reason", "")
        if "latency=" in reason:
            try:
                lat_str = reason.split("latency=")[1].split("ms")[0]
                latencies.setdefault(eid, []).append(float(lat_str))
            except (ValueError, IndexError):
                pass

    results = []
    for eid, name in engine_names.items():
        info = latest.get(eid, {"state": "UNKNOWN", "timestamp": "", "reason": ""})
        lats = latencies.get(eid, [])
        p95 = 0.0
        if lats:
            sorted_lats = sorted(lats)
            idx = min(int(len(sorted_lats) * 0.95), len(sorted_lats) - 1)
            p95 = sorted_lats[idx]

        results.append({
            "engine": name,
            "engine_id": eid,
            "status": info["state"],
            "latency_p95": round(p95, 1),
            "last_heartbeat": info["timestamp"],
            "alerts": 0,
        })

    return results


