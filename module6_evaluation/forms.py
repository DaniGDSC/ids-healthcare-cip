"""Pure helpers behind the dashboard's form + alert-processing logic.

Extracted from ``module6_app.py`` so they're testable without Streamlit.
"""
from __future__ import annotations

import hashlib
import random
from datetime import datetime, timezone

from .constants import _ACTION_PRIORITY


def assign_ab_conditions(n_alerts: int, participant_id: str) -> list[bool]:
    """Return a balanced ``with_xai`` boolean per alert, seeded per participant.

    Half of n_alerts are True (XAI shown), half False (control). Order is
    pseudo-random per participant so different participants see different
    alert orderings — but the same participant gets a reproducible sequence
    across sessions (Y10-related determinism).
    """
    if n_alerts <= 0:
        return []
    n_xai = n_alerts // 2
    pattern = [True] * n_xai + [False] * (n_alerts - n_xai)
    seed_int = int(hashlib.sha256(participant_id.encode()).hexdigest()[:8], 16)
    rng = random.Random(seed_int)
    rng.shuffle(pattern)
    return pattern


def process_alert(sample_index: int, alert_data: dict) -> dict:
    """Roll a Module-5 policy alert up to dashboard-bucket form.

    Produces the compact dict the dashboard's alert-feed table consumes —
    plus the operator-bucket action (isolate / escalate / investigate /
    monitor) derived via :data:`_ACTION_PRIORITY`.
    """
    response = alert_data.get("response", {}) if isinstance(alert_data.get("response"), dict) else {}
    actions = response.get("actions", []) if isinstance(response, dict) else []

    bucket_rank = {"isolate": 4, "escalate": 3, "investigate": 2, "monitor": 1}
    bucket = "monitor"
    for a in actions:
        b = _ACTION_PRIORITY.get(a, "monitor")
        if bucket_rank.get(b, 0) > bucket_rank.get(bucket, 0):
            bucket = b

    return {
        "sample_index": int(sample_index),
        "alert_id": alert_data.get("alert_id", f"ALERT-{sample_index:05d}"),
        "risk_level": str(alert_data.get("risk_level", "UNKNOWN")).upper(),
        "risk_score": float(alert_data.get("risk_score", 0.0)),
        "attack_category": alert_data.get("attack_category", "unknown"),
        "ground_truth": alert_data.get("ground_truth", "unknown"),
        "actions": list(actions),
        "operator_bucket": bucket,
        "device_tier": response.get("device_tier", "unknown") if isinstance(response, dict) else "unknown",
    }


def build_fda_record_for_alert(
    alert: dict,
    *,
    participant_id: str,
    role: str,
    chosen_action: str,
    rationale: str = "",
    confidence: int | None = None,
    decision_time_sec: float | None = None,
) -> dict:
    """Assemble the FDA-style audit record for a participant decision."""
    return {
        "alert_id": alert.get("alert_id", ""),
        "sample_index": int(alert.get("sample_index", -1)),
        "risk_level": str(alert.get("risk_level", "")).upper(),
        "attack_category": alert.get("attack_category", ""),
        "ground_truth": alert.get("ground_truth", "unknown"),
        "reviewer": {
            "participant_id": participant_id,
            "role": role,
            "decided_at": datetime.now(timezone.utc).isoformat(),
        },
        "decision": {
            "chosen_action": chosen_action,
            "rationale": rationale[:500],
            "confidence": int(confidence) if confidence is not None else None,
            "decision_time_sec": float(decision_time_sec)
            if decision_time_sec is not None else None,
        },
        "policy_actions": (
            alert.get("response", {}).get("actions", [])
            if isinstance(alert.get("response"), dict) else []
        ),
    }


__all__ = [
    "assign_ab_conditions",
    "process_alert",
    "build_fda_record_for_alert",
]
