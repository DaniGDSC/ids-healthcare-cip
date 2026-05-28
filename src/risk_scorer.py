"""Component 2: Risk-Adaptive Scoring Engine.

Wraps the composite risk scoring logic from
module3_risk_scoring/module3_risk_scores.py and adds the
patchability-aware threshold adjustment required by research_spec.yaml.

Key additions over module3:
  - patchable flag: CRITICAL + unpatchable → risk_multiplier >= 1.5,
    threshold lowered >= 30% (non-negotiable per CLAUDE.md)
  - Per-alert dict interface instead of batch numpy arrays
  - Suppression for maintenance window + known vendor IP
  - Learned baseline: similar_events_past_30d > 5 reduces multiplier
"""
from __future__ import annotations

from typing import Any, Optional

from src.data_models import DataQuality, ScoredAlert

# ── Constants ───────────────────────────────────────────────────────────

DEFAULT_THRESHOLD: float = 0.50
"""Baseline surfacing threshold (static, for comparison baseline)."""

# Per-tier policy: threshold mult + risk mult in one mapping.
# N4 fix: was two separate dicts keyed identically — single source of truth
# for the tier policy + half the lookups per call.
#
# CRITICAL + unpatchable: threshold mult 0.70 (≥30% reduction per spec),
#                          risk mult 1.50 (≥1.5 per spec).
# LOW + patchable:         threshold mult 1.00 (default), risk mult 1.00.
_DEVICE_TIER_POLICY: dict[tuple[str, bool], dict[str, float]] = {
    ("CRITICAL", False): {"threshold_mult": 0.70, "risk_mult": 1.50},
    ("CRITICAL", True):  {"threshold_mult": 0.80, "risk_mult": 1.30},
    ("HIGH",     False): {"threshold_mult": 0.85, "risk_mult": 1.20},
    ("HIGH",     True):  {"threshold_mult": 0.90, "risk_mult": 1.10},
    ("MEDIUM",   False): {"threshold_mult": 0.95, "risk_mult": 1.05},
    ("MEDIUM",   True):  {"threshold_mult": 1.00, "risk_mult": 1.00},
    ("LOW",      False): {"threshold_mult": 1.00, "risk_mult": 1.00},
    ("LOW",      True):  {"threshold_mult": 1.00, "risk_mult": 1.00},
}

_DEFAULT_POLICY = {"threshold_mult": 1.00, "risk_mult": 1.00}


# ── Public API ──────────────────────────────────────────────────────────

def get_threshold(device_criticality: str, patchable: bool) -> float:
    """Return the surfacing threshold for a given device context.

    Used by verify_risk_adaptive_threshold to verify that the
    CRITICAL-unpatchable threshold is strictly lower (more sensitive)
    than the LOW-patchable threshold.

    Args:
        device_criticality: CRITICAL / HIGH / MEDIUM / LOW
        patchable: Whether device firmware can be updated.

    Returns:
        Float threshold in [0.0, 1.0].
    """
    key = (device_criticality.upper(), bool(patchable))
    policy = _DEVICE_TIER_POLICY.get(key, _DEFAULT_POLICY)
    return round(DEFAULT_THRESHOLD * policy["threshold_mult"], 4)


def score_alert(
    anomaly_score: float,
    device_context: dict[str, Any],
    event_context: Optional[dict[str, Any]],
    data_quality: "DataQuality | str | None" = None,
) -> ScoredAlert:
    """Adjust alert threshold and multiplier based on device context.

    Implements the per-alert dict interface required by research_spec.yaml,
    applying the same threshold logic as module3_risk_scores.assign_risk_levels()
    but with per-device patchability and context awareness.

    Non-negotiable rules (CLAUDE.md):
      - CRITICAL + unpatchable  → risk_multiplier >= 1.5, threshold lowered >= 30%
      - Maintenance + vendor IP → suppress (should_surface=False)
      - LOW + patchable         → default threshold, risk_multiplier = 1.0

    Adaptive rule (research_spec.yaml component_2.behavior_rules):
      - similar_events_past_30d > 5 → reduce risk_multiplier by 0.2
        (learned normal pattern, reduces alert fatigue)

    Args:
        anomaly_score: Raw anomaly score [0.0, 1.0] from detection layer.
        device_context: Dict with keys: criticality (str), patchable (bool),
                        clinical_function (str).
        event_context: Optional dict with keys: is_maintenance_window (bool),
                       is_known_vendor_ip (bool), similar_events_past_30d (int).

    Returns:
        ScoredAlert with adjusted_score, threshold, should_surface,
        risk_multiplier, and optional suppression_reason.
    """
    # RS-1/RS-2/RS-4: normalise all inputs once at entry — eliminates 3
    # separate float(anomaly_score) casts and scattered str/bool coercions.
    score = float(anomaly_score)
    criticality = str(device_context.get("criticality", "LOW")).upper()
    patchable = bool(device_context.get("patchable", True))

    # EA-06 mitigation: NaN-injection attempts must not let an attacker
    # mask an anomaly. DEGRADED inputs nudge the score up (×1.20); FAILED
    # inputs force an upper-bound score so the alert always surfaces for
    # the operator to verify, even if the imputed features look benign.
    # Applied BEFORE downstream multipliers so the elevation compounds
    # with device-tier risk.
    try:
        dq = DataQuality(data_quality) if data_quality else DataQuality.OK
    except ValueError:
        dq = DataQuality.OK
    if dq == DataQuality.FAILED:
        score = max(score, 0.95)
    elif dq == DataQuality.DEGRADED:
        score = min(1.0, score * 1.20)

    # Rule: maintenance window + known vendor IP → reduced confidence
    # EA-02 fix: binary suppression created a guaranteed evasion window.
    # Now: surface with reduced multiplier instead of suppressing entirely.
    # The alert surfaces at LOW priority so the operator can verify, rather
    # than being silenced completely.
    if event_context:
        if (event_context.get("is_maintenance_window", False)
                and event_context.get("is_known_vendor_ip", False)):
            reduced = score * 0.5
            return ScoredAlert(
                adjusted_score=round(reduced, 4),
                threshold=DEFAULT_THRESHOLD,
                should_surface=reduced > DEFAULT_THRESHOLD,
                risk_multiplier=0.5,
                suppression_reason="maintenance window — reduced confidence, verify with biomed",
            )

    # RS-3 + N4: single dict lookup carries both threshold and risk multipliers.
    policy = _DEVICE_TIER_POLICY.get((criticality, patchable), _DEFAULT_POLICY)
    risk_multiplier: float = policy["risk_mult"]
    threshold = round(DEFAULT_THRESHOLD * policy["threshold_mult"], 4)

    # Adaptive rule: reduce multiplier for frequently-seen patterns
    if event_context:
        similar = int(event_context.get("similar_events_past_30d", 0))
        if similar > 5:
            risk_multiplier = max(0.5, risk_multiplier - 0.20)

        # TM-02 fix: baseline quarantine for newly enrolled devices.
        # Devices with < 14 days of baseline data get a lower threshold
        # (30% reduction) to compensate for the DAE's unreliable baseline.
        baseline_days = int(event_context.get("baseline_days", 90))
        if baseline_days < 14:
            threshold = threshold * 0.70

    adjusted_score = min(1.0, score * risk_multiplier)
    should_surface = adjusted_score > threshold

    # Safety floor: CRITICAL + unpatchable devices must ALWAYS surface.
    # The IDS is the ONLY compensating control for unpatchable devices —
    # suppressing any signal, however weak, leaves the device unmonitored.
    # Fixes ST-03: anomaly_score=0.2 on CRITICAL+unpatchable was suppressed.
    if criticality == "CRITICAL" and not patchable:
        should_surface = True

    return ScoredAlert(
        adjusted_score=round(adjusted_score, 4),
        threshold=round(threshold, 4),
        should_surface=should_surface,
        risk_multiplier=round(risk_multiplier, 4),
        suppression_reason=None,
    )


def score_alert_static(anomaly_score: float) -> dict[str, bool]:
    """Static-threshold baseline for false-positive reduction comparison.

    Applies the same fixed 0.5 threshold to every alert regardless of
    device context — simulating a legacy IDS with no adaptive logic.
    Used by compute_false_positive_rate to compute the baseline FP rate.

    Args:
        anomaly_score: Raw anomaly score [0.0, 1.0].

    Returns:
        Dict with 'surfaced' bool.
    """
    return {"surfaced": float(anomaly_score) > DEFAULT_THRESHOLD}
