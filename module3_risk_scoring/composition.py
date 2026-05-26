"""Composite risk formula + tier assignment.

These two functions are the public-facing surface of Module 3 most
heavily consumed by downstream code (``src/risk_scorer.py``,
``tools/diagnostics/*``).
"""

from __future__ import annotations

import numpy as np

from .config import RISK_THRESHOLDS, WEIGHTS


def compute_composite_risk(
    c_detect: np.ndarray,
    d_crit: np.ndarray,
    s_data: np.ndarray,
    d_clinical_tier: np.ndarray,
    weights: dict | None = None,
) -> np.ndarray:
    """R = w1·C_detect + w2·D_crit + w3·S_data + w4·D_clinical_tier.

    Args:
        c_detect: Cascaded Track A → Track B fused detection score in [0, 1].
        d_crit:   Device criticality × CIA threat interaction in [0, 1].
        s_data:   Data sensitivity in [0, 1].
        d_clinical_tier: Patient-acuity proxy in [0, 1].
        weights:  Optional override of WEIGHTS. Must carry keys w1..w4.

    Returns:
        R in [0, 1] (output is clipped — extreme weights cannot push
        the composite outside the unit interval).
    """
    w = weights or WEIGHTS
    R = (
        w["w1"] * c_detect
        + w["w2"] * d_crit
        + w["w3"] * s_data
        + w["w4"] * d_clinical_tier
    )
    return np.clip(R, 0.0, 1.0)


def assign_risk_levels(
    R: np.ndarray,
    thresholds: dict | None = None,
) -> np.ndarray:
    """Map composite scores to 4 alert tiers using 3 thresholds.

    Args:
        R: Composite risk scores in [0, 1].
        thresholds: Optional dict ``{"CRITICAL": 0.80, "HIGH": 0.60,
            "MEDIUM": 0.40}``. Falls back to module-level
            ``RISK_THRESHOLDS`` when *None*.

    Returns:
        np.ndarray of tier labels — one of ``"CRITICAL"``, ``"HIGH"``,
        ``"MEDIUM"``, ``"LOW"``.
    """
    if thresholds is None:
        t_crit, t_high, t_med = 0.80, 0.60, 0.40
    else:
        t_crit = thresholds.get("CRITICAL", 0.80)
        t_high = thresholds.get("HIGH", 0.60)
        t_med = thresholds.get("MEDIUM", 0.40)

    conditions = [R >= t_crit, R >= t_high, R >= t_med]
    choices = ["CRITICAL", "HIGH", "MEDIUM"]
    return np.select(conditions, choices, default="LOW")


__all__ = ["compute_composite_risk", "assign_risk_levels"]
