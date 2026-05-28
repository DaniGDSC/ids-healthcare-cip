"""Composite risk formula + tier assignment.

These two functions are the public-facing surface of Module 3 most
heavily consumed by downstream code (``src/risk_scorer.py``,
``tools/diagnostics/*``).
"""

from __future__ import annotations

import numpy as np

from .config import MIN_DETECTION_GATE, RISK_THRESHOLDS, WEIGHTS


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
    *,
    c_detect: np.ndarray | None = None,
    detection_gate: float | None = None,
) -> np.ndarray:
    """Map composite scores to 5 alert tiers using 4 thresholds.

    Tiers (descending severity):
      - ``CRITICAL`` when ``R >= 0.80``
      - ``HIGH``     when ``R >= 0.60``
      - ``MEDIUM``   when ``R >= 0.40``
      - ``LOW``      when ``R >= 0.30``
      - ``NORMAL``   otherwise (anything below the LOW threshold,
                     or — when ``c_detect`` is supplied — any sample
                     whose detector confidence is below ``detection_gate``)

    Args:
        R: Composite risk scores in [0, 1].
        thresholds: Optional dict ``{"CRITICAL": 0.80, "HIGH": 0.60,
            "MEDIUM": 0.40, "LOW": 0.30}``. Falls back to the canonical
            ``RISK_THRESHOLDS`` constant when *None*.
        c_detect: Optional cascaded-detection score array aligned with
            ``R``. When provided, samples with ``c_detect <
            detection_gate`` are forced to NORMAL. This implements
            Phase B of the formula fix — a sample with negligible
            detector signal must not be promoted to LOW by
            context-component weight alone.
        detection_gate: Threshold for the detection gate. Defaults to
            ``MIN_DETECTION_GATE`` (0.02) when ``c_detect`` is given
            and this is None. Ignored when ``c_detect`` is None, so
            callers that only want the tier table can omit both.

    Returns:
        np.ndarray of tier labels — one of ``"CRITICAL"``, ``"HIGH"``,
        ``"MEDIUM"``, ``"LOW"``, ``"NORMAL"``.

    The detection gate's empirical rationale is recorded in
    ``results/formula_comparison.json``: on the test split it cuts
    ~2000 context-driven false alerts while dropping only 12 attacks
    whose model probability is already below the XGBoost decision
    threshold (i.e. those were already false negatives at the model
    layer — the formula isn't the thing that should rescue them).
    """
    if thresholds is None:
        t_crit, t_high, t_med, t_low = 0.80, 0.60, 0.40, 0.30
    else:
        t_crit = thresholds.get("CRITICAL", 0.80)
        t_high = thresholds.get("HIGH", 0.60)
        t_med  = thresholds.get("MEDIUM", 0.40)
        t_low  = thresholds.get("LOW", 0.30)

    conditions = [R >= t_crit, R >= t_high, R >= t_med, R >= t_low]
    choices = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
    tiers = np.select(conditions, choices, default="NORMAL")

    if c_detect is not None:
        gate = MIN_DETECTION_GATE if detection_gate is None else float(detection_gate)
        tiers = np.where(np.asarray(c_detect) < gate, "NORMAL", tiers)
    return tiers


__all__ = ["compute_composite_risk", "assign_risk_levels"]
