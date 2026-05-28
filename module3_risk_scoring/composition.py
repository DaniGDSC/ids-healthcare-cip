"""Composite risk formula + tier assignment.

These two functions are the public-facing surface of Module 3 most
heavily consumed by downstream code (``src/risk_scorer.py``,
``tools/diagnostics/*``).
"""

from __future__ import annotations

import numpy as np

from .config import (
    CONTEXT_WEIGHTS_V2,
    MIN_DETECTION_GATE,
    RISK_THRESHOLDS,
    RISK_THRESHOLDS_V2,
    WEIGHTS,
)


# Supported formula versions. ``v1`` is the original linear weighted
# sum (paper-frozen). ``v2`` is the Sprint-4 two-layer architecture
# (production-deployed). Keeping both alongside lets RQ1 reproduction
# stay byte-exact while new builds use the architecturally honest
# formula.
SUPPORTED_FORMULA_VERSIONS = ("v1", "v2")
# Default = v1 so existing call sites (tests, downstream modules,
# diagnostics) keep their current behaviour. v2 is opted into via the
# ``module3_risk_scores --formula-version v2`` CLI flag for new regen
# runs; the resulting npz carries a ``formula_version`` field so the
# dashboard can render the right interpretation.
DEFAULT_FORMULA_VERSION = "v1"


def compute_composite_risk(
    c_detect: np.ndarray,
    d_crit: np.ndarray,
    s_data: np.ndarray,
    d_clinical_tier: np.ndarray,
    weights: dict | None = None,
    *,
    formula_version: str = DEFAULT_FORMULA_VERSION,
) -> np.ndarray:
    """Composite risk score — dispatches between v1 (paper) and v2 (deployed).

    Args:
        c_detect: Cascaded Track A → Track B fused detection score in [0, 1].
        d_crit:   Device criticality × CIA threat interaction in [0, 1].
        s_data:   Data sensitivity in [0, 1].
        d_clinical_tier: Patient-acuity proxy in [0, 1].
        weights:  v1 override only — must carry keys w1..w4. Ignored by v2.
        formula_version: ``"v1"`` (legacy linear sum, paper-frozen) or
            ``"v2"`` (Sprint-4 two-layer gate + amplification — default).

    Returns:
        R in [0, 1] (output is clipped).

    Raises:
        ValueError on unsupported ``formula_version``.
    """
    if formula_version == "v1":
        w = weights or WEIGHTS
        R = (
            w["w1"] * c_detect
            + w["w2"] * d_crit
            + w["w3"] * s_data
            + w["w4"] * d_clinical_tier
        )
        return np.clip(R, 0.0, 1.0)
    if formula_version == "v2":
        return _compute_composite_risk_v2(c_detect, d_crit, s_data, d_clinical_tier)
    raise ValueError(
        f"Unknown formula_version {formula_version!r}; supported: {SUPPORTED_FORMULA_VERSIONS}"
    )


def _compute_composite_risk_v2(
    c_detect: np.ndarray,
    d_crit: np.ndarray,
    s_data: np.ndarray,
    d_clinical_tier: np.ndarray,
    weights: dict | None = None,
) -> np.ndarray:
    """Sprint-4 two-layer formula::

        Layer 1 (gate):       passthrough if C_detect ≥ MIN_DETECTION_GATE,
                              else 0
        Layer 2 (modulate):   R = C_detect × (1 + α·D_crit
                                              + β·S_data
                                              + γ·D_clinical_tier)

    Context never creates an alert when detection is silent —
    fixing the v1 "vital-monitoring-idle floor at R ≈ 0.21" bug.
    The full derivation lives in ``docs/formula_v2_rationale.md``.
    """
    w = weights or CONTEXT_WEIGHTS_V2
    c_detect = np.asarray(c_detect, dtype=float)
    context = (
        w["alpha"] * np.asarray(d_crit, dtype=float)
        + w["beta"]  * np.asarray(s_data, dtype=float)
        + w["gamma"] * np.asarray(d_clinical_tier, dtype=float)
    )
    R = c_detect * (1.0 + context)
    gated = np.where(c_detect < MIN_DETECTION_GATE, 0.0, R)
    return np.clip(gated, 0.0, 1.0)


def _default_thresholds_for(formula_version: str) -> tuple[float, float, float, float]:
    table = (
        RISK_THRESHOLDS_V2 if formula_version == "v2" else RISK_THRESHOLDS
    )
    by_name = {name: t for t, name in table}
    return (
        by_name["CRITICAL"], by_name["HIGH"], by_name["MEDIUM"], by_name["LOW"],
    )


def assign_risk_levels(
    R: np.ndarray,
    thresholds: dict | None = None,
    *,
    c_detect: np.ndarray | None = None,
    detection_gate: float | None = None,
    formula_version: str = DEFAULT_FORMULA_VERSION,
) -> np.ndarray:
    """Map composite scores to 5 alert tiers.

    The default threshold table depends on ``formula_version`` so v2's
    different R distribution doesn't get mismapped through v1 cutoffs.
    Both versions share the 5-tier vocabulary (CRITICAL/HIGH/MEDIUM/
    LOW/NORMAL); only the numeric cutoffs differ.

    Args:
        R: Composite risk scores in [0, 1].
        thresholds: Optional explicit override (any keys missing fall
            back to the version's defaults).
        c_detect: Optional detection-score array. When supplied, samples
            below ``detection_gate`` are forced to NORMAL. v2 already
            embeds the gate in ``compute_composite_risk`` (returns 0
            below the gate, which falls below any positive LOW
            threshold), so passing c_detect here is redundant but
            harmless.
        detection_gate: Override the gate value used at the tier-
            assignment step.
        formula_version: ``"v1"`` or ``"v2"`` — selects which threshold
            table to use as the default.
    """
    defaults = _default_thresholds_for(formula_version)
    if thresholds is None:
        t_crit, t_high, t_med, t_low = defaults
    else:
        t_crit = thresholds.get("CRITICAL", defaults[0])
        t_high = thresholds.get("HIGH",     defaults[1])
        t_med  = thresholds.get("MEDIUM",   defaults[2])
        t_low  = thresholds.get("LOW",      defaults[3])

    conditions = [R >= t_crit, R >= t_high, R >= t_med, R >= t_low]
    choices = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
    tiers = np.select(conditions, choices, default="NORMAL")

    if c_detect is not None:
        gate = MIN_DETECTION_GATE if detection_gate is None else float(detection_gate)
        tiers = np.where(np.asarray(c_detect) < gate, "NORMAL", tiers)
    return tiers


__all__ = [
    "compute_composite_risk",
    "assign_risk_levels",
    "SUPPORTED_FORMULA_VERSIONS",
    "DEFAULT_FORMULA_VERSION",
]
