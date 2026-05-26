"""Feedback-loop adjustments for thresholds + weights.

These functions consume the *suggested_threshold_change* dict produced
by ``module5_responses.module5_pipeline.FeedbackLoop`` and apply
clamped per-iteration adjustments. The clamps prevent oscillation when
the feedback signal is noisy.
"""

from __future__ import annotations

import logging
from typing import Dict

import numpy as np
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)


def apply_feedback(
    current_thresholds: dict,
    feedback: dict,
    max_delta: float = 0.10,
) -> dict:
    """Apply feedback-loop adjustments to tier thresholds.

    Takes the *suggested_threshold_change* dict from
    ``FeedbackLoop.compute_adjustments()`` and clamps each per-tier
    adjustment to ±max_delta.

    Args:
        current_thresholds: e.g. ``{"CRITICAL": 0.80, "HIGH": 0.60, "MEDIUM": 0.40}``
        feedback: must contain ``"suggested_threshold_change"`` key with
            the same tier keys as *current_thresholds*. Unknown tier
            keys are logged as a WARNING (potential operator typo) and
            then dropped.
        max_delta: maximum absolute change per tier per iteration.

    Returns:
        Updated thresholds dict with the same keys as *current_thresholds*.
    """
    suggested = feedback.get("suggested_threshold_change", {})

    # Warn on unknown tier keys (N3 fix).
    unknown_tiers = set(suggested) - set(current_thresholds)
    if unknown_tiers:
        logger.warning(
            "apply_feedback: suggestion contains unknown tier(s) %s; "
            "ignoring. Known tiers: %s",
            sorted(unknown_tiers),
            sorted(current_thresholds),
        )

    updated: Dict[str, float] = {}
    for tier, cur_val in current_thresholds.items():
        new_val = suggested.get(tier, cur_val)
        delta = new_val - cur_val
        clamped = max(-max_delta, min(max_delta, delta))
        updated[tier] = round(cur_val + clamped, 4)
    return updated


def apply_weight_feedback(
    current_weights: dict,
    component_variances: dict,
    y_true: np.ndarray,
    c_detect: np.ndarray,
    d_crit: np.ndarray,
    s_data: np.ndarray,
    d_clinical_tier: np.ndarray,
    max_delta: float = 0.05,
) -> dict:
    """Adjust composite weights using AUROC as the optimisation target.

    Two-stage update:
      1. **Variance-based redistribution** — components contributing
         <5% of total variance shrink by 20% of their weight (capped at
         ``max_delta``); the saved mass is spread evenly across the
         remaining components.
      2. **Local AUROC hill-climb** — each weight is swept across 11
         points in [-max_delta, +max_delta] around its current value
         (clamped at 0.05 lower bound). Trial composites are computed
         in one (11, N) broadcast per weight, then scored against
         ``y_true`` via ``roc_auc_score``. The argmax wins.

    Output weights are renormalised to sum=1.0.
    """
    components = {
        "w1": c_detect, "w2": d_crit, "w3": s_data, "w4": d_clinical_tier,
    }
    w = dict(current_weights)

    # ── Variance-based redistribution ──
    total_var = sum(component_variances.values()) or 1.0
    low_var_keys = [
        k for k, v in component_variances.items()
        if v / total_var < 0.05  # contributes < 5% of total variance
    ]
    if low_var_keys:
        redistribute = 0.0
        for k in low_var_keys:
            reduction = min(w[k] * 0.2, max_delta)
            w[k] -= reduction
            redistribute += reduction
        others = [k for k in w if k not in low_var_keys]
        per_other = redistribute / len(others) if others else 0
        for k in others:
            w[k] += per_other

    # ── Local AUROC hill-climb (vectorised broadcast) ──
    # Each weight swept across 11 steps; trial R vectors are stacked
    # into a (11, N) matrix and computed in one broadcast.
    steps = np.linspace(-max_delta, max_delta, 11)
    comp_arrays = np.array([c_detect, d_crit, s_data, d_clinical_tier])  # (4, N)
    wkeys = ["w1", "w2", "w3", "w4"]

    for wi, wk in enumerate(wkeys):
        trial_vals = np.clip(w[wk] + steps, 0.05, None)  # (11,)
        w_base = np.array([w[k] for k in wkeys])         # (4,)
        w_matrix = np.tile(w_base, (11, 1))              # (11, 4)
        w_matrix[:, wi] = trial_vals
        row_sums = w_matrix.sum(axis=1, keepdims=True)
        w_matrix /= row_sums                              # normalised rows
        R_trials = np.clip(w_matrix @ comp_arrays, 0.0, 1.0)  # (11, N)
        aurocs = np.array([roc_auc_score(y_true, R_trials[i]) for i in range(11)])
        best_i = int(np.argmax(aurocs))
        w[wk] = float(w_matrix[best_i, wi])

    # Final renormalise
    s = sum(w.values())
    return {k: round(float(v / s), 4) for k, v in w.items()}


__all__ = ["apply_feedback", "apply_weight_feedback"]
