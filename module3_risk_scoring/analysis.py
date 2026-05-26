"""Module 3 analysis utilities — fusion, contributions, sensitivity, examples.

Each function is a pure read-only computation on already-computed
component arrays. They produce summary dicts suitable for JSON
serialisation by ``io.save_outputs``.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

from .composition import compute_composite_risk
from .config import DAE_BINARY_THRESHOLD, RESPONSE_MAPPING, WEIGHTS

logger = logging.getLogger(__name__)


# ── Dual-track fusion analysis ───────────────────────────────────────


def dual_track_fusion_analysis(
    c_sup: np.ndarray,
    c_anom: np.ndarray,
    y_true: np.ndarray,
    attack_cats: np.ndarray,
    xgb_threshold: float,
) -> dict:
    """Analyse 4 quadrants of dual-track detection.

    Args:
        c_sup: Track A (supervised) probability of attack.
        c_anom: Track B (DAE) anomaly score, scaled to [0, 1].
        y_true: Ground-truth labels (0=benign, 1=attack).
        attack_cats: Optional per-row attack category strings.
        xgb_threshold: Decision threshold for Track A. Track B uses
            ``DAE_BINARY_THRESHOLD`` from config.
    """
    xgb_flags = c_sup >= xgb_threshold
    dae_flags = c_anom >= DAE_BINARY_THRESHOLD

    both = xgb_flags & dae_flags
    only_xgb = xgb_flags & ~dae_flags
    only_dae = ~xgb_flags & dae_flags
    neither = ~xgb_flags & ~dae_flags

    attack_mask = y_true == 1
    n_attacks = int(attack_mask.sum())

    quadrants: dict[str, dict[str, Any]] = {}
    for name, mask in [
        ("both_flag", both),
        ("only_xgboost", only_xgb),
        ("only_dae", only_dae),
        ("neither", neither),
    ]:
        attack_in_quad = mask & attack_mask
        cats_in_quad: dict[str, int] = {}
        if attack_cats is not None and attack_in_quad.any():
            cats_arr = attack_cats[attack_in_quad].astype(str)
            cats_arr = cats_arr[cats_arr != "None"]
            if len(cats_arr):
                uniq, counts = np.unique(cats_arr, return_counts=True)
                cats_in_quad = {u: int(c) for u, c in zip(uniq, counts)}

        quadrants[name] = {
            "total": int(mask.sum()),
            "true_attacks": int(attack_in_quad.sum()),
            "true_benign": int((mask & ~attack_mask).sum()),
            "attack_categories": cats_in_quad,
        }

    xgb_recall = (
        int((xgb_flags & attack_mask).sum()) / n_attacks if n_attacks > 0 else 0
    )
    dae_recall = (
        int((dae_flags & attack_mask).sum()) / n_attacks if n_attacks > 0 else 0
    )
    union_recall = (
        int(((xgb_flags | dae_flags) & attack_mask).sum()) / n_attacks
        if n_attacks > 0
        else 0
    )
    best_single = max(xgb_recall, dae_recall)
    fusion_gain = union_recall - best_single

    return {
        "quadrants": quadrants,
        "xgb_threshold": xgb_threshold,
        "dae_threshold": DAE_BINARY_THRESHOLD,
        "recall": {
            "xgboost_alone": round(xgb_recall, 4),
            "dae_alone": round(dae_recall, 4),
            "union_fusion": round(union_recall, 4),
            "best_single_track": round(best_single, 4),
            "fusion_gain": round(fusion_gain, 4),
        },
        "total_attacks": n_attacks,
    }


# ── Component contribution analysis ─────────────────────────────────


def component_contribution_analysis(
    c_detect: np.ndarray,
    d_crit: np.ndarray,
    s_data: np.ndarray,
    d_clinical_tier: np.ndarray,
    levels: np.ndarray,
) -> dict:
    """Analyse which component dominates per risk level."""
    comp_names = ["C_detect", "D_crit", "S_data", "D_clinical_tier"]
    w = [WEIGHTS["w1"], WEIGHTS["w2"], WEIGHTS["w3"], WEIGHTS["w4"]]
    weighted = np.column_stack(
        [w[0] * c_detect, w[1] * d_crit, w[2] * s_data, w[3] * d_clinical_tier]
    )

    dominant_idx = np.argmax(weighted, axis=1)
    dominant_names = np.array(comp_names)[dominant_idx]

    per_level: dict[str, dict[str, Any]] = {}
    for level in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]:
        mask = levels == level
        if mask.sum() == 0:
            per_level[level] = {"count": 0}
            continue
        mean_contrib = weighted[mask].mean(axis=0)
        dominant_counts = {
            cn: int((dominant_names[mask] == cn).sum()) for cn in comp_names
        }
        per_level[level] = {
            "count": int(mask.sum()),
            "mean_contributions": {
                cn: round(float(v), 6) for cn, v in zip(comp_names, mean_contrib)
            },
            "dominant_component_counts": dominant_counts,
        }

    overall_dominant = {
        cn: int((dominant_names == cn).sum()) for cn in comp_names
    }

    return {
        "per_level": per_level,
        "overall_dominant": overall_dominant,
    }


# ── Weight sensitivity analysis ─────────────────────────────────────


def weight_sensitivity_analysis(
    c_detect: np.ndarray,
    d_crit: np.ndarray,
    s_data: np.ndarray,
    d_clinical_tier: np.ndarray,
    y_true: np.ndarray,
    *,
    baseline_weights: dict | None = None,
    output_dir=None,
) -> dict:
    """Grid search over weight space; evaluate AUROC of R as binary classifier.

    Args:
        baseline_weights: anchor for the per-component sweep. Defaults
            to module-global ``WEIGHTS`` — pass an override to evaluate
            sensitivity around a non-canonical operating point.
        output_dir: Optional directory for the produced PNG plot. If
            ``None``, plotting is skipped (caller hooks the chart write
            via ``plotting.plot_weight_sensitivity_curve`` instead).
    """
    logger.info("Running weight sensitivity grid search (vectorized)...")

    baseline = baseline_weights or WEIGHTS

    grid_points = np.array([0.10, 0.20, 0.30, 0.40, 0.50])
    n_grid = len(grid_points)

    g1, g2, g3 = np.meshgrid(grid_points, grid_points, grid_points, indexing="ij")
    g4 = np.round(1.0 - g1 - g2 - g3, 2)
    valid = (g4 >= 0.05) & (g4 <= 0.60)

    w1_v = g1[valid]
    w2_v = g2[valid]
    w3_v = g3[valid]
    w4_v = g4[valid]
    n_valid = w1_v.shape[0]

    C = c_detect[np.newaxis, :]
    D = d_crit[np.newaxis, :]
    S = s_data[np.newaxis, :]
    A = d_clinical_tier[np.newaxis, :]

    R_all = (
        w1_v[:, np.newaxis] * C
        + w2_v[:, np.newaxis] * D
        + w3_v[:, np.newaxis] * S
        + w4_v[:, np.newaxis] * A
    )
    R_all = np.clip(R_all, 0.0, 1.0)

    best_auroc = 0.0
    best_weights = dict(baseline)
    all_results = []

    for i in range(n_valid):
        auroc = roc_auc_score(y_true, R_all[i])
        w = {
            "w1": round(float(w1_v[i]), 2),
            "w2": round(float(w2_v[i]), 2),
            "w3": round(float(w3_v[i]), 2),
            "w4": round(float(w4_v[i]), 2),
        }
        all_results.append({"weights": w, "auroc": round(auroc, 4)})
        if auroc > best_auroc:
            best_auroc = auroc
            best_weights = dict(w)

    logger.info("  Grid: %d valid weight combos (of %d total)", n_valid, n_grid ** 3)
    all_results.sort(key=lambda x: -x["auroc"])

    # Per-component sensitivity: fix others, sweep one
    per_component: dict[str, list] = {}
    comp_labels = ["C_detect", "D_crit", "S_data", "D_clinical_tier"]
    weight_keys = ["w1", "w2", "w3", "w4"]
    sweep = np.arange(0.05, 0.65, 0.05)

    for wk, label in zip(weight_keys, comp_labels):
        curve = []
        for val in sweep:
            w_sweep = dict(baseline)
            w_sweep[wk] = round(float(val), 2)
            total = sum(w_sweep.values())
            w_sweep = {k: round(v / total, 4) for k, v in w_sweep.items()}
            R_var = compute_composite_risk(
                c_detect, d_crit, s_data, d_clinical_tier, w_sweep,
            )
            auroc = roc_auc_score(y_true, R_var)
            curve.append({"weight": round(float(val), 2), "auroc": round(auroc, 4)})
        per_component[label] = curve

    logger.info("  Best AUROC: %.4f with weights %s", best_auroc, best_weights)

    return {
        "grid_size": len(all_results),
        "best_weights": best_weights,
        "best_auroc": round(best_auroc, 4),
        "default_weights": dict(baseline),
        "top_10": all_results[:10],
        "per_component_sensitivity": per_component,
    }


# ── Worked examples for the thesis ──────────────────────────────────


def generate_worked_examples(
    R: np.ndarray,
    c_detect: np.ndarray,
    d_crit: np.ndarray,
    s_data: np.ndarray,
    d_clinical_tier: np.ndarray,
    c_track_a: np.ndarray,
    c_track_b: np.ndarray,
    levels: np.ndarray,
    y_true: np.ndarray,
    attack_cats: np.ndarray,
) -> list:
    """Generate fully worked numerical examples for thesis."""
    examples = []
    attack_mask = y_true == 1

    if attack_mask.any():
        idx = int(np.where(attack_mask)[0][np.argmax(R[attack_mask])])
        examples.append(_build_example(
            "Highest-risk true attack", idx,
            R, c_detect, d_crit, s_data, d_clinical_tier,
            c_track_a, c_track_b, levels, y_true, attack_cats,
        ))
        idx = int(np.where(attack_mask)[0][np.argmin(R[attack_mask])])
        examples.append(_build_example(
            "Lowest-risk true attack (potential under-triage)", idx,
            R, c_detect, d_crit, s_data, d_clinical_tier,
            c_track_a, c_track_b, levels, y_true, attack_cats,
        ))

    benign_mask = y_true == 0
    if benign_mask.any():
        idx = int(np.where(benign_mask)[0][np.argmax(R[benign_mask])])
        examples.append(_build_example(
            "Highest-risk benign sample (false alarm analysis)", idx,
            R, c_detect, d_crit, s_data, d_clinical_tier,
            c_track_a, c_track_b, levels, y_true, attack_cats,
        ))

    return examples


def _build_example(
    title: str, idx: int,
    R, c_detect, d_crit, s_data, d_clinical_tier,
    c_track_a, c_track_b, levels, y_true, attack_cats,
) -> dict:
    """Build a single worked example with full numerical trace."""
    w = WEIGHTS
    return {
        "title": title,
        "sample_index": idx,
        "ground_truth": "attack" if y_true[idx] == 1 else "benign",
        "attack_category": str(attack_cats[idx]) if attack_cats is not None else "unknown",
        "components": {
            "Track_A (XGBoost proba)": round(float(c_track_a[idx]), 6),
            "Track_B (DAE proba)": round(float(c_track_b[idx]), 6),
            "C_detect (fused)": round(float(c_detect[idx]), 6),
            "D_crit": round(float(d_crit[idx]), 6),
            "S_data": round(float(s_data[idx]), 6),
            "D_clinical_tier": round(float(d_clinical_tier[idx]), 6),
        },
        "weighted_contributions": {
            f"w1({w['w1']})×C_detect": round(float(w["w1"] * c_detect[idx]), 6),
            f"w2({w['w2']})×D_crit": round(float(w["w2"] * d_crit[idx]), 6),
            f"w3({w['w3']})×S_data": round(float(w["w3"] * s_data[idx]), 6),
            f"w4({w['w4']})×D_clinical_tier": round(float(w["w4"] * d_clinical_tier[idx]), 6),
        },
        "R": round(float(R[idx]), 6),
        "risk_level": str(levels[idx]),
        "response": RESPONSE_MAPPING.get(str(levels[idx]), {}),
    }


__all__ = [
    "dual_track_fusion_analysis",
    "component_contribution_analysis",
    "weight_sensitivity_analysis",
    "generate_worked_examples",
    "_build_example",
]
