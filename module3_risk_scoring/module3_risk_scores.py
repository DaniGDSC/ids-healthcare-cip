#!/usr/bin/env python3
"""Module 3 — Composite Risk Scores (RQ2/RO2).

Combines dual-track detection into a fused confidence score, then merges
with device criticality, data sensitivity, and clinical tier:

    R = 0.40·C_detect + 0.25·D_crit + 0.15·S_data + 0.20·D_clinical_tier

where C_detect uses cascaded Track A → Track B fusion: the DAE receives
[raw_features || Track_A_probabilities] as input, making spoofing attacks
visible through the joint feature-prediction space.

Maps scores to alert priority levels and demonstrates dual-track fusion
value — cases where combining Track A + Track B catches threats that
a single track misses.

Usage:
    python compute_risk_scores.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "results/reports"
CHARTS_DIR = PROJECT_ROOT / "results/charts"

# ── Configuration ──────────────────────────────────────────────────────

# Composite formula: R = w1·C_detect + w2·D_crit + w3·S_data + w4·D_clinical_tier
WEIGHTS = {"w1": 0.40, "w2": 0.25, "w3": 0.15, "w4": 0.20}

# Risk level thresholds — 3 boundaries, 4 tiers
RISK_THRESHOLDS = [(0.80, "CRITICAL"), (0.60, "HIGH"), (0.40, "MEDIUM")]

from common.phi import BIOMETRIC_COLUMNS

# Stable list ordering for downstream callers; backed by the canonical
# PHI set in common/phi.py.
BIOMETRIC_FEATURES = sorted(BIOMETRIC_COLUMNS)
SIGMA_THRESHOLD = 1.5

# ── Pre-computed feature indices (Opt-6) ───────────────────────────────
# WUSTL-EHMS-2020 feature order is stable across pipeline runs.
# Pre-computing indices once at module load avoids O(n_feat × n_bio)
# list.index() scans inside compute_s_data() and compute_d_clinical_tier().
# These are populated lazily on first call to _get_bio_idx().

_BIO_IDX_CACHE: np.ndarray | None = None
_BIO_IDX_FEAT_NAMES: list | None = None   # the feat_names used to build cache


def _get_bio_idx(feat_names: list) -> np.ndarray:
    """Return cached biometric feature indices for the given feature list.

    Recomputes only when feat_names changes (never in a normal pipeline run).
    """
    global _BIO_IDX_CACHE, _BIO_IDX_FEAT_NAMES
    if _BIO_IDX_CACHE is None or feat_names is not _BIO_IDX_FEAT_NAMES:
        _BIO_IDX_CACHE = np.array(
            [feat_names.index(f) for f in BIOMETRIC_FEATURES if f in feat_names],
            dtype=np.intp,
        )
        _BIO_IDX_FEAT_NAMES = feat_names
    return _BIO_IDX_CACHE

# CIA threat profile per attack category
CIA_THREATS = {
    "Spoofing":        {"C": 0.6, "I": 0.9, "A": 0.3},
    "Data Alteration": {"C": 0.3, "I": 1.0, "A": 0.2},
}

# Pre-computed per-category D_crit scores (M3-1).
# max(C, I, A) × base_tier evaluated once at module load instead of
# per-row inside compute_d_crit().  Populated after DEVICE_TIERS is defined.
_CIA_SCORE: dict[str, float] = {}   # filled below, after DEVICE_TIERS

# Device criticality tiers (WUSTL-EHMS-2020 is a generic IoMT testbed)
DEVICE_TIERS = {
    "life_sustaining": 1.0,   # infusion pumps, ventilators
    "vital_monitoring": 0.8,  # ECG, pulse oximeter
    "diagnostic":      0.5,   # blood pressure, temperature
    "auxiliary":        0.3,   # environmental sensors
}
DEFAULT_DEVICE_TIER = "vital_monitoring"  # WUSTL-EHMS-2020 default

# Finalise pre-computed lookup now that DEVICE_TIERS is defined (M3-1).
_BASE_TIER = DEVICE_TIERS[DEFAULT_DEVICE_TIER]
_CIA_SCORE = {
    cat: _BASE_TIER * max(t.values())
    for cat, t in CIA_THREATS.items()
}
_DEFAULT_CIA_SCORE = _BASE_TIER * 0.5

# Data sensitivity classification
DATA_SENSITIVITY = {
    "phi_realtime":    1.0,  # real-time vital signs (SpO2, HR, BP)
    "phi_stored":      0.7,  # stored patient records
    "device_telemetry": 0.4, # network flow metadata
    "non_sensitive":   0.1,  # timestamps, flags
}

# Response mapping per risk level
RESPONSE_MAPPING = {
    "CRITICAL": {
        "action": "Immediate network isolation + page physician + escalate to CISO",
        "max_response_min": 5,
        "auto_actions": ["isolate_device", "page_oncall", "snapshot_forensics"],
    },
    "HIGH": {
        "action": "Active investigation + isolate segment + notify biomedical engineering",
        "max_response_min": 15,
        "auto_actions": ["isolate_segment", "notify_biomed", "create_ticket"],
    },
    "MEDIUM": {
        "action": "Flag for review + enhanced monitoring + notify security team",
        "max_response_min": 60,
        "auto_actions": ["enhanced_logging", "notify_soc"],
    },
    "LOW": {
        "action": "Log for audit + review at next shift",
        "max_response_min": 480,
        "auto_actions": ["log_event"],
    },
    "NORMAL": {
        "action": "No action — routine logging",
        "max_response_min": 0,
        "auto_actions": [],
    },
}


# ── Data loading ────────────────────────────────────────────────────────

def _split_paths(split: str) -> dict:
    """Resolve per-split paths. Test = paper-clean; demo = operator-clean.

    Thin wrapper over :mod:`common.split_paths` so the call sites below
    keep their original dict-access shape; the canonical path mapping
    now lives in common.
    """
    from common import split_paths as sp
    return {
        "parquet": sp.parquet(split),
        "out_npz": sp.risk_scores(split),
    }


def load_test_data(parquet_path: Path | None = None) -> tuple:
    """Load a split's parquet → X, y, attack_cats, feat_names."""
    path = parquet_path or (PROJECT_ROOT / "data/processed/test_phase1.parquet")
    df = pd.read_parquet(path)
    drop_cols = ["Label", "Attack Category", "row_id", "device_class"]
    feat_names = [c for c in df.columns if c not in drop_cols]
    X_test = df[feat_names].values.astype(np.float32)
    y_test = df["Label"].values
    attack_cats = df["Attack Category"].values if "Attack Category" in df.columns else None
    return X_test, y_test, attack_cats, feat_names


def load_xgboost_proba() -> tuple:
    """Load XGBoost predict_proba and optimal threshold."""
    preds = np.load(PROJECT_ROOT / "results/models/xgboost_test_predictions.npz")
    with open(PROJECT_ROOT / "results/models/xgboost_final_report.json") as f:
        threshold = json.load(f)["optimal_threshold"]
    return preds["y_proba"], threshold


# ── Component computation ──────────────────────────────────────────────

# C_detect (cascaded Track A → Track B fusion) is produced by
# detection_engine.DetectionEngine — Module 3 just consumes c_detect,
# c_track_a, and c_track_b from the engine result. The augmented-input
# construction lives in one place there; this module focuses on risk
# composition (D_crit + S_data + D_clinical_tier + composite R).


def compute_d_crit(attack_cats: np.ndarray) -> np.ndarray:
    """Device criticality from tier + CIA threat interaction.

    M3-1: replaces O(N) Python loop with a single vectorised Pandas map.
    _CIA_SCORE is pre-computed at module load (one max() per category).
    """
    scores = (
        pd.Series(attack_cats, dtype=str)
        .map(_CIA_SCORE)
        .fillna(_DEFAULT_CIA_SCORE)
        .values.astype(np.float64)
    )
    return np.clip(scores, 0.0, 1.0)


def compute_s_data(X_test: np.ndarray, feat_names: list) -> np.ndarray:
    """Data sensitivity: weighted mix of PHI (biometric) vs telemetry features.

    Biometric features carry PHI real-time sensitivity (1.0).
    Network features carry device telemetry sensitivity (0.4).
    Per-sample score = fraction of high-sensitivity features that are active
    (non-zero or anomalous), weighted by their sensitivity tier.

    Optimisation (Opt-6): uses _get_bio_idx() for cached feature index lookup.
    """
    bio_idx = _get_bio_idx(feat_names)
    n_feats = len(feat_names)
    n_bio = len(bio_idx)
    n_net = n_feats - n_bio

    # Sensitivity weight per feature
    phi_weight = DATA_SENSITIVITY["phi_realtime"]
    net_weight = DATA_SENSITIVITY["device_telemetry"]

    # Any biometric feature present (non-zero) indicates PHI in the flow
    bio_active = (np.abs(X_test[:, bio_idx]) > 0.01).sum(axis=1) / n_bio
    # Network features are always present in flow data
    net_present = np.ones(len(X_test))

    s_data = (phi_weight * bio_active + net_weight * net_present) / (phi_weight + net_weight)
    return np.clip(s_data, 0.0, 1.0)


def compute_d_clinical_tier(X_test: np.ndarray, feat_names: list) -> np.ndarray:
    """Patient acuity: fraction of biometric features exceeding 1.5 sigma.

    Optimisation (Opt-6): uses _get_bio_idx() for cached feature index lookup.
    """
    bio_idx = _get_bio_idx(feat_names)
    bio_vals = X_test[:, bio_idx]
    abnormal_count = (np.abs(bio_vals) > SIGMA_THRESHOLD).sum(axis=1)
    return abnormal_count / len(BIOMETRIC_FEATURES)


def compute_composite_risk(
    c_detect: np.ndarray,
    d_crit: np.ndarray,
    s_data: np.ndarray,
    d_clinical_tier: np.ndarray,
    weights: dict | None = None,
) -> np.ndarray:
    """R = w1·C_detect + w2·D_crit + w3·S_data + w4·D_clinical_tier."""
    w = weights or WEIGHTS
    R = (w["w1"] * c_detect +
         w["w2"] * d_crit +
         w["w3"] * s_data +
         w["w4"] * d_clinical_tier)
    return np.clip(R, 0.0, 1.0)


def assign_risk_levels(
    R: np.ndarray,
    thresholds: dict | None = None,
) -> np.ndarray:
    """Map composite scores to 4 alert tiers using 3 thresholds.

    Parameters
    ----------
    thresholds : dict, optional
        {"CRITICAL": 0.80, "HIGH": 0.60, "MEDIUM": 0.40}.
        Falls back to module-level RISK_THRESHOLDS when *None*.
    """
    if thresholds is None:
        t_crit, t_high, t_med = 0.80, 0.60, 0.40
    else:
        t_crit = thresholds.get("CRITICAL", 0.80)
        t_high = thresholds.get("HIGH", 0.60)
        t_med  = thresholds.get("MEDIUM", 0.40)

    conditions = [R >= t_crit, R >= t_high, R >= t_med]
    choices = ["CRITICAL", "HIGH", "MEDIUM"]
    return np.select(conditions, choices, default="LOW")


def apply_feedback(
    current_thresholds: dict,
    feedback: dict,
    max_delta: float = 0.10,
) -> dict:
    """Apply feedback-loop adjustments to tier thresholds.

    Takes the *suggested_threshold_change* dict produced by
    ``FeedbackLoop.compute_adjustments()`` and clamps each per-tier
    adjustment to ±max_delta to prevent oscillation.

    Parameters
    ----------
    current_thresholds : dict
        e.g. {"CRITICAL": 0.80, "HIGH": 0.60, "MEDIUM": 0.40}
    feedback : dict
        Must contain ``"suggested_threshold_change"`` key with the same
        tier keys as *current_thresholds*.
    max_delta : float
        Maximum absolute change allowed per tier per iteration (default 0.10).

    Returns
    -------
    dict  — updated thresholds with the same keys.
    """
    suggested = feedback.get("suggested_threshold_change", {})
    updated = {}
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
    """Adjust Module 3 weights using AUROC as the optimization target.

    If a component has low variance contribution (e.g. D_crit), reduce
    its weight and redistribute proportionally to the others.  Then do a
    local 1-D line search per weight to maximise AUROC, clamped to
    ±max_delta per iteration.

    Returns
    -------
    dict  — updated weights (sum = 1.0).
    """
    from sklearn.metrics import roc_auc_score

    components = {
        "w1": c_detect, "w2": d_crit, "w3": s_data, "w4": d_clinical_tier,
    }
    w = dict(current_weights)

    # --- Variance-based redistribution ---
    total_var = sum(component_variances.values()) or 1.0
    low_var_keys = [
        k for k, v in component_variances.items()
        if v / total_var < 0.05  # contributes < 5 % of total variance
    ]
    if low_var_keys:
        redistribute = 0.0
        for k in low_var_keys:
            reduction = min(w[k] * 0.2, max_delta)  # shrink by 20 % of its weight
            w[k] -= reduction
            redistribute += reduction
        # spread evenly among the others
        others = [k for k in w if k not in low_var_keys]
        per_other = redistribute / len(others) if others else 0
        for k in others:
            w[k] += per_other

    # --- Local AUROC hill-climb per weight (M3-5: vectorised broadcast) ---
    # Each weight is swept over 11 steps; trial R vectors are stacked into a
    # (11, N) matrix and computed in one broadcast instead of 11 serial calls.
    steps = np.linspace(-max_delta, max_delta, 11)
    comp_arrays = np.array([c_detect, d_crit, s_data, d_clinical_tier])  # (4, N)
    wkeys = ["w1", "w2", "w3", "w4"]

    for wi, wk in enumerate(wkeys):
        trial_vals = np.clip(w[wk] + steps, 0.05, None)  # (11,)

        # Build (11, 4) weight matrices, one row per step
        w_base = np.array([w[k] for k in wkeys])  # (4,)
        w_matrix = np.tile(w_base, (11, 1))         # (11, 4)
        w_matrix[:, wi] = trial_vals

        # Normalise each row
        row_sums = w_matrix.sum(axis=1, keepdims=True)
        w_matrix /= row_sums  # (11, 4)

        # R_trials: (11, N) — single broadcast
        R_trials = np.clip(w_matrix @ comp_arrays, 0.0, 1.0)  # (11, N)

        aurocs = np.array([roc_auc_score(y_true, R_trials[i]) for i in range(11)])
        best_i = int(np.argmax(aurocs))
        w[wk] = float(w_matrix[best_i, wi])

    # Final normalize
    s = sum(w.values())
    w = {k: round(float(v / s), 4) for k, v in w.items()}
    return w


# ── Dual-track fusion analysis ─────────────────────────────────────────

def dual_track_fusion_analysis(
    c_sup: np.ndarray,
    c_anom: np.ndarray,
    y_true: np.ndarray,
    attack_cats: np.ndarray,
    xgb_threshold: float,
) -> dict:
    """Analyze 4 quadrants of dual-track detection."""
    # DAE binary threshold: use the DAE's own threshold on predict_proba scale
    # Since predict_proba clips to [0,1] with min-max from benign, any value
    # significantly above 0 indicates anomaly. Use 0.5 as midpoint.
    dae_threshold = 0.5

    xgb_flags = c_sup >= xgb_threshold
    dae_flags = c_anom >= dae_threshold

    both = xgb_flags & dae_flags
    only_xgb = xgb_flags & ~dae_flags
    only_dae = ~xgb_flags & dae_flags
    neither = ~xgb_flags & ~dae_flags

    attack_mask = y_true == 1
    n_attacks = int(attack_mask.sum())

    quadrants = {}
    for name, mask in [("both_flag", both), ("only_xgboost", only_xgb),
                       ("only_dae", only_dae), ("neither", neither)]:
        attack_in_quad = mask & attack_mask
        cats_in_quad = {}
        if attack_cats is not None and attack_in_quad.any():
            # M3-2: np.unique with return_counts — O(N log N) C-level sort,
            # replaces O(K×N) Python list-comprehension per category.
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

    # Recall metrics
    xgb_recall = int((xgb_flags & attack_mask).sum()) / n_attacks if n_attacks > 0 else 0
    dae_recall = int((dae_flags & attack_mask).sum()) / n_attacks if n_attacks > 0 else 0
    union_recall = int(((xgb_flags | dae_flags) & attack_mask).sum()) / n_attacks if n_attacks > 0 else 0
    best_single = max(xgb_recall, dae_recall)
    fusion_gain = union_recall - best_single

    return {
        "quadrants": quadrants,
        "xgb_threshold": xgb_threshold,
        "dae_threshold": dae_threshold,
        "recall": {
            "xgboost_alone": round(xgb_recall, 4),
            "dae_alone": round(dae_recall, 4),
            "union_fusion": round(union_recall, 4),
            "best_single_track": round(best_single, 4),
            "fusion_gain": round(fusion_gain, 4),
        },
        "total_attacks": n_attacks,
    }


# ── Component contribution analysis ───────────────────────────────────

def component_contribution_analysis(
    c_detect: np.ndarray,
    d_crit: np.ndarray,
    s_data: np.ndarray,
    d_clinical_tier: np.ndarray,
    levels: np.ndarray,
) -> dict:
    """Analyze which component dominates per risk level."""
    comp_names = ["C_detect", "D_crit", "S_data", "D_clinical_tier"]
    w = [WEIGHTS["w1"], WEIGHTS["w2"], WEIGHTS["w3"], WEIGHTS["w4"]]
    weighted = np.column_stack([
        w[0] * c_detect, w[1] * d_crit, w[2] * s_data, w[3] * d_clinical_tier,
    ])

    # Dominant component per sample
    dominant_idx = np.argmax(weighted, axis=1)
    dominant_names = np.array(comp_names)[dominant_idx]

    # Per risk level breakdown
    per_level = {}
    for level in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]:
        mask = levels == level
        if mask.sum() == 0:
            per_level[level] = {"count": 0}
            continue
        mean_contrib = weighted[mask].mean(axis=0)
        dominant_counts = {}
        for cn in comp_names:
            dominant_counts[cn] = int((dominant_names[mask] == cn).sum())

        per_level[level] = {
            "count": int(mask.sum()),
            "mean_contributions": {cn: round(float(v), 6) for cn, v in zip(comp_names, mean_contrib)},
            "dominant_component_counts": dominant_counts,
        }

    # Overall dominant
    overall_dominant = {cn: int((dominant_names == cn).sum()) for cn in comp_names}

    return {
        "per_level": per_level,
        "overall_dominant": overall_dominant,
    }


# ── Visualizations ─────────────────────────────────────────────────────

def plot_risk_distribution(R: np.ndarray, levels: np.ndarray) -> None:
    """Histogram of risk scores with level boundaries."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Color-coded background regions
    colors = {"LOW": "#f1c40f", "MEDIUM": "#e67e22",
              "HIGH": "#e74c3c", "CRITICAL": "#8e44ad"}
    boundaries = [(0, 0.4, "LOW"), (0.4, 0.6, "MEDIUM"),
                  (0.6, 0.8, "HIGH"), (0.8, 1.0, "CRITICAL")]
    for lo, hi, label in boundaries:
        ax.axvspan(lo, hi, alpha=0.15, color=colors[label])

    ax.hist(R, bins=100, edgecolor="black", linewidth=0.5, alpha=0.8, color="#3274A1")

    for thresh, label in RISK_THRESHOLDS:
        count = (levels == label).sum()
        ax.axvline(thresh, color=colors[label], linestyle="--", linewidth=1.5)
        ax.text(thresh + 0.01, ax.get_ylim()[1] * 0.9, f"{label}\n(n={count})",
                fontsize=8, color=colors[label], fontweight="bold")

    low_count = (levels == "LOW").sum()
    ax.text(0.02, ax.get_ylim()[1] * 0.9, f"LOW\n(n={low_count})",
            fontsize=8, color=colors["LOW"], fontweight="bold")

    ax.set_xlabel("Composite Risk Score R")
    ax.set_ylabel("Count")
    ax.set_title("Risk Score Distribution with Alert Priority Levels")
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "risk_distribution.png", dpi=150)
    plt.close(fig)
    logger.info("  Chart: risk_distribution.png")


def plot_component_breakdown(contributions: dict) -> None:
    """Stacked bar of mean weighted contributions per risk level."""
    comp_names = ["C_detect", "D_crit", "S_data", "D_clinical_tier"]
    colors = ["#C44E52", "#3274A1", "#55A868", "#CCB974"]
    level_order = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

    active_levels = [l for l in level_order if contributions["per_level"][l].get("count", 0) > 0]
    if not active_levels:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(active_levels))
    bottom = np.zeros(len(active_levels))

    for cn, color in zip(comp_names, colors):
        vals = [contributions["per_level"][l]["mean_contributions"].get(cn, 0)
                for l in active_levels]
        ax.bar(x, vals, bottom=bottom, color=color, label=cn, width=0.6)
        bottom += np.array(vals)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{l}\n(n={contributions['per_level'][l]['count']})"
                        for l in active_levels])
    ax.set_ylabel("Mean Weighted Contribution")
    ax.set_title("Component Breakdown by Risk Level")
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "component_breakdown.png", dpi=150)
    plt.close(fig)
    logger.info("  Chart: component_breakdown.png")


def plot_dual_track_heatmap(fusion: dict) -> None:
    """2x2 heatmap showing dual-track detection quadrants."""
    q = fusion["quadrants"]
    # Rows: DAE (flag/no), Cols: XGB (flag/no)
    # [DAE+ XGB+, DAE+ XGB-]
    # [DAE- XGB+, DAE- XGB-]
    matrix_total = np.array([
        [q["both_flag"]["true_attacks"], q["only_dae"]["true_attacks"]],
        [q["only_xgboost"]["true_attacks"], q["neither"]["true_attacks"]],
    ])
    matrix_all = np.array([
        [q["both_flag"]["total"], q["only_dae"]["total"]],
        [q["only_xgboost"]["total"], q["neither"]["total"]],
    ])

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix_total, cmap="YlOrRd", aspect="auto")

    labels = [
        [f"Both flag\n{q['both_flag']['true_attacks']} attacks\n({q['both_flag']['total']} total)",
         f"Only DAE\n{q['only_dae']['true_attacks']} attacks\n({q['only_dae']['total']} total)"],
        [f"Only XGBoost\n{q['only_xgboost']['true_attacks']} attacks\n({q['only_xgboost']['total']} total)",
         f"Neither\n{q['neither']['true_attacks']} attacks\n({q['neither']['total']} total)"],
    ]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, labels[i][j], ha="center", va="center", fontsize=10, fontweight="bold")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["XGBoost Flags", "XGBoost Clear"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["DAE Flags", "DAE Clear"])
    ax.set_title("Dual-Track Detection Quadrants (True Attacks)")
    plt.colorbar(im, label="True Attacks")
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "dual_track_venn.png", dpi=150)
    plt.close(fig)
    logger.info("  Chart: dual_track_venn.png")


def plot_component_scatter(
    c_sup: np.ndarray,
    c_anom: np.ndarray,
    y_true: np.ndarray,
) -> None:
    """Scatter of C_supervised vs C_anomaly colored by ground truth."""
    fig, ax = plt.subplots(figsize=(10, 8))

    benign = y_true == 0
    attack = y_true == 1

    ax.scatter(c_sup[benign], c_anom[benign], c="#2ecc71", alpha=0.3, s=10, label="Benign")
    ax.scatter(c_sup[attack], c_anom[attack], c="#e74c3c", alpha=0.6, s=20, label="Attack")

    ax.set_xlabel("C_supervised (XGBoost probability)")
    ax.set_ylabel("C_anomaly (DAE normalized score)")
    ax.set_title("Track A vs Track B — Complementary Detection Zones")
    ax.legend()
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "component_scatter.png", dpi=150)
    plt.close(fig)
    logger.info("  Chart: component_scatter.png")


def plot_risk_by_category(
    R: np.ndarray,
    attack_cats: np.ndarray,
    y_true: np.ndarray,
) -> None:
    """Box plot of risk scores by attack category."""
    categories = []
    scores = []
    normal_mask = y_true == 0
    categories.extend(["Normal"] * int(normal_mask.sum()))
    scores.extend(R[normal_mask].tolist())

    if attack_cats is not None:
        # M3-3: cast once outside the loop — boolean mask via numpy, O(N) per cat.
        cats_str = attack_cats.astype(str)
        attack_mask = y_true == 1
        for cat in sorted(np.unique(cats_str[attack_mask])):
            mask = (cats_str == cat) & attack_mask
            categories.extend([cat] * int(mask.sum()))
            scores.extend(R[mask].tolist())

    df = pd.DataFrame({"Category": categories, "Risk Score": scores})

    fig, ax = plt.subplots(figsize=(10, 6))
    cats = df["Category"].unique()
    colors = {"Normal": "#2ecc71", "Spoofing": "#e74c3c", "Data Alteration": "#8e44ad"}
    bp_data = [df[df["Category"] == c]["Risk Score"].values for c in cats]
    bp = ax.boxplot(bp_data, labels=cats, patch_artist=True, widths=0.5)
    for patch, cat in zip(bp["boxes"], cats):
        patch.set_facecolor(colors.get(cat, "#3274A1"))
        patch.set_alpha(0.7)

    ax.set_ylabel("Composite Risk Score R")
    ax.set_title("Risk Score Distribution by Attack Category")
    ax.axhline(0.4, color="orange", linestyle="--", alpha=0.5, label="MEDIUM threshold")
    ax.axhline(0.6, color="red", linestyle="--", alpha=0.5, label="HIGH threshold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "risk_by_category.png", dpi=150)
    plt.close(fig)
    logger.info("  Chart: risk_by_category.png")


def plot_risk_by_label(R: np.ndarray, y_true: np.ndarray) -> None:
    """Overlaid histograms of R for benign vs attack — verify separation."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(R[y_true == 0], bins=80, alpha=0.6, color="#2ecc71", label="Benign", density=True)
    ax.hist(R[y_true == 1], bins=80, alpha=0.6, color="#e74c3c", label="Attack", density=True)
    ax.axvline(0.40, color="orange", linestyle="--", label="MEDIUM threshold")
    ax.axvline(0.60, color="red", linestyle="--", label="HIGH threshold")
    ax.set_xlabel("Composite Risk Score R")
    ax.set_ylabel("Density")
    ax.set_title("Risk Score Distribution by True Label — Separation Quality")
    ax.legend()
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "risk_by_label.png", dpi=150)
    plt.close(fig)
    logger.info("  Chart: risk_by_label.png")


# ── Standalone config exports (Tasks 3.1, 3.2, 3.8) ───────────────────

def export_config_jsons() -> None:
    """Export standalone JSON config files for device criticality, data sensitivity, risk config."""
    # 3.1 Device criticality
    device_crit = {
        "description": "Device criticality tiers mapped to D_crit scores",
        "tiers": {
            "1_life_sustaining": {"score": 1.0, "examples": ["infusion pump", "ventilator"]},
            "2_vital_monitoring": {"score": 0.8, "examples": ["ECG monitor", "pulse oximeter"]},
            "3_diagnostic": {"score": 0.5, "examples": ["blood pressure monitor", "thermometer"]},
            "4_auxiliary": {"score": 0.3, "examples": ["environmental sensor", "room monitor"]},
        },
        "default_tier": DEFAULT_DEVICE_TIER,
        "cia_threat_profiles": CIA_THREATS,
    }
    (OUTPUT_DIR / "device_criticality.json").write_text(
        json.dumps(device_crit, indent=2), encoding="utf-8")
    logger.info("  Saved: device_criticality.json")

    # 3.2 Data sensitivity
    data_sens = {
        "description": "Data sensitivity classification mapped to S_data scores",
        "tiers": {
            "phi_realtime": {"score": 1.0, "examples": ["real-time vital signs (SpO2, HR, BP)"]},
            "phi_stored": {"score": 0.7, "examples": ["stored patient records"]},
            "operational": {"score": 0.4, "examples": ["network flow metadata, device telemetry"]},
            "administrative": {"score": 0.1, "examples": ["timestamps, flags, non-clinical"]},
        },
    }
    (OUTPUT_DIR / "data_sensitivity.json").write_text(
        json.dumps(data_sens, indent=2), encoding="utf-8")
    logger.info("  Saved: data_sensitivity.json")

    # 3.8 Risk scoring config
    risk_cfg = {
        "formula": "R = w1*C_detect + w2*D_crit + w3*S_data + w4*D_clinical_tier",
        "fusion": "C_detect = cascaded(Track_A → Track_B): DAE input = [raw_features || Track_A_probas]; DAE forward pass skipped where Track_A (XGBoost) proba >= 0.90 (compute optimisation; c_track_b=0 for those rows)",
        "weights": WEIGHTS,
        "thresholds": {label: thresh for thresh, label in RISK_THRESHOLDS},
        "alert_tiers": ["CRITICAL", "HIGH", "MEDIUM", "LOW"],
        "biometric_features": list(BIOMETRIC_FEATURES),
        "sigma_threshold": SIGMA_THRESHOLD,
        "device_tiers": DEVICE_TIERS,
        "data_sensitivity": DATA_SENSITIVITY,
        "cia_threats": CIA_THREATS,
        "response_mapping": RESPONSE_MAPPING,
    }
    (OUTPUT_DIR / "risk_config.json").write_text(
        json.dumps(risk_cfg, indent=2), encoding="utf-8")
    logger.info("  Saved: risk_config.json")


# ── Save outputs ───────────────────────────────────────────────────────

# ── Sensitivity analysis ───────────────────────────────────────────────

def weight_sensitivity_analysis(
    c_detect: np.ndarray,
    d_crit: np.ndarray,
    s_data: np.ndarray,
    d_clinical_tier: np.ndarray,
    y_true: np.ndarray,
) -> dict:
    """Grid search over weight space; evaluate AUROC of R as binary classifier.

    Optimisation (Opt-3): replaces triple nested Python loop with a
    vectorized numpy meshgrid approach.

    Old: O(n_grid³) Python loop × O(n_samples) composite_risk call each iter.
    New: single broadcast over shape (n_grid, n_grid, n_grid, n_samples),
         one roc_auc_score call per valid combo — the inner axis (n_samples)
         is computed in C, not Python.  For 5-point grid (125 combos) and
         5000 samples: ~100× wall-time reduction on this function.
    """
    from sklearn.metrics import roc_auc_score
    logger.info("Running weight sensitivity grid search (vectorized)...")

    grid_points = np.array([0.10, 0.20, 0.30, 0.40, 0.50])
    n_grid = len(grid_points)

    # Build all (w1, w2, w3) combinations at once via meshgrid
    g1, g2, g3 = np.meshgrid(grid_points, grid_points, grid_points, indexing="ij")
    g4 = np.round(1.0 - g1 - g2 - g3, 2)

    # Valid mask: w4 in [0.05, 0.60]
    valid = (g4 >= 0.05) & (g4 <= 0.60)

    # Extract valid weight vectors: shape (n_valid, 4)
    w1_v = g1[valid]; w2_v = g2[valid]; w3_v = g3[valid]; w4_v = g4[valid]
    n_valid = w1_v.shape[0]

    # Broadcast R computation: (n_valid, n_samples)
    C = c_detect[np.newaxis, :]   # (1, n_samples)
    D = d_crit[np.newaxis, :]
    S = s_data[np.newaxis, :]
    A = d_clinical_tier[np.newaxis, :]

    R_all = (
        w1_v[:, np.newaxis] * C +
        w2_v[:, np.newaxis] * D +
        w3_v[:, np.newaxis] * S +
        w4_v[:, np.newaxis] * A
    )
    R_all = np.clip(R_all, 0.0, 1.0)  # (n_valid, n_samples)

    # Compute AUROC per weight combo — still a Python loop but O(n_valid) not O(n³)
    best_auroc = 0.0
    best_weights = dict(WEIGHTS)
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

    # Sort by AUROC
    all_results.sort(key=lambda x: -x["auroc"])

    # Per-component sensitivity: fix others, vary one
    per_component = {}
    comp_labels = ["C_detect", "D_crit", "S_data", "D_clinical_tier"]
    weight_keys = ["w1", "w2", "w3", "w4"]
    sweep = np.arange(0.05, 0.65, 0.05)

    for i, (wk, label) in enumerate(zip(weight_keys, comp_labels)):
        curve = []
        for val in sweep:
            w = dict(WEIGHTS)
            w[wk] = round(float(val), 2)
            total = sum(w.values())
            w = {k: round(v / total, 4) for k, v in w.items()}
            R_var = compute_composite_risk(c_detect, d_crit, s_data, d_clinical_tier, w)
            auroc = roc_auc_score(y_true, R_var)
            curve.append({"weight": round(float(val), 2), "auroc": round(auroc, 4)})
        per_component[label] = curve

    logger.info("  Grid: %d weight combos evaluated", len(all_results))
    logger.info("  Best AUROC: %.4f with weights %s", best_auroc, best_weights)
    logger.info("  Default AUROC: %.4f", all_results[0]["auroc"]
                if all_results else 0)

    # Sensitivity plot
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#C44E52", "#3274A1", "#55A868", "#CCB974"]
    for (label, curve), color in zip(per_component.items(), colors):
        ws = [c["weight"] for c in curve]
        aucs = [c["auroc"] for c in curve]
        ax.plot(ws, aucs, "o-", color=color, label=label, linewidth=2, markersize=5)
    ax.axhline(best_auroc, color="black", linestyle=":", alpha=0.5, label=f"Best={best_auroc:.4f}")
    ax.set_xlabel("Component Weight")
    ax.set_ylabel("AUROC (R as binary classifier)")
    ax.set_title("Weight Sensitivity Analysis — AUROC vs Component Weight")
    ax.legend()
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "weight_sensitivity.png", dpi=150)
    plt.close(fig)
    logger.info("  Chart: weight_sensitivity.png")

    return {
        "grid_size": len(all_results),
        "best_weights": best_weights,
        "best_auroc": round(best_auroc, 4),
        "default_weights": dict(WEIGHTS),
        "top_10": all_results[:10],
        "per_component_sensitivity": per_component,
    }


# ── Worked examples ────────────────────────────────────────────────────

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

    # Example 1: highest-R true attack (should be CRITICAL)
    attack_mask = y_true == 1
    if attack_mask.any():
        idx = int(np.where(attack_mask)[0][np.argmax(R[attack_mask])])
        examples.append(_build_example(
            "Highest-risk true attack", idx,
            R, c_detect, d_crit, s_data, d_clinical_tier,
            c_track_a, c_track_b, levels, y_true, attack_cats,
        ))

    # Example 2: lowest-R true attack (borderline / missed by risk scoring)
    if attack_mask.any():
        idx = int(np.where(attack_mask)[0][np.argmin(R[attack_mask])])
        examples.append(_build_example(
            "Lowest-risk true attack (potential under-triage)", idx,
            R, c_detect, d_crit, s_data, d_clinical_tier,
            c_track_a, c_track_b, levels, y_true, attack_cats,
        ))

    # Example 3: highest-R benign sample (false alarm candidate)
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


# ── Save outputs ───────────────────────────────────────────────────────

def save_outputs(
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
    fusion: dict,
    contributions: dict,
    sensitivity: dict,
    worked_examples: list,
    *,
    out_npz: Path | None = None,
) -> None:
    """Save all risk score artifacts.

    `out_npz` defaults to `risk_scores.npz` (test); demo runs pass
    `demo_scores.npz`. The auxiliary CSV/JSON outputs stay at canonical
    paths to preserve test as the source-of-truth for paper artifacts.
    """
    npz_path = out_npz or (OUTPUT_DIR / "risk_scores.npz")
    np.savez(
        npz_path,
        R=R, c_detect=c_detect, d_crit=d_crit,
        s_data=s_data, d_clinical_tier=d_clinical_tier,
        c_track_a=c_track_a, c_track_b=c_track_b,
        risk_levels=levels, y_true=y_true,
    )
    logger.info("  Saved: %s", npz_path.name)

    # CSV detail
    df = pd.DataFrame({
        "R": R, "risk_level": levels, "y_true": y_true,
        "attack_category": attack_cats,
        "c_detect": c_detect, "c_track_a": c_track_a, "c_track_b": c_track_b,
        "d_crit": d_crit, "s_data": s_data, "d_clinical_tier": d_clinical_tier,
    })
    df.to_csv(OUTPUT_DIR / "risk_scores_detail.csv", index_label="sample_index")
    logger.info("  Saved: risk_scores_detail.csv")

    # JSON report
    level_dist = {}
    for level in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]:
        mask = levels == level
        level_dist[level] = {
            "count": int(mask.sum()),
            "pct": round(float(mask.mean() * 100), 1),
            "mean_R": round(float(R[mask].mean()), 4) if mask.any() else 0,
        }

    report = {
        "formula": "R = w1*C_detect + w2*D_crit + w3*S_data + w4*D_clinical_tier",
        "fusion": "C_detect = cascaded(Track_A → Track_B): DAE input = [raw_features || Track_A_probas]; DAE forward pass skipped where Track_A (XGBoost) proba >= 0.90 (compute optimisation; c_track_b=0 for those rows)",
        "weights": WEIGHTS,
        "risk_thresholds": {label: thresh for thresh, label in RISK_THRESHOLDS},
        "total_samples": int(len(R)),
        "risk_level_distribution": level_dist,
        "response_mapping": RESPONSE_MAPPING,
        "overall_stats": {
            "mean_R": round(float(R.mean()), 4),
            "std_R": round(float(R.std()), 4),
            "median_R": round(float(np.median(R)), 4),
        },
        "per_category_stats": {},
        "dual_track_fusion": fusion,
        "component_contributions": contributions,
        "weight_sensitivity": sensitivity,
        "worked_examples": worked_examples,
        "limitations": [
            "Patient acuity proxy uses biometric deviation magnitude, not clinical diagnosis — a simplified surrogate for real patient acuity scoring (e.g., APACHE, NEWS2).",
            "Device criticality uses a static tier assignment for the WUSTL-EHMS-2020 testbed; production deployment requires integration with hospital asset management systems.",
            "Data sensitivity classification is feature-type-based, not content-aware — cannot distinguish encrypted vs plaintext PHI.",
            "Linear weighted sum assumes component independence; multiplicative or Bayesian formulations may better capture risk interactions.",
            "Weights are expert-calibrated defaults; institutional tuning via AHP or operational feedback loops is recommended for deployment.",
            "The WUSTL-EHMS-2020 dataset contains only 2 attack categories (Spoofing, Data Alteration); generalizability to broader IoMT threat landscapes requires validation on additional datasets.",
        ],
    }

    # Per-category R stats
    # M3-4: cast attack_cats to str once — avoids O(K×N) Python loop per category.
    if attack_cats is not None:
        cats_str = attack_cats.astype(str)
        for cat in ["normal", "Spoofing", "Data Alteration"]:
            mask = y_true == 0 if cat == "normal" else (cats_str == cat) & (y_true == 1)
            if mask.any():
                report["per_category_stats"][cat] = {
                    "count": int(mask.sum()),
                    "mean_R": round(float(R[mask].mean()), 4),
                    "median_R": round(float(np.median(R[mask])), 4),
                    "std_R": round(float(R[mask].std()), 4),
                }

    report_path = OUTPUT_DIR / "risk_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("  Saved: risk_report.json")


# ── Main ────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        prog="python -m module3_risk_scoring.module3_risk_scores",
        description="Module 3 — composite risk scoring. Operates on the "
                    "selected frozen split (test=paper-clean, demo=operator-clean).",
    )
    parser.add_argument(
        "--split",
        choices=["test", "demo", "both"],
        default="test",
        help="Frozen split to process. 'test' writes the paper-clean "
             "`risk_scores.npz`; 'demo' writes `demo_scores.npz`.",
    )
    args = parser.parse_args()

    splits_to_run = ["test", "demo"] if args.split == "both" else [args.split]

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    sep = "=" * 72
    t0 = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    for split in splits_to_run:
        _run_one_split(split, sep)
    logger.info("Module 3 complete (%.1fs, splits=%s)",
                time.perf_counter() - t0, splits_to_run)


def _run_one_split(split: str, sep: str) -> None:
    paths = _split_paths(split)
    logger.info(sep)
    logger.info("MODULE 3 — COMPOSITE RISK SCORES (RQ2/RO2) — split=%s", split)
    logger.info(sep)

    # ── Load data ──
    X_test, y_test, attack_cats, feat_names = load_test_data(paths["parquet"])
    n_samples = len(y_test)
    n_attacks = (y_test == 1).sum()
    logger.info("Data: %d samples (%d attacks) from %s",
                n_samples, n_attacks, paths["parquet"].name)

    # ── Compute components ──
    logger.info("Computing risk components...")

    # XGBoost threshold log — only available for test (cached test predictions).
    # Demo runs skip this informational read since the threshold doesn't enter
    # the formula (it's used only for the operating-point log line).
    if split == "test":
        try:
            _, xgb_threshold = load_xgboost_proba()
            logger.info("  Track A: XGBoost proba, threshold=%.3f", xgb_threshold)
        except FileNotFoundError:
            logger.warning("  XGBoost cached test predictions absent; skipping threshold log")

    from detection_engine import DetectionEngine
    det_result = DetectionEngine().predict(X_test)
    c_track_a = det_result.c_track_a
    c_track_b = det_result.c_track_b
    c_detect = det_result.c_detect
    logger.info("  C_detect (cascaded fusion): range [%.4f, %.4f]",
                c_detect.min(), c_detect.max())

    d_crit = compute_d_crit(attack_cats)
    logger.info("  D_crit: device tier=%s, %.0f elevated (attacks)",
                DEFAULT_DEVICE_TIER, (d_crit > DEVICE_TIERS[DEFAULT_DEVICE_TIER] * 0.5).sum())

    s_data = compute_s_data(X_test, feat_names)
    logger.info("  S_data: range [%.4f, %.4f]", s_data.min(), s_data.max())

    d_clinical_tier = compute_d_clinical_tier(X_test, feat_names)
    logger.info("  D_clinical_tier: %.1f%% samples have abnormal biometrics",
                (d_clinical_tier > 0).mean() * 100)

    # ── Composite risk ──
    R = compute_composite_risk(c_detect, d_crit, s_data, d_clinical_tier)
    levels = assign_risk_levels(R)
    logger.info("")
    logger.info("Composite risk R: mean=%.4f, median=%.4f, std=%.4f",
                R.mean(), np.median(R), R.std())

    for level in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]:
        count = (levels == level).sum()
        pct = count / n_samples * 100
        logger.info("  %-10s %5d (%5.1f%%)", level, count, pct)

    # ── Dual-track fusion ──
    logger.info("")
    logger.info("── Dual-Track Fusion Analysis ──")
    # On demo splits where the test-cached XGBoost threshold isn't loaded, use
    # the canonical 0.5 fallback (only affects an informational log line).
    _xgb_threshold = locals().get("xgb_threshold", 0.5)
    fusion = dual_track_fusion_analysis(c_track_a, c_track_b, y_test, attack_cats, _xgb_threshold)
    r = fusion["recall"]
    logger.info("  XGBoost recall: %.4f", r["xgboost_alone"])
    logger.info("  DAE recall:     %.4f", r["dae_alone"])
    logger.info("  Union recall:   %.4f (fusion gain: +%.4f)", r["union_fusion"], r["fusion_gain"])
    for qname, qdata in fusion["quadrants"].items():
        logger.info("  %-15s %4d total, %3d attacks %s",
                    qname, qdata["total"], qdata["true_attacks"],
                    qdata.get("attack_categories", ""))

    # ── Component contribution ──
    contributions = component_contribution_analysis(c_detect, d_crit, s_data, d_clinical_tier, levels)

    # ── Sensitivity analysis ──
    logger.info("")
    sensitivity = weight_sensitivity_analysis(c_detect, d_crit, s_data, d_clinical_tier, y_test)

    # ── Worked examples ──
    logger.info("")
    logger.info("Generating worked examples...")
    worked_examples = generate_worked_examples(
        R, c_detect, d_crit, s_data, d_clinical_tier,
        c_track_a, c_track_b, levels, y_test, attack_cats,
    )
    for ex in worked_examples:
        logger.info("  %s (sample %d): R=%.4f → %s",
                    ex["title"], ex["sample_index"], ex["R"], ex["risk_level"])

    # ── Save ──
    logger.info("")
    logger.info("Saving outputs...")
    save_outputs(R, c_detect, d_crit, s_data, d_clinical_tier, c_track_a, c_track_b,
                 levels, y_test, attack_cats, fusion, contributions,
                 sensitivity, worked_examples,
                 out_npz=paths["out_npz"])

    # ── Visualizations + config JSON exports (test split only — paper figures
    #    must not be clobbered by demo runs) ──
    if split == "test":
        logger.info("Generating charts...")
        plot_risk_distribution(R, levels)
        plot_component_breakdown(contributions)
        plot_dual_track_heatmap(fusion)
        plot_component_scatter(c_track_a, c_track_b, y_test)
        plot_risk_by_category(R, attack_cats, y_test)
        plot_risk_by_label(R, y_test)
        logger.info("Exporting config JSONs...")
        export_config_jsons()

    logger.info("")
    logger.info(sep)
    logger.info("SPLIT %s COMPLETE", split.upper())
    logger.info(sep)
    logger.info("  Formula   : R = %.2f·C_detect + %.2f·D_crit + %.2f·S_data + %.2f·D_clinical_tier",
                WEIGHTS["w1"], WEIGHTS["w2"], WEIGHTS["w3"], WEIGHTS["w4"])
    logger.info("  Fusion    : C_detect = cascaded(Track_A → Track_B)")
    logger.info("  Output    : %s", paths["out_npz"])
    logger.info(sep)


if __name__ == "__main__":
    main()
