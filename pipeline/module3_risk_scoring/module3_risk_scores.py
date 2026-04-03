#!/usr/bin/env python3
"""Module 3 — Composite Risk Scores (RQ2/RO2).

Combines dual-track detection into a fused confidence score, then merges
with device criticality, data sensitivity, and patient acuity:

    R = w1·C_detect + w2·D_crit + w3·S_data + w4·A_patient

where C_detect = max(Track_A_proba, Track_B_normalized_RE) fuses
supervised and novelty-based anomaly scores.

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
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data/phase2/risk_scores"
CHARTS_DIR = OUTPUT_DIR / "charts"

# ── Configuration ──────────────────────────────────────────────────────

# Composite formula: R = w1·C_detect + w2·D_crit + w3·S_data + w4·A_patient
WEIGHTS = {"w1": 0.40, "w2": 0.25, "w3": 0.15, "w4": 0.20}

# Risk level thresholds — 3 boundaries, 4 tiers
RISK_THRESHOLDS = [(0.80, "CRITICAL"), (0.60, "HIGH"), (0.40, "MEDIUM")]

BIOMETRIC_FEATURES = [
    "Temp", "SpO2", "Pulse_Rate", "SYS", "DIA",
    "Heart_rate", "Resp_Rate", "ST",
]
SIGMA_THRESHOLD = 1.5

# CIA threat profile per attack category
CIA_THREATS = {
    "Spoofing":        {"C": 0.6, "I": 0.9, "A": 0.3},
    "Data Alteration": {"C": 0.3, "I": 1.0, "A": 0.2},
}

# Device criticality tiers (WUSTL-EHMS-2020 is a generic IoMT testbed)
DEVICE_TIERS = {
    "life_sustaining": 1.0,   # infusion pumps, ventilators
    "vital_monitoring": 0.8,  # ECG, pulse oximeter
    "diagnostic":      0.5,   # blood pressure, temperature
    "auxiliary":        0.3,   # environmental sensors
}
DEFAULT_DEVICE_TIER = "vital_monitoring"  # WUSTL-EHMS-2020 default

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

def load_test_data() -> tuple:
    """Load test parquet → X_test, y_test, attack_cats, feat_names."""
    df = pd.read_parquet(PROJECT_ROOT / "data/processed/test_phase1.parquet")
    drop_cols = ["Label", "Attack Category"]
    feat_names = [c for c in df.columns if c not in drop_cols]
    X_test = df[feat_names].values.astype(np.float32)
    y_test = df["Label"].values
    attack_cats = df["Attack Category"].values if "Attack Category" in df.columns else None
    return X_test, y_test, attack_cats, feat_names


def load_xgboost_proba() -> tuple:
    """Load XGBoost predict_proba and optimal threshold."""
    preds = np.load(PROJECT_ROOT / "data/phase2/xgboost/final/test_predictions.npz")
    with open(PROJECT_ROOT / "data/phase2/xgboost/final/final_report.json") as f:
        threshold = json.load(f)["optimal_threshold"]
    return preds["y_proba"], threshold


# ── Component computation ──────────────────────────────────────────────

def compute_c_detect(
    c_track_a: np.ndarray,
    X_test: np.ndarray,
) -> tuple:
    """Fused detection confidence: C_detect = max(Track_A, Track_B)."""
    det = joblib.load(PROJECT_ROOT / "data/phase2/dae/final/dae_detector.pkl")
    c_track_b = det.predict_proba(X_test)
    c_detect = np.maximum(c_track_a, c_track_b)
    return np.clip(c_detect, 0.0, 1.0), c_track_b


def compute_d_crit(attack_cats: np.ndarray) -> np.ndarray:
    """Device criticality from tier + CIA threat interaction."""
    base_tier = DEVICE_TIERS[DEFAULT_DEVICE_TIER]
    scores = np.full(len(attack_cats), base_tier * 0.5, dtype=np.float64)
    for i, cat in enumerate(attack_cats):
        cat_str = str(cat) if cat is not None else ""
        if cat_str in CIA_THREATS:
            threat = CIA_THREATS[cat_str]
            cia_max = max(threat["C"], threat["I"], threat["A"])
            scores[i] = base_tier * cia_max
    return np.clip(scores, 0.0, 1.0)


def compute_s_data(X_test: np.ndarray, feat_names: list) -> np.ndarray:
    """Data sensitivity: weighted mix of PHI (biometric) vs telemetry features.

    Biometric features carry PHI real-time sensitivity (1.0).
    Network features carry device telemetry sensitivity (0.4).
    Per-sample score = fraction of high-sensitivity features that are active
    (non-zero or anomalous), weighted by their sensitivity tier.
    """
    bio_idx = [feat_names.index(f) for f in BIOMETRIC_FEATURES]
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


def compute_a_patient(X_test: np.ndarray, feat_names: list) -> np.ndarray:
    """Patient acuity: fraction of biometric features exceeding 1.5 sigma."""
    bio_idx = [feat_names.index(f) for f in BIOMETRIC_FEATURES]
    bio_vals = X_test[:, bio_idx]
    abnormal_count = (np.abs(bio_vals) > SIGMA_THRESHOLD).sum(axis=1)
    return abnormal_count / len(BIOMETRIC_FEATURES)


def compute_composite_risk(
    c_detect: np.ndarray,
    d_crit: np.ndarray,
    s_data: np.ndarray,
    a_patient: np.ndarray,
    weights: dict | None = None,
) -> np.ndarray:
    """R = w1·C_detect + w2·D_crit + w3·S_data + w4·A_patient."""
    w = weights or WEIGHTS
    R = (w["w1"] * c_detect +
         w["w2"] * d_crit +
         w["w3"] * s_data +
         w["w4"] * a_patient)
    return np.clip(R, 0.0, 1.0)


def assign_risk_levels(R: np.ndarray) -> np.ndarray:
    """Map composite scores to 4 alert tiers using 3 thresholds."""
    conditions = [R >= 0.80, R >= 0.60, R >= 0.40]
    choices = ["CRITICAL", "HIGH", "MEDIUM"]
    return np.select(conditions, choices, default="LOW")


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
        if attack_cats is not None:
            for cat in sorted(set(str(c) for c in attack_cats[attack_in_quad] if c)):
                cats_in_quad[cat] = int(np.sum(
                    [str(c) == cat for c in attack_cats[attack_in_quad]]
                ))

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
    a_patient: np.ndarray,
    levels: np.ndarray,
) -> dict:
    """Analyze which component dominates per risk level."""
    comp_names = ["C_detect", "D_crit", "S_data", "A_patient"]
    w = [WEIGHTS["w1"], WEIGHTS["w2"], WEIGHTS["w3"], WEIGHTS["w4"]]
    weighted = np.column_stack([
        w[0] * c_detect, w[1] * d_crit, w[2] * s_data, w[3] * a_patient,
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
    comp_names = ["C_detect", "D_crit", "S_data", "A_patient"]
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
    for cat_label, mask in [("Normal", y_true == 0)]:
        categories.extend(["Normal"] * mask.sum())
        scores.extend(R[mask].tolist())

    if attack_cats is not None:
        for cat in sorted(set(str(c) for c in attack_cats[y_true == 1])):
            mask = np.array([str(c) == cat for c in attack_cats]) & (y_true == 1)
            categories.extend([cat] * mask.sum())
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
        "formula": "R = w1*C_detect + w2*D_crit + w3*S_data + w4*A_patient",
        "fusion": "C_detect = max(Track_A_proba, Track_B_normalized_RE)",
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
    a_patient: np.ndarray,
    y_true: np.ndarray,
) -> dict:
    """Grid search over weight space; evaluate AUROC of R as binary classifier."""
    from sklearn.metrics import roc_auc_score
    logger.info("Running weight sensitivity grid search...")

    grid_points = [0.10, 0.20, 0.30, 0.40, 0.50]
    best_auroc = 0.0
    best_weights = dict(WEIGHTS)
    all_results = []

    for w1 in grid_points:
        for w2 in grid_points:
            for w3 in grid_points:
                w4 = round(1.0 - w1 - w2 - w3, 2)
                if w4 < 0.05 or w4 > 0.60:
                    continue
                w = {"w1": w1, "w2": w2, "w3": w3, "w4": w4}
                R_var = compute_composite_risk(c_detect, d_crit, s_data, a_patient, w)
                auroc = roc_auc_score(y_true, R_var)
                all_results.append({"weights": w, "auroc": round(auroc, 4)})
                if auroc > best_auroc:
                    best_auroc = auroc
                    best_weights = dict(w)

    # Sort by AUROC
    all_results.sort(key=lambda x: -x["auroc"])

    # Per-component sensitivity: fix others, vary one
    per_component = {}
    comp_labels = ["C_detect", "D_crit", "S_data", "A_patient"]
    weight_keys = ["w1", "w2", "w3", "w4"]
    sweep = np.arange(0.05, 0.65, 0.05)

    for i, (wk, label) in enumerate(zip(weight_keys, comp_labels)):
        curve = []
        for val in sweep:
            w = dict(WEIGHTS)
            w[wk] = round(float(val), 2)
            total = sum(w.values())
            w = {k: round(v / total, 4) for k, v in w.items()}
            R_var = compute_composite_risk(c_detect, d_crit, s_data, a_patient, w)
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
    a_patient: np.ndarray,
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
            R, c_detect, d_crit, s_data, a_patient,
            c_track_a, c_track_b, levels, y_true, attack_cats,
        ))

    # Example 2: lowest-R true attack (borderline / missed by risk scoring)
    if attack_mask.any():
        idx = int(np.where(attack_mask)[0][np.argmin(R[attack_mask])])
        examples.append(_build_example(
            "Lowest-risk true attack (potential under-triage)", idx,
            R, c_detect, d_crit, s_data, a_patient,
            c_track_a, c_track_b, levels, y_true, attack_cats,
        ))

    # Example 3: highest-R benign sample (false alarm candidate)
    benign_mask = y_true == 0
    if benign_mask.any():
        idx = int(np.where(benign_mask)[0][np.argmax(R[benign_mask])])
        examples.append(_build_example(
            "Highest-risk benign sample (false alarm analysis)", idx,
            R, c_detect, d_crit, s_data, a_patient,
            c_track_a, c_track_b, levels, y_true, attack_cats,
        ))

    return examples


def _build_example(
    title: str, idx: int,
    R, c_detect, d_crit, s_data, a_patient,
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
            "A_patient": round(float(a_patient[idx]), 6),
        },
        "weighted_contributions": {
            f"w1({w['w1']})×C_detect": round(float(w["w1"] * c_detect[idx]), 6),
            f"w2({w['w2']})×D_crit": round(float(w["w2"] * d_crit[idx]), 6),
            f"w3({w['w3']})×S_data": round(float(w["w3"] * s_data[idx]), 6),
            f"w4({w['w4']})×A_patient": round(float(w["w4"] * a_patient[idx]), 6),
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
    a_patient: np.ndarray,
    c_track_a: np.ndarray,
    c_track_b: np.ndarray,
    levels: np.ndarray,
    y_true: np.ndarray,
    attack_cats: np.ndarray,
    fusion: dict,
    contributions: dict,
    sensitivity: dict,
    worked_examples: list,
) -> None:
    """Save all risk score artifacts."""
    # NPZ
    np.savez(
        OUTPUT_DIR / "risk_scores.npz",
        R=R, c_detect=c_detect, d_crit=d_crit,
        s_data=s_data, a_patient=a_patient,
        c_track_a=c_track_a, c_track_b=c_track_b,
        risk_levels=levels, y_true=y_true,
    )
    logger.info("  Saved: risk_scores.npz")

    # CSV detail
    df = pd.DataFrame({
        "R": R, "risk_level": levels, "y_true": y_true,
        "attack_category": attack_cats,
        "c_detect": c_detect, "c_track_a": c_track_a, "c_track_b": c_track_b,
        "d_crit": d_crit, "s_data": s_data, "a_patient": a_patient,
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
        "formula": "R = w1*C_detect + w2*D_crit + w3*S_data + w4*A_patient",
        "fusion": "C_detect = max(Track_A_proba, Track_B_normalized_RE)",
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
    if attack_cats is not None:
        for cat in ["normal", "Spoofing", "Data Alteration"]:
            if cat == "normal":
                mask = y_true == 0
            else:
                mask = np.array([str(c) == cat for c in attack_cats]) & (y_true == 1)
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
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    sep = "=" * 72
    t0 = time.perf_counter()

    logger.info(sep)
    logger.info("MODULE 3 — COMPOSITE RISK SCORES (RQ2/RO2)")
    logger.info(sep)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load data ──
    X_test, y_test, attack_cats, feat_names = load_test_data()
    n_samples = len(y_test)
    n_attacks = (y_test == 1).sum()
    logger.info("Test data: %d samples (%d attacks)", n_samples, n_attacks)

    # ── Compute components ──
    logger.info("Computing risk components...")

    c_track_a, xgb_threshold = load_xgboost_proba()
    logger.info("  Track A: XGBoost proba, threshold=%.3f", xgb_threshold)

    c_detect, c_track_b = compute_c_detect(c_track_a, X_test)
    logger.info("  C_detect (max fusion): range [%.4f, %.4f]",
                c_detect.min(), c_detect.max())

    d_crit = compute_d_crit(attack_cats)
    logger.info("  D_crit: device tier=%s, %.0f elevated (attacks)",
                DEFAULT_DEVICE_TIER, (d_crit > DEVICE_TIERS[DEFAULT_DEVICE_TIER] * 0.5).sum())

    s_data = compute_s_data(X_test, feat_names)
    logger.info("  S_data: range [%.4f, %.4f]", s_data.min(), s_data.max())

    a_patient = compute_a_patient(X_test, feat_names)
    logger.info("  A_patient: %.1f%% samples have abnormal biometrics",
                (a_patient > 0).mean() * 100)

    # ── Composite risk ──
    R = compute_composite_risk(c_detect, d_crit, s_data, a_patient)
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
    fusion = dual_track_fusion_analysis(c_track_a, c_track_b, y_test, attack_cats, xgb_threshold)
    r = fusion["recall"]
    logger.info("  XGBoost recall: %.4f", r["xgboost_alone"])
    logger.info("  DAE recall:     %.4f", r["dae_alone"])
    logger.info("  Union recall:   %.4f (fusion gain: +%.4f)", r["union_fusion"], r["fusion_gain"])
    for qname, qdata in fusion["quadrants"].items():
        logger.info("  %-15s %4d total, %3d attacks %s",
                    qname, qdata["total"], qdata["true_attacks"],
                    qdata.get("attack_categories", ""))

    # ── Component contribution ──
    contributions = component_contribution_analysis(c_detect, d_crit, s_data, a_patient, levels)

    # ── Sensitivity analysis ──
    logger.info("")
    sensitivity = weight_sensitivity_analysis(c_detect, d_crit, s_data, a_patient, y_test)

    # ── Worked examples ──
    logger.info("")
    logger.info("Generating worked examples...")
    worked_examples = generate_worked_examples(
        R, c_detect, d_crit, s_data, a_patient,
        c_track_a, c_track_b, levels, y_test, attack_cats,
    )
    for ex in worked_examples:
        logger.info("  %s (sample %d): R=%.4f → %s",
                    ex["title"], ex["sample_index"], ex["R"], ex["risk_level"])

    # ── Save ──
    logger.info("")
    logger.info("Saving outputs...")
    save_outputs(R, c_detect, d_crit, s_data, a_patient, c_track_a, c_track_b,
                 levels, y_test, attack_cats, fusion, contributions,
                 sensitivity, worked_examples)

    # ── Visualizations ──
    logger.info("Generating charts...")
    plot_risk_distribution(R, levels)
    plot_component_breakdown(contributions)
    plot_dual_track_heatmap(fusion)
    plot_component_scatter(c_track_a, c_track_b, y_test)
    plot_risk_by_category(R, attack_cats, y_test)
    plot_risk_by_label(R, y_test)

    # ── Export standalone config JSONs (Tasks 3.1, 3.2, 3.8) ──
    logger.info("Exporting config JSONs...")
    export_config_jsons()

    # ── Summary ──
    elapsed = round(time.perf_counter() - t0, 1)
    logger.info("")
    logger.info(sep)
    logger.info("RISK SCORING COMPLETE — %.1fs", elapsed)
    logger.info(sep)
    logger.info("  Formula   : R = %.2f·C_detect + %.2f·D_crit + %.2f·S_data + %.2f·A_patient",
                WEIGHTS["w1"], WEIGHTS["w2"], WEIGHTS["w3"], WEIGHTS["w4"])
    logger.info("  Fusion    : C_detect = max(Track_A, Track_B)")
    logger.info("  Output    : %s", OUTPUT_DIR)
    logger.info(sep)


if __name__ == "__main__":
    main()
