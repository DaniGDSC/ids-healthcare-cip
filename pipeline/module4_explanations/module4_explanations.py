#!/usr/bin/env python3
"""Module 4 — Generate Explanations (RQ1/RO1).

Computes explanations for every Phase 2 final-model test prediction:
  Track A (XGBoost, RF, DT): TreeSHAP global + local feature attributions
  Track B (DAE): per-feature weighted reconstruction error decomposition

Produces stakeholder-tailored outputs:
  - Security analyst: SHAP waterfall/bar plots, top contributing features
  - Clinician: plain-language alert summaries
  - Administrator: aggregated risk dashboard data

Usage:
    python generate_explanations.py
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
import shap

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data/phase2/explanations"
CHARTS_DIR = OUTPUT_DIR / "charts"

BIOMETRIC_FEATURES = frozenset({
    "Temp", "SpO2", "Pulse_Rate", "SYS", "DIA",
    "Heart_rate", "Resp_Rate", "ST",
})

TOP_N_WATERFALL = 5
TOP_K_FEATURES = 10

TRACK_A_MODELS = {
    "xgboost": {
        "pipeline": "data/phase2/xgboost/final/final_pipeline.pkl",
        "predictions": "data/phase2/xgboost/final/test_predictions.npz",
        "report": "data/phase2/xgboost/final/final_report.json",
    },
    "random_forest": {
        "pipeline": "data/phase2/random_forest/final/final_pipeline.pkl",
        "predictions": "data/phase2/random_forest/final/test_predictions.npz",
        "report": "data/phase2/random_forest/final/final_report.json",
    },
    "decision_tree": {
        "pipeline": "data/phase2/decision_tree/final/final_pipeline.pkl",
        "predictions": "data/phase2/decision_tree/final/test_predictions.npz",
        "report": "data/phase2/decision_tree/final/final_report.json",
    },
}

CLINICIAN_TEMPLATES = {
    "CRITICAL": (
        "CRITICAL ALERT (Sample {idx}): The system detected a likely intrusion "
        "affecting this patient's monitoring session. The primary indicator was "
        "abnormal {top_feature} ({feature_type} metric). "
        "{biometric_note}"
        "Recommend immediate review of device connectivity and patient vitals."
    ),
    "HIGH": (
        "HIGH ALERT (Sample {idx}): Suspicious activity detected. "
        "Key factor: {top_feature} ({feature_type}). "
        "{biometric_note}"
        "Consider verifying device integrity."
    ),
    "MEDIUM": (
        "MODERATE ALERT (Sample {idx}): Minor anomaly detected in "
        "{top_feature} ({feature_type}). "
        "{biometric_note}"
        "No immediate clinical action required, but flagged for review."
    ),
    "LOW": (
        "LOW ALERT (Sample {idx}): Borderline detection by one model. "
        "Likely benign; logged for audit purposes."
    ),
}


# ── Data loading ────────────────────────────────────────────────────────

def load_test_data() -> tuple:
    """Load test parquet and return X, y, attack categories, feature names."""
    df = pd.read_parquet(PROJECT_ROOT / "data/processed/test_phase1.parquet")
    drop_cols = ["Label", "Attack Category"]
    feat_names = [c for c in df.columns if c not in drop_cols]
    X_test = df[feat_names].values.astype(np.float32)
    y_test = df["Label"].values
    attack_cats = df["Attack Category"].values if "Attack Category" in df.columns else None
    return X_test, y_test, attack_cats, feat_names


def load_predictions(npz_path: Path) -> dict:
    """Load pre-computed predictions from npz."""
    data = np.load(npz_path)
    return {k: data[k] for k in data.files}


# ── Track A: TreeSHAP ──────────────────────────────────────────────────

def compute_tree_shap(
    model_name: str,
    pipeline_path: Path,
    X_test: np.ndarray,
    feat_names: list,
) -> tuple:
    """Compute TreeSHAP values for a Track A model."""
    logger.info("Computing TreeSHAP for %s...", model_name)
    t0 = time.perf_counter()

    pipeline = joblib.load(pipeline_path)
    clf = pipeline.named_steps["classifier"]

    explainer = shap.TreeExplainer(clf)
    sv = explainer.shap_values(X_test)

    # Handle various return shapes for binary classification:
    # - list of 2 arrays: take index [1] (attack class)
    # - 3D array (n, features, 2): take [:, :, 1]
    # - 2D array (n, features): already correct
    if isinstance(sv, list):
        sv = sv[1]
    elif isinstance(sv, np.ndarray) and sv.ndim == 3:
        sv = sv[:, :, 1]

    expected = explainer.expected_value
    if isinstance(expected, (list, np.ndarray)):
        expected = np.atleast_1d(expected)
        expected = float(expected[1]) if len(expected) > 1 else float(expected[0])
    else:
        expected = float(expected)

    elapsed = time.perf_counter() - t0
    logger.info("  %s TreeSHAP done: shape=%s, expected=%.4f (%.1fs)",
                model_name, sv.shape, expected, elapsed)
    return sv, expected


def save_shap_values(model_name: str, sv: np.ndarray, expected: float, feat_names: list) -> None:
    """Save SHAP values to npz."""
    path = OUTPUT_DIR / f"shap_values_{model_name}.npz"
    np.savez(path, shap_values=sv, expected_value=np.array(expected),
             feature_names=np.array(feat_names))
    logger.info("  Saved: %s", path)


def compute_global_importance(sv: np.ndarray, feat_names: list) -> list:
    """Compute ranked global feature importance from SHAP values."""
    mean_abs = np.mean(np.abs(sv), axis=0)
    ranked = sorted(zip(feat_names, mean_abs), key=lambda x: -x[1])
    return [
        {"rank": i + 1, "feature": name, "mean_abs_shap": round(float(val), 6)}
        for i, (name, val) in enumerate(ranked)
    ]


def save_global_importance(model_name: str, importance: list) -> None:
    """Save global importance to JSON."""
    path = OUTPUT_DIR / f"global_importance_{model_name}.json"
    path.write_text(json.dumps(
        {"model": model_name, "features": importance}, indent=2,
    ), encoding="utf-8")
    logger.info("  Saved: %s", path)


# ── Track B: DAE per-feature error ─────────────────────────────────────

def compute_dae_feature_errors(
    dae_path: Path,
    X_test: np.ndarray,
    feat_names: list,
) -> tuple:
    """Decompose DAE reconstruction error into per-feature contributions."""
    logger.info("Computing DAE per-feature error decomposition...")
    det = joblib.load(dae_path)

    X_norm = det._normalise(X_test)
    recon = det.model.predict(X_norm, verbose=0)
    sq_err = (X_norm - recon) ** 2
    weighted_err = sq_err * det._feat_weights  # per-feature contribution

    logger.info("  DAE decomposition done: shape=%s", weighted_err.shape)
    return sq_err, weighted_err, det._feat_weights


def save_dae_errors(
    sq_err: np.ndarray,
    weighted_err: np.ndarray,
    feat_weights: np.ndarray,
    feat_names: list,
) -> None:
    """Save DAE per-feature errors to npz."""
    path = OUTPUT_DIR / "dae_feature_errors.npz"
    np.savez(path, per_feature_error=sq_err, weighted_per_feature_error=weighted_err,
             feature_weights=feat_weights, feature_names=np.array(feat_names))
    logger.info("  Saved: %s", path)


# ── Visualizations ─────────────────────────────────────────────────────

def _feat_color(name: str) -> str:
    return "#3274A1" if name in BIOMETRIC_FEATURES else "#C44E52"


def plot_global_importance_bar(
    model_name: str,
    importance: list,
    title_suffix: str = "mean |SHAP|",
    value_key: str = "mean_abs_shap",
) -> None:
    """Horizontal bar chart of top-K feature importance."""
    top = importance[:TOP_K_FEATURES]
    names = [f["feature"] for f in top][::-1]
    values = [f[value_key] for f in top][::-1]
    colors = [_feat_color(n) for n in names]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(names, values, color=colors)
    ax.set_xlabel(title_suffix)
    ax.set_title(f"{model_name} — Global Feature Importance ({title_suffix})")
    ax.legend(handles=[
        Patch(facecolor="#C44E52", label="Network"),
        Patch(facecolor="#3274A1", label="Biometric"),
    ], loc="lower right")
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / f"global_importance_{model_name}.png", dpi=150)
    plt.close(fig)
    logger.info("  Chart: global_importance_%s.png", model_name)


def plot_waterfalls(
    model_name: str,
    sv: np.ndarray,
    expected: float,
    X_test: np.ndarray,
    feat_names: list,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> None:
    """Waterfall plots for top-N highest-confidence attack predictions."""
    attack_idx = np.where(y_pred == 1)[0]
    if len(attack_idx) == 0:
        logger.info("  No attacks predicted by %s, skipping waterfalls", model_name)
        return

    # Top-N by confidence
    top_idx = attack_idx[np.argsort(y_proba[attack_idx])[-TOP_N_WATERFALL:]][::-1]

    for rank, idx in enumerate(top_idx):
        explanation = shap.Explanation(
            values=sv[idx],
            base_values=expected,
            data=X_test[idx],
            feature_names=feat_names,
        )
        fig = plt.figure(figsize=(10, 7))
        shap.plots.waterfall(explanation, show=False)
        plt.title(f"{model_name} — Sample {idx} (proba={y_proba[idx]:.3f})")
        plt.tight_layout()
        plt.savefig(CHARTS_DIR / f"waterfall_{model_name}_sample_{idx:04d}.png", dpi=150)
        plt.close(fig)

    logger.info("  Charts: %d waterfall plots for %s", len(top_idx), model_name)


def plot_dae_breakdowns(
    weighted_err: np.ndarray,
    feat_names: list,
    y_pred: np.ndarray,
    recon_errors: np.ndarray,
) -> None:
    """Bar chart of per-feature error for top-N DAE anomalies."""
    anomaly_idx = np.where(y_pred == 1)[0]
    if len(anomaly_idx) == 0:
        logger.info("  No DAE anomalies, skipping breakdown plots")
        return

    top_idx = anomaly_idx[np.argsort(recon_errors[anomaly_idx])[-TOP_N_WATERFALL:]][::-1]

    for idx in top_idx:
        errs = weighted_err[idx]
        sorted_i = np.argsort(errs)[::-1][:TOP_K_FEATURES]
        names = [feat_names[i] for i in sorted_i][::-1]
        values = [float(errs[i]) for i in sorted_i][::-1]
        colors = [_feat_color(n) for n in names]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(names, values, color=colors)
        ax.set_xlabel("Weighted Reconstruction Error")
        ax.set_title(f"DAE — Sample {idx} (error={recon_errors[idx]:.6f})")
        ax.legend(handles=[
            Patch(facecolor="#C44E52", label="Network"),
            Patch(facecolor="#3274A1", label="Biometric"),
        ], loc="lower right")
        plt.tight_layout()
        plt.savefig(CHARTS_DIR / f"dae_error_breakdown_sample_{idx:04d}.png", dpi=150)
        plt.close(fig)

    logger.info("  Charts: %d DAE breakdown plots", len(top_idx))


def plot_beeswarm(
    model_name: str,
    sv: np.ndarray,
    X_test: np.ndarray,
    feat_names: list,
) -> None:
    """SHAP beeswarm (summary) plot — feature value vs SHAP impact."""
    explanation = shap.Explanation(
        values=sv,
        data=X_test,
        feature_names=feat_names,
    )
    fig = plt.figure(figsize=(10, 8))
    shap.plots.beeswarm(explanation, show=False, max_display=TOP_K_FEATURES)
    plt.title(f"{model_name} — SHAP Beeswarm (attack class)")
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / f"beeswarm_{model_name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Chart: beeswarm_%s.png", model_name)


def plot_force(
    model_name: str,
    sv: np.ndarray,
    expected: float,
    X_test: np.ndarray,
    feat_names: list,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> None:
    """SHAP force plots for top-N highest-confidence attack predictions."""
    attack_idx = np.where(y_pred == 1)[0]
    if len(attack_idx) == 0:
        return

    top_idx = attack_idx[np.argsort(y_proba[attack_idx])[-TOP_N_WATERFALL:]][::-1]

    for idx in top_idx:
        explanation = shap.Explanation(
            values=sv[idx],
            base_values=expected,
            data=X_test[idx],
            feature_names=feat_names,
        )
        fig = plt.figure(figsize=(14, 3))
        shap.plots.force(explanation, show=False, matplotlib=True)
        plt.title(f"{model_name} — Sample {idx} (proba={y_proba[idx]:.3f})", y=1.05)
        plt.tight_layout()
        plt.savefig(CHARTS_DIR / f"force_{model_name}_sample_{idx:04d}.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

    logger.info("  Charts: %d force plots for %s", len(top_idx), model_name)


def plot_per_category_importance(
    model_name: str,
    sv: np.ndarray,
    y_test: np.ndarray,
    attack_cats: np.ndarray | None,
    feat_names: list,
) -> dict:
    """Per-attack-category SHAP importance + bar charts."""
    if attack_cats is None:
        return {}

    categories = {}
    unique_cats = sorted(set(str(c) for c in attack_cats if c and str(c) != "normal"))

    for cat in unique_cats:
        mask = np.array([str(c) == cat for c in attack_cats]) & (y_test == 1)
        if mask.sum() == 0:
            continue

        mean_abs = np.mean(np.abs(sv[mask]), axis=0)
        ranked = sorted(zip(feat_names, mean_abs), key=lambda x: -x[1])
        importance = [
            {"rank": i + 1, "feature": n, "mean_abs_shap": round(float(v), 6)}
            for i, (n, v) in enumerate(ranked)
        ]
        categories[cat] = importance

        # Bar chart per category
        top = importance[:TOP_K_FEATURES]
        names = [f["feature"] for f in top][::-1]
        values = [f["mean_abs_shap"] for f in top][::-1]
        colors = [_feat_color(n) for n in names]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(names, values, color=colors)
        ax.set_xlabel("mean |SHAP|")
        ax.set_title(f"{model_name} — {cat} (n={mask.sum()}) Feature Importance")
        ax.legend(handles=[
            Patch(facecolor="#C44E52", label="Network"),
            Patch(facecolor="#3274A1", label="Biometric"),
        ], loc="lower right")
        plt.tight_layout()
        safe_cat = cat.replace(" ", "_").lower()
        plt.savefig(CHARTS_DIR / f"importance_{model_name}_{safe_cat}.png", dpi=150)
        plt.close(fig)

    if categories:
        path = OUTPUT_DIR / f"per_category_importance_{model_name}.json"
        path.write_text(json.dumps(
            {"model": model_name, "categories": categories}, indent=2,
        ), encoding="utf-8")
        logger.info("  Per-category importance: %d categories for %s",
                    len(categories), model_name)
    return categories


def plot_dae_global_weights(feat_weights: np.ndarray, feat_names: list) -> None:
    """Bar chart of DAE inverse-variance feature weights."""
    importance = sorted(
        [{"feature": n, "weight": float(w)} for n, w in zip(feat_names, feat_weights)],
        key=lambda x: -x["weight"],
    )
    top = importance[:TOP_K_FEATURES]
    names = [f["feature"] for f in top][::-1]
    values = [f["weight"] for f in top][::-1]
    colors = [_feat_color(n) for n in names]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(names, values, color=colors)
    ax.set_xlabel("Feature Weight (inverse variance)")
    ax.set_title("DAE — Feature Monitoring Weights")
    ax.legend(handles=[
        Patch(facecolor="#C44E52", label="Network"),
        Patch(facecolor="#3274A1", label="Biometric"),
    ], loc="lower right")
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "global_importance_dae.png", dpi=150)
    plt.close(fig)
    logger.info("  Chart: global_importance_dae.png")


# ── Stakeholder outputs ────────────────────────────────────────────────

def _severity(n_models_flagged: int) -> str:
    if n_models_flagged >= 4:
        return "CRITICAL"
    elif n_models_flagged == 3:
        return "HIGH"
    elif n_models_flagged == 2:
        return "MEDIUM"
    return "LOW"


def _top_features_shap(sv_row: np.ndarray, feat_names: list, k: int = 3) -> list:
    """Top-k features by |SHAP| for one sample."""
    abs_vals = np.abs(sv_row)
    top_i = np.argsort(abs_vals)[-k:][::-1]
    return [
        {
            "feature": feat_names[i],
            "shap_value": round(float(sv_row[i]), 6),
            "direction": "increases_risk" if sv_row[i] > 0 else "decreases_risk",
        }
        for i in top_i
    ]


def _top_features_dae(werr_row: np.ndarray, feat_names: list, k: int = 3) -> list:
    """Top-k features by weighted error for one DAE sample."""
    total = werr_row.sum()
    top_i = np.argsort(werr_row)[-k:][::-1]
    return [
        {
            "feature": feat_names[i],
            "weighted_error": round(float(werr_row[i]), 8),
            "pct_contribution": round(float(werr_row[i] / total * 100), 1) if total > 0 else 0.0,
        }
        for i in top_i
    ]


def build_analyst_report(
    all_shap: dict,
    all_preds: dict,
    weighted_err: np.ndarray,
    dae_preds: dict,
    feat_names: list,
    n_samples: int,
) -> list:
    """Build per-alert analyst report."""
    logger.info("Building analyst report...")
    alerts = []

    for idx in range(n_samples):
        models_flagged = []
        entry = {"sample_index": int(idx), "models": {}}

        # Track A
        for name in TRACK_A_MODELS:
            pred = int(all_preds[name]["y_pred"][idx])
            if pred == 1:
                models_flagged.append(name)
            entry["models"][name] = {
                "prediction": pred,
                "confidence": round(float(all_preds[name]["y_proba"][idx]), 4),
                "top_features": _top_features_shap(all_shap[name][idx], feat_names),
            }

        # Track B
        dae_pred = int(dae_preds["y_pred"][idx])
        if dae_pred == 1:
            models_flagged.append("dae")
        entry["models"]["dae"] = {
            "prediction": dae_pred,
            "reconstruction_error": round(float(dae_preds["reconstruction_error"][idx]), 8),
            "top_features": _top_features_dae(weighted_err[idx], feat_names),
        }

        if not models_flagged:
            continue

        entry["consensus"] = f"{len(models_flagged)}/4 models flagged"
        entry["severity"] = _severity(len(models_flagged))
        alerts.append(entry)

    path = OUTPUT_DIR / "analyst_report.json"
    path.write_text(json.dumps(alerts, indent=2), encoding="utf-8")
    logger.info("  Saved: %s (%d alerts)", path, len(alerts))
    return alerts


def build_clinician_summaries(
    all_shap: dict,
    all_preds: dict,
    dae_preds: dict,
    feat_names: list,
    n_samples: int,
) -> None:
    """Build plain-language clinician summaries for XGBoost-flagged alerts."""
    logger.info("Building clinician summaries...")
    summaries = []

    xgb_preds = all_preds["xgboost"]
    xgb_shap = all_shap["xgboost"]

    for idx in range(n_samples):
        if xgb_preds["y_pred"][idx] != 1:
            continue

        # Count models flagging this sample
        n_flagged = sum(
            1 for name in TRACK_A_MODELS if all_preds[name]["y_pred"][idx] == 1
        ) + (1 if dae_preds["y_pred"][idx] == 1 else 0)
        severity = _severity(n_flagged)

        top = _top_features_shap(xgb_shap[idx], feat_names, k=3)
        top_feature = top[0]["feature"]
        feature_type = "biometric" if top_feature in BIOMETRIC_FEATURES else "network"

        # Biometric safety note
        bio_feats = [f["feature"] for f in top if f["feature"] in BIOMETRIC_FEATURES]
        biometric_note = (
            f"Note: Biometric data ({', '.join(bio_feats)}) showed unusual values. "
            if bio_feats else ""
        )

        template = CLINICIAN_TEMPLATES[severity]
        summary = template.format(
            idx=idx,
            top_feature=top_feature,
            feature_type=feature_type,
            biometric_note=biometric_note,
        )
        summaries.append({
            "sample_index": int(idx),
            "severity": severity,
            "summary": summary,
        })

    path = OUTPUT_DIR / "clinician_summaries.json"
    path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    logger.info("  Saved: %s (%d summaries)", path, len(summaries))


def build_admin_dashboard(
    all_shap: dict,
    all_preds: dict,
    dae_preds: dict,
    feat_names: list,
    feat_weights: np.ndarray,
    global_importances: dict,
    attack_cats: np.ndarray | None,
    n_samples: int,
) -> None:
    """Build aggregated administrator dashboard data."""
    logger.info("Building admin dashboard...")

    # Count alerts per severity
    severity_counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
    agreement_counts = {"4_of_4": 0, "3_of_4": 0, "2_of_4": 0, "1_of_4": 0}

    for idx in range(n_samples):
        n_flagged = sum(
            1 for name in TRACK_A_MODELS if all_preds[name]["y_pred"][idx] == 1
        ) + (1 if dae_preds["y_pred"][idx] == 1 else 0)
        if n_flagged == 0:
            continue
        severity_counts[_severity(n_flagged)] += 1
        agreement_counts[f"{n_flagged}_of_4"] += 1

    total_alerts = sum(severity_counts.values())

    # Feature importance rankings per model
    rankings = {}
    for name, imp in global_importances.items():
        rankings[name] = imp[:TOP_K_FEATURES]

    # DAE rankings by weight
    dae_ranked = sorted(
        [{"rank": 0, "feature": n, "weight": round(float(w), 6)}
         for n, w in zip(feat_names, feat_weights)],
        key=lambda x: -x["weight"],
    )
    for i, entry in enumerate(dae_ranked):
        entry["rank"] = i + 1
    rankings["dae"] = dae_ranked[:TOP_K_FEATURES]

    # Biometric vs network in top-5
    bio_net = {}
    for name, imp in global_importances.items():
        top5 = [f["feature"] for f in imp[:5]]
        bio_net[name] = {
            "biometric_in_top5": sum(1 for f in top5 if f in BIOMETRIC_FEATURES),
            "network_in_top5": sum(1 for f in top5 if f not in BIOMETRIC_FEATURES),
        }
    dae_top5 = [e["feature"] for e in dae_ranked[:5]]
    bio_net["dae"] = {
        "biometric_in_top5": sum(1 for f in dae_top5 if f in BIOMETRIC_FEATURES),
        "network_in_top5": sum(1 for f in dae_top5 if f not in BIOMETRIC_FEATURES),
    }

    # Alerts by attack category
    cat_counts = {}
    if attack_cats is not None:
        xgb_preds = all_preds["xgboost"]["y_pred"]
        for idx in range(n_samples):
            if xgb_preds[idx] == 1 and attack_cats[idx]:
                cat = str(attack_cats[idx])
                cat_counts[cat] = cat_counts.get(cat, 0) + 1

    dashboard = {
        "total_samples": int(n_samples),
        "total_alerts": total_alerts,
        "alerts_by_severity": severity_counts,
        "model_agreement": agreement_counts,
        "feature_importance_rankings": rankings,
        "biometric_vs_network": bio_net,
        "alerts_by_attack_category": cat_counts,
    }

    path = OUTPUT_DIR / "admin_dashboard.json"
    path.write_text(json.dumps(dashboard, indent=2), encoding="utf-8")
    logger.info("  Saved: %s", path)


# ── 4.4 Feature-to-concept mapping ─────────────────────────────────────

FEATURE_CONCEPTS = {
    "Flgs":        {"label": "TCP Flag Pattern",         "category": "network",   "direction_high": "unusual flag combination detected",        "direction_low": "normal flag pattern"},
    "Sport":       {"label": "Source Port",              "category": "network",   "direction_high": "abnormal source port used",                "direction_low": "standard port range"},
    "SrcBytes":    {"label": "Outbound Byte Volume",     "category": "network",   "direction_high": "unusually high outbound data volume",      "direction_low": "minimal outbound traffic"},
    "DstBytes":    {"label": "Inbound Byte Volume",      "category": "network",   "direction_high": "unusually high inbound data volume",       "direction_low": "minimal inbound traffic"},
    "SrcLoad":     {"label": "Source Load",              "category": "network",   "direction_high": "high source bandwidth utilization",         "direction_low": "normal source load"},
    "DstLoad":     {"label": "Destination Load",         "category": "network",   "direction_high": "high destination bandwidth utilization",    "direction_low": "normal destination load"},
    "SIntPkt":     {"label": "Source Inter-Packet Gap",  "category": "network",   "direction_high": "abnormal packet timing (slow)",             "direction_low": "rapid packet bursts"},
    "DIntPkt":     {"label": "Dest Inter-Packet Gap",    "category": "network",   "direction_high": "abnormal response timing",                 "direction_low": "unusually fast responses"},
    "SIntPktAct":  {"label": "Active Inter-Packet Time", "category": "network",   "direction_high": "extended active session timing",            "direction_low": "brief session activity"},
    "sMaxPktSz":   {"label": "Max Source Packet Size",   "category": "network",   "direction_high": "large packets sent",                       "direction_low": "small packet sizes"},
    "dMaxPktSz":   {"label": "Max Dest Packet Size",     "category": "network",   "direction_high": "large packets received",                   "direction_low": "small packet sizes"},
    "sMinPktSz":   {"label": "Min Source Packet Size",   "category": "network",   "direction_high": "varying source packet sizes",              "direction_low": "consistent small packets"},
    "Dur":         {"label": "Flow Duration",            "category": "network",   "direction_high": "prolonged connection duration",             "direction_low": "unusually brief connection"},
    "TotBytes":    {"label": "Total Byte Volume",        "category": "network",   "direction_high": "high total data transfer",                 "direction_low": "minimal data transferred"},
    "Load":        {"label": "Network Load",             "category": "network",   "direction_high": "high network utilization",                  "direction_low": "low network activity"},
    "pSrcLoss":    {"label": "Source Packet Loss",       "category": "network",   "direction_high": "significant packet loss from source",       "direction_low": "no source packet loss"},
    "pDstLoss":    {"label": "Dest Packet Loss",         "category": "network",   "direction_high": "significant packet loss at destination",    "direction_low": "no destination packet loss"},
    "Temp":        {"label": "Body Temperature",         "category": "biometric", "direction_high": "elevated temperature reading",              "direction_low": "below-normal temperature"},
    "SpO2":        {"label": "Blood Oxygen Saturation",  "category": "biometric", "direction_high": "normal SpO2",                              "direction_low": "dangerously low oxygen saturation"},
    "Pulse_Rate":  {"label": "Pulse Rate",               "category": "biometric", "direction_high": "elevated heart rate (tachycardia)",         "direction_low": "low heart rate (bradycardia)"},
    "SYS":         {"label": "Systolic Blood Pressure",  "category": "biometric", "direction_high": "elevated systolic BP (hypertension)",       "direction_low": "low systolic BP (hypotension)"},
    "DIA":         {"label": "Diastolic Blood Pressure",  "category": "biometric", "direction_high": "elevated diastolic BP",                    "direction_low": "low diastolic BP"},
    "Heart_rate":  {"label": "Heart Rate",               "category": "biometric", "direction_high": "elevated heart rate",                      "direction_low": "low heart rate"},
    "Resp_Rate":   {"label": "Respiratory Rate",         "category": "biometric", "direction_high": "rapid breathing (tachypnea)",              "direction_low": "slow breathing (bradypnea)"},
    "ST":          {"label": "ST Segment (ECG)",         "category": "biometric", "direction_high": "ST elevation (possible cardiac event)",     "direction_low": "ST depression"},
}


def export_feature_concepts() -> None:
    """Export feature-to-concept mapping as standalone JSON (Task 4.4)."""
    path = OUTPUT_DIR / "feature_concepts.json"
    path.write_text(json.dumps(FEATURE_CONCEPTS, indent=2), encoding="utf-8")
    logger.info("  Saved: feature_concepts.json")


# ── 4.6 NLG template library ──────────────────────────────────────────

NLG_TEMPLATES = {
    "severity_header": {
        "CRITICAL": "CRITICAL SECURITY ALERT — Immediate action required.",
        "HIGH":     "HIGH PRIORITY ALERT — Active investigation needed.",
        "MEDIUM":   "MODERATE ALERT — Flagged for review.",
        "LOW":      "LOW PRIORITY — Logged for audit.",
    },
    "detection_sentence": (
        "The intrusion detection system flagged this network flow with "
        "{consensus} using a confidence of {confidence:.0%}."
    ),
    "feature_explanation_network": (
        "{label} showed {direction} "
        "(SHAP contribution: {shap_value:+.3f})."
    ),
    "feature_explanation_biometric": (
        "Patient {label} showed {direction} "
        "(SHAP contribution: {shap_value:+.3f}). "
        "Clinical review of this vital sign is recommended."
    ),
    "risk_context": (
        "Composite risk score: {risk_score:.2f} ({risk_level}). "
        "Components — detection confidence: {c_detect:.2f}, "
        "device criticality: {d_crit:.2f}, "
        "data sensitivity: {s_data:.2f}, "
        "patient acuity: {a_patient:.2f}."
    ),
    "acuity_note_normal": "Patient vitals are within normal ranges.",
    "acuity_note_abnormal": (
        "Note: {n_abnormal} of 8 biometric readings are outside normal range. "
        "Verify patient condition independently."
    ),
    "action_recommendation": {
        "CRITICAL": "Recommended: Isolate device immediately. Page on-call physician and CISO. Initiate incident response.",
        "HIGH":     "Recommended: Isolate network segment. Notify SOC and biomedical engineering.",
        "MEDIUM":   "Recommended: Enable enhanced monitoring. Queue for security team review.",
        "LOW":      "Recommended: No immediate action. Review at next security audit.",
    },
}


def export_nlg_templates() -> None:
    """Export NLG template library as JSON (Task 4.6)."""
    path = OUTPUT_DIR / "nlg_templates.json"
    path.write_text(json.dumps(NLG_TEMPLATES, indent=2), encoding="utf-8")
    logger.info("  Saved: nlg_templates.json")


# ── 4.7 Clinician NLG engine (6-step assembly) ────────────────────────

def generate_clinician_alert(
    idx: int,
    sv_row: np.ndarray,
    feat_names: list,
    severity: str,
    confidence: float,
    consensus: str,
    risk_score: float = 0.0,
    risk_components: dict | None = None,
    a_patient_val: float = 0.0,
) -> str:
    """6-step NLG assembly for clinician-facing alert."""
    parts = []

    # Step 1: Severity header
    parts.append(NLG_TEMPLATES["severity_header"].get(severity, severity))

    # Step 2: Detection sentence
    parts.append(NLG_TEMPLATES["detection_sentence"].format(
        consensus=consensus, confidence=confidence,
    ))

    # Step 3: Top-5 feature explanations
    abs_vals = np.abs(sv_row)
    top_idx = np.argsort(abs_vals)[-5:][::-1]
    for fi in top_idx:
        fname = feat_names[fi]
        concept = FEATURE_CONCEPTS.get(fname, {})
        label = concept.get("label", fname)
        cat = concept.get("category", "network")
        direction = concept.get(
            "direction_high" if sv_row[fi] > 0 else "direction_low",
            "abnormal value",
        )
        template_key = f"feature_explanation_{cat}"
        parts.append(NLG_TEMPLATES[template_key].format(
            label=label, direction=direction, shap_value=float(sv_row[fi]),
        ))

    # Step 4: Risk context
    if risk_components:
        parts.append(NLG_TEMPLATES["risk_context"].format(
            risk_score=risk_score, risk_level=severity, **risk_components,
        ))

    # Step 5: Patient acuity note
    if a_patient_val > 0:
        n_abnormal = int(round(a_patient_val * 8))
        parts.append(NLG_TEMPLATES["acuity_note_abnormal"].format(n_abnormal=n_abnormal))
    else:
        parts.append(NLG_TEMPLATES["acuity_note_normal"])

    # Step 6: Action recommendation
    parts.append(NLG_TEMPLATES["action_recommendation"].get(severity, ""))

    return "\n\n".join(p for p in parts if p)


# ── 4.10 Stakeholder router ───────────────────────────────────────────

def route_explanation(
    idx: int,
    stakeholder_role: str,
    sv_row: np.ndarray,
    feat_names: list,
    severity: str,
    confidence: float,
    consensus: str,
    risk_score: float,
    risk_components: dict,
    a_patient_val: float,
    dae_top_features: list,
) -> dict:
    """Route alert to correct stakeholder view."""
    if stakeholder_role == "clinician":
        return {
            "role": "clinician",
            "format": "text",
            "content": generate_clinician_alert(
                idx, sv_row, feat_names, severity, confidence, consensus,
                risk_score, risk_components, a_patient_val,
            ),
        }
    elif stakeholder_role == "analyst":
        return {
            "role": "analyst",
            "format": "json",
            "content": {
                "sample_index": idx,
                "severity": severity,
                "consensus": consensus,
                "top_features_shap": _top_features_shap(sv_row, feat_names, k=5),
                "dae_top_features": dae_top_features,
                "risk_score": risk_score,
                "risk_components": risk_components,
                "charts": [
                    f"waterfall_xgboost_sample_{idx:04d}.png",
                    f"force_xgboost_sample_{idx:04d}.png",
                ],
            },
        }
    elif stakeholder_role == "administrator":
        return {
            "role": "administrator",
            "format": "json",
            "content": {
                "sample_index": idx,
                "severity": severity,
                "risk_score": risk_score,
                "risk_level": severity,
                "action_required": severity in ("CRITICAL", "HIGH"),
                "global_charts": [
                    "global_importance_xgboost.png",
                    "beeswarm_xgboost.png",
                ],
            },
        }
    return {"role": stakeholder_role, "format": "text", "content": "Unknown role"}


# ── 4.11 Example explanations for thesis ──────────────────────────────

def generate_example_explanations(
    all_shap: dict,
    all_preds: dict,
    dae_preds: dict,
    weighted_err: np.ndarray,
    feat_names: list,
    y_test: np.ndarray,
    attack_cats: np.ndarray | None,
) -> list:
    """Generate multi-view examples for 5 alerts across all 3 stakeholder views."""
    logger.info("Generating example explanations for thesis figures...")

    xgb_sv = all_shap["xgboost"]
    xgb_preds = all_preds["xgboost"]

    # Pick 5 diverse alerts: highest confidence, one per category, one borderline
    attack_idx = np.where(xgb_preds["y_pred"] == 1)[0]
    if len(attack_idx) == 0:
        return []

    # Top-2 by confidence
    sorted_by_conf = attack_idx[np.argsort(xgb_preds["y_proba"][attack_idx])[::-1]]
    picks = list(sorted_by_conf[:2])

    # One spoofing, one data alteration (if available)
    if attack_cats is not None:
        for cat in ["Spoofing", "Data Alteration"]:
            cat_idx = [i for i in attack_idx if str(attack_cats[i]) == cat and i not in picks]
            if cat_idx:
                picks.append(cat_idx[0])

    # One borderline (lowest confidence attack)
    borderline = sorted_by_conf[-1]
    if borderline not in picks:
        picks.append(borderline)

    picks = picks[:5]

    # Load risk scores if available
    risk_data = {}
    try:
        rd = np.load(PROJECT_ROOT / "data/phase2/risk_scores/risk_scores.npz", allow_pickle=True)
        risk_data = {k: rd[k] for k in rd.files}
    except FileNotFoundError:
        pass

    examples = []
    for idx in picks:
        sv_row = xgb_sv[idx]
        confidence = float(xgb_preds["y_proba"][idx])
        n_flagged = sum(1 for name in all_preds if all_preds[name]["y_pred"][idx] == 1)
        n_flagged += 1 if dae_preds["y_pred"][idx] == 1 else 0
        severity = "CRITICAL" if n_flagged >= 4 else "HIGH" if n_flagged == 3 else "MEDIUM" if n_flagged == 2 else "LOW"
        consensus = f"{n_flagged}/4 models flagged"

        dae_top = _top_features_dae(weighted_err[idx], feat_names, k=3)

        risk_score = float(risk_data["R"][idx]) if "R" in risk_data else 0.0
        risk_comps = {}
        if "c_detect" in risk_data:
            risk_comps = {
                "c_detect": float(risk_data["c_detect"][idx]),
                "d_crit": float(risk_data["d_crit"][idx]),
                "s_data": float(risk_data["s_data"][idx]),
                "a_patient": float(risk_data["a_patient"][idx]),
            }
        a_pat = float(risk_data["a_patient"][idx]) if "a_patient" in risk_data else 0.0

        example = {
            "sample_index": int(idx),
            "ground_truth": "attack" if y_test[idx] == 1 else "benign",
            "attack_category": str(attack_cats[idx]) if attack_cats is not None else "unknown",
            "views": {},
        }

        for role in ["clinician", "analyst", "administrator"]:
            example["views"][role] = route_explanation(
                int(idx), role, sv_row, feat_names, severity, confidence,
                consensus, risk_score, risk_comps, a_pat, dae_top,
            )

        examples.append(example)
        logger.info("  Example: sample %d (%s, %s) — %s",
                    idx, example["attack_category"], severity, consensus)

    path = OUTPUT_DIR / "example_explanations.json"
    path.write_text(json.dumps(examples, indent=2, default=str), encoding="utf-8")
    logger.info("  Saved: example_explanations.json (%d examples)", len(examples))
    return examples


# ── Validation ──────────────────────────────────────────────────────────

def validate_consistency(
    all_shap: dict,
    feat_names: list,
) -> dict:
    """Compare SHAP feature rankings with native feature_importances_.

    For tree models, sklearn provides a gini/gain-based importance.
    SHAP should roughly agree on top features, though ordering may
    differ since SHAP accounts for interactions.
    """
    logger.info("Running consistency check (SHAP vs native importances)...")
    results = {}

    for name, cfg in TRACK_A_MODELS.items():
        pipeline = joblib.load(PROJECT_ROOT / cfg["pipeline"])
        clf = pipeline.named_steps["classifier"]

        if not hasattr(clf, "feature_importances_"):
            continue

        native_imp = clf.feature_importances_
        native_ranked = [feat_names[i] for i in np.argsort(native_imp)[::-1]]

        shap_mean_abs = np.mean(np.abs(all_shap[name]), axis=0)
        shap_ranked = [feat_names[i] for i in np.argsort(shap_mean_abs)[::-1]]

        # Top-5 overlap
        native_top5 = set(native_ranked[:5])
        shap_top5 = set(shap_ranked[:5])
        overlap = native_top5 & shap_top5

        # Rank correlation (Spearman)
        from scipy.stats import spearmanr
        native_ranks = np.argsort(np.argsort(-native_imp))
        shap_ranks = np.argsort(np.argsort(-shap_mean_abs))
        rho, p_val = spearmanr(native_ranks, shap_ranks)

        results[name] = {
            "native_top5": native_ranked[:5],
            "shap_top5": shap_ranked[:5],
            "top5_overlap": sorted(overlap),
            "top5_overlap_count": len(overlap),
            "spearman_rho": round(float(rho), 4),
            "spearman_p_value": round(float(p_val), 6),
        }
        logger.info("  %s: top-5 overlap=%d/5, Spearman rho=%.4f (p=%.4f)",
                    name, len(overlap), rho, p_val)

    path = OUTPUT_DIR / "validation_consistency.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info("  Saved: %s", path)
    return results


def validate_perturbation(
    all_shap: dict,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feat_names: list,
    top_n_features: int = 5,
) -> dict:
    """Mask top-N SHAP features and measure accuracy drop.

    If SHAP explanations are faithful, masking the most important
    features should cause a significant performance drop.
    """
    from sklearn.metrics import f1_score

    logger.info("Running perturbation test (mask top-%d features)...", top_n_features)
    results = {}

    for name, cfg in TRACK_A_MODELS.items():
        pipeline = joblib.load(PROJECT_ROOT / cfg["pipeline"])
        clf = pipeline.named_steps["classifier"]

        # Baseline predictions
        y_proba_base = clf.predict_proba(X_test)[:, 1]

        # Load optimal threshold from report
        with open(PROJECT_ROOT / cfg["report"]) as f:
            threshold = json.load(f)["optimal_threshold"]

        y_pred_base = (y_proba_base >= threshold).astype(int)
        f1_base = f1_score(y_test, y_pred_base, pos_label=1)

        # Identify top-N features by mean |SHAP|
        shap_mean = np.mean(np.abs(all_shap[name]), axis=0)
        top_feat_idx = np.argsort(shap_mean)[-top_n_features:]

        # Mask: replace top features with their training mean (neutral value)
        X_masked = X_test.copy()
        for fi in top_feat_idx:
            X_masked[:, fi] = np.mean(X_test[:, fi])

        y_proba_masked = clf.predict_proba(X_masked)[:, 1]
        y_pred_masked = (y_proba_masked >= threshold).astype(int)
        f1_masked = f1_score(y_test, y_pred_masked, pos_label=1)

        drop = f1_base - f1_masked
        drop_pct = (drop / f1_base * 100) if f1_base > 0 else 0.0

        results[name] = {
            "top_features_masked": [feat_names[i] for i in top_feat_idx],
            "f1_baseline": round(float(f1_base), 4),
            "f1_after_masking": round(float(f1_masked), 4),
            "f1_drop": round(float(drop), 4),
            "f1_drop_pct": round(float(drop_pct), 1),
            "faithful": drop_pct > 5.0,
        }
        logger.info("  %s: F1 %.4f → %.4f (drop=%.1f%%) %s",
                    name, f1_base, f1_masked, drop_pct,
                    "FAITHFUL" if drop_pct > 5.0 else "WEAK")

    path = OUTPUT_DIR / "validation_perturbation.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info("  Saved: %s", path)
    return results


def validate_cross_model(
    global_importances: dict,
) -> dict:
    """Compare SHAP rankings across models.

    Consistent rankings strengthen confidence; divergent rankings
    may indicate model-specific artifacts.
    """
    from scipy.stats import spearmanr

    logger.info("Running cross-model ranking comparison...")

    model_names = list(global_importances.keys())
    # Build rank vectors keyed by feature name
    rank_vectors = {}
    for name, imp in global_importances.items():
        rank_vectors[name] = {f["feature"]: f["rank"] for f in imp}

    all_features = list(rank_vectors[model_names[0]].keys())

    comparisons = {}
    for i, m1 in enumerate(model_names):
        for m2 in model_names[i + 1:]:
            ranks1 = [rank_vectors[m1][f] for f in all_features]
            ranks2 = [rank_vectors[m2][f] for f in all_features]
            rho, p_val = spearmanr(ranks1, ranks2)

            top5_1 = set(f for f, r in rank_vectors[m1].items() if r <= 5)
            top5_2 = set(f for f, r in rank_vectors[m2].items() if r <= 5)
            overlap = top5_1 & top5_2

            pair = f"{m1}_vs_{m2}"
            comparisons[pair] = {
                "spearman_rho": round(float(rho), 4),
                "spearman_p_value": round(float(p_val), 6),
                "top5_overlap": sorted(overlap),
                "top5_overlap_count": len(overlap),
            }
            logger.info("  %s vs %s: rho=%.4f, top-5 overlap=%d/5",
                        m1, m2, rho, len(overlap))

    # Consensus top features (in top-5 across all models)
    from collections import Counter
    all_top5 = []
    for name in model_names:
        all_top5.extend(f for f, r in rank_vectors[name].items() if r <= 5)
    consensus = [feat for feat, cnt in Counter(all_top5).most_common()
                 if cnt == len(model_names)]

    result = {
        "pairwise_comparisons": comparisons,
        "consensus_top5_all_models": consensus,
    }

    path = OUTPUT_DIR / "validation_cross_model.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger.info("  Consensus features (top-5 in all models): %s", consensus)
    logger.info("  Saved: %s", path)
    return result


# ── Main ────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    sep = "=" * 72
    t0 = time.perf_counter()

    logger.info(sep)
    logger.info("MODULE 4 — GENERATE EXPLANATIONS (RQ1/RO1)")
    logger.info(sep)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load test data
    X_test, y_test, attack_cats, feat_names = load_test_data()
    n_samples = len(y_test)
    logger.info("Test data: %d samples, %d features", n_samples, len(feat_names))

    # ── Track A: TreeSHAP ──
    all_shap = {}
    all_preds = {}
    global_importances = {}

    for name, cfg in TRACK_A_MODELS.items():
        sv, expected = compute_tree_shap(
            name, PROJECT_ROOT / cfg["pipeline"], X_test, feat_names,
        )
        save_shap_values(name, sv, expected, feat_names)

        importance = compute_global_importance(sv, feat_names)
        save_global_importance(name, importance)
        global_importances[name] = importance

        plot_global_importance_bar(name, importance)

        preds = load_predictions(PROJECT_ROOT / cfg["predictions"])
        all_preds[name] = preds
        all_shap[name] = sv

        plot_waterfalls(name, sv, expected, X_test, feat_names,
                        preds["y_pred"], preds["y_proba"])

        plot_beeswarm(name, sv, X_test, feat_names)

        plot_force(name, sv, expected, X_test, feat_names,
                   preds["y_pred"], preds["y_proba"])

        plot_per_category_importance(name, sv, y_test, attack_cats, feat_names)

    # ── Track B: DAE ──
    dae_path = PROJECT_ROOT / "data/phase2/dae/final/dae_detector.pkl"
    sq_err, weighted_err, feat_weights = compute_dae_feature_errors(
        dae_path, X_test, feat_names,
    )
    save_dae_errors(sq_err, weighted_err, feat_weights, feat_names)
    plot_dae_global_weights(feat_weights, feat_names)

    dae_preds = load_predictions(
        PROJECT_ROOT / "data/phase2/dae/final/test_predictions.npz"
    )
    plot_dae_breakdowns(weighted_err, feat_names,
                        dae_preds["y_pred"], dae_preds["reconstruction_error"])

    # ── Stakeholder outputs ──
    all_preds_with_dae = dict(all_preds)
    alerts = build_analyst_report(
        all_shap, all_preds, weighted_err, dae_preds, feat_names, n_samples,
    )
    build_clinician_summaries(
        all_shap, all_preds, dae_preds, feat_names, n_samples,
    )
    build_admin_dashboard(
        all_shap, all_preds, dae_preds, feat_names, feat_weights,
        global_importances, attack_cats, n_samples,
    )

    # ── Export feature concepts + NLG templates (Tasks 4.4, 4.6) ──
    export_feature_concepts()
    export_nlg_templates()

    # ── Example explanations for thesis (Tasks 4.10, 4.11) ──
    generate_example_explanations(
        all_shap, all_preds, dae_preds, weighted_err,
        feat_names, y_test, attack_cats,
    )

    # ── Validation ──
    logger.info("")
    logger.info("── Explanation Validation ──")
    validate_consistency(all_shap, feat_names)
    validate_perturbation(all_shap, X_test, y_test, feat_names)
    validate_cross_model(global_importances)

    # ── Summary ──
    elapsed = round(time.perf_counter() - t0, 1)
    logger.info("")
    logger.info(sep)
    logger.info("EXPLANATIONS COMPLETE — %.1fs", elapsed)
    logger.info(sep)
    logger.info("  Output dir    : %s", OUTPUT_DIR)
    logger.info("  SHAP files    : %d models", len(TRACK_A_MODELS))
    logger.info("  DAE errors    : dae_feature_errors.npz")
    logger.info("  Analyst alerts: %d", len(alerts))
    logger.info("  Charts        : %s", CHARTS_DIR)
    logger.info(sep)


if __name__ == "__main__":
    main()
