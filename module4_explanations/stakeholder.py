"""Stakeholder-tailored output builders (analyst / clinician / admin).

Three builders take the already-computed SHAP + DAE outputs and pivot
them into per-stakeholder views. No compute, no plotting — pure
serialisation + light aggregation.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .compute import _top_features_dae, _top_features_shap
from .config import (
    BIOMETRIC_FEATURES,
    CLINICIAN_TEMPLATES,
    SHAP_MODELS,
    TOP_K_FEATURES,
    TRACK_A_MODELS,
    format_clinician_template,
)
from .feature_groups import _feature_to_narrative
from .io import OUTPUT_DIR, write_json_strict

logger = logging.getLogger(__name__)


def _severity(n_models_flagged: int) -> str:
    """Map ``n_flagged`` ∈ {0..4} to severity tier."""
    if n_models_flagged >= 4:
        return "CRITICAL"
    if n_models_flagged == 3:
        return "HIGH"
    if n_models_flagged == 2:
        return "MEDIUM"
    return "LOW"


# ── 1. Analyst report ────────────────────────────────────────────────


def build_analyst_report(
    all_shap: dict,
    all_preds: dict,
    weighted_err: np.ndarray,
    dae_preds: dict,
    feat_names: list,
    *,
    suffix: str = "",
    output_dir: Path | None = None,
) -> list:
    """Build per-alert analyst report.

    Y5: ``n_samples`` arg removed — derived from prediction array length.
    """
    logger.info("Building analyst report...")
    alerts = []
    out_dir = output_dir or OUTPUT_DIR

    # Pre-compute per-sample flag counts; iterate only flagged indices.
    pred_matrix = np.column_stack(
        [all_preds[name]["y_pred"] for name in TRACK_A_MODELS] + [dae_preds["y_pred"]]
    )
    n_flagged_all = pred_matrix.sum(axis=1)
    flagged_indices = np.where(n_flagged_all > 0)[0]

    for idx in flagged_indices:
        idx = int(idx)
        entry = {"sample_index": idx, "models": {}}

        # Track A
        for name in TRACK_A_MODELS:
            pred = int(all_preds[name]["y_pred"][idx])
            model_entry = {
                "prediction": pred,
                "confidence": round(float(all_preds[name]["y_proba"][idx]), 4),
            }
            if name in all_shap:
                model_entry["top_features"] = _top_features_shap(
                    all_shap[name][idx], feat_names,
                )
            entry["models"][name] = model_entry

        # Track B
        dae_pred = int(dae_preds["y_pred"][idx])
        entry["models"]["dae"] = {
            "prediction": dae_pred,
            "reconstruction_error": round(
                float(dae_preds["reconstruction_error"][idx]), 8,
            ),
            "top_features": _top_features_dae(weighted_err[idx], feat_names),
        }

        entry["consensus"] = f"{int(n_flagged_all[idx])}/4 models flagged"
        entry["severity"] = _severity(int(n_flagged_all[idx]))
        alerts.append(entry)

    path = out_dir / f"analyst_report{suffix}.json"
    write_json_strict(path, alerts)
    logger.info("  Saved: %s (%d alerts)", path, len(alerts))
    return alerts


# ── 2. Clinician summaries ───────────────────────────────────────────


def build_clinician_summaries(
    all_shap: dict,
    all_preds: dict,
    dae_preds: dict,
    feat_names: list,
    *,
    suffix: str = "",
    output_dir: Path | None = None,
) -> list:
    """Build plain-language clinician summaries for XGBoost-flagged alerts.

    Confidence-band logic: when the top-2 SHAP feature is ≥80% of the
    top-1's magnitude, cite both in the narrative.
    """
    logger.info("Building clinician summaries...")
    summaries = []
    out_dir = output_dir or OUTPUT_DIR

    xgb_preds = all_preds["xgboost"]
    xgb_shap = all_shap["xgboost"]

    pred_matrix = np.column_stack(
        [all_preds[name]["y_pred"] for name in TRACK_A_MODELS] + [dae_preds["y_pred"]]
    )
    n_flagged_all = pred_matrix.sum(axis=1)

    for idx in np.where(xgb_preds["y_pred"] == 1)[0]:
        idx = int(idx)
        severity = _severity(int(n_flagged_all[idx]))

        top = _top_features_shap(xgb_shap[idx], feat_names, k=3)
        top1_feat = top[0]["feature"]
        top1_val = abs(top[0]["shap_value"])
        narrative, category = _feature_to_narrative(top1_feat)

        secondary_note = ""
        if len(top) >= 2:
            top2_val = abs(top[1]["shap_value"])
            if top1_val > 0 and top2_val / top1_val > 0.8:
                narrative_2, cat_2 = _feature_to_narrative(top[1]["feature"])
                if category != cat_2:
                    secondary_note = (
                        f"A secondary indicator ({narrative_2}) also contributed. "
                    )
            bio_feats = [
                f["feature"] for f in top if f["feature"] in BIOMETRIC_FEATURES
            ]
            if bio_feats and category != "biometric":
                secondary_note += (
                    f"Note: Biometric data ({', '.join(bio_feats)}) "
                    "showed unusual values. "
                )

        summary = format_clinician_template(
            severity,
            sample_index=idx,
            narrative=narrative,
            secondary_note=secondary_note,
        )
        summaries.append(
            {"sample_index": int(idx), "severity": severity, "summary": summary}
        )

    path = out_dir / f"clinician_summaries{suffix}.json"
    write_json_strict(path, summaries)
    logger.info("  Saved: %s (%d summaries)", path, len(summaries))
    return summaries


# ── 3. Admin dashboard ──────────────────────────────────────────────


def build_admin_dashboard(
    all_shap: dict,
    all_preds: dict,
    dae_preds: dict,
    feat_names: list,
    feat_weights: np.ndarray,
    global_importances: dict,
    attack_cats: np.ndarray | None,
    *,
    output_dir: Path | None = None,
) -> dict:
    """Build aggregated administrator dashboard data.

    N7 fix: ``cat_counts`` computed via vectorised ``np.unique`` instead
    of the Python ``for idx in range(n_samples)`` loop.
    Y5 fix: ``n_samples`` removed; derived from arrays.
    """
    logger.info("Building admin dashboard...")
    out_dir = output_dir or OUTPUT_DIR

    pred_matrix = np.column_stack(
        [all_preds[name]["y_pred"] for name in TRACK_A_MODELS] + [dae_preds["y_pred"]]
    )
    n_flagged_all = pred_matrix.sum(axis=1)
    n_samples = int(pred_matrix.shape[0])
    flagged_mask = n_flagged_all > 0

    severity_counts = {
        "CRITICAL": int(((n_flagged_all >= 4) & flagged_mask).sum()),
        "HIGH":     int(((n_flagged_all == 3) & flagged_mask).sum()),
        "MEDIUM":   int(((n_flagged_all == 2) & flagged_mask).sum()),
        "LOW":      int(((n_flagged_all == 1) & flagged_mask).sum()),
    }
    agreement_counts = {
        f"{k}_of_4": int((n_flagged_all == k).sum()) for k in range(1, 5)
    }
    total_alerts = sum(severity_counts.values())

    # Per-model feature rankings
    rankings = {name: imp[:TOP_K_FEATURES] for name, imp in global_importances.items()}

    # DAE rankings by feature weight
    dae_ranked = sorted(
        [
            {"rank": 0, "feature": n, "weight": round(float(w), 6)}
            for n, w in zip(feat_names, feat_weights)
        ],
        key=lambda x: -x["weight"],
    )
    for i, entry in enumerate(dae_ranked):
        entry["rank"] = i + 1
    rankings["dae"] = dae_ranked[:TOP_K_FEATURES]

    # Biometric vs network in top-5 per model
    bio_net = {}
    for name, imp in global_importances.items():
        top5 = [f["feature"] for f in imp[:5]]
        bio_net[name] = {
            "biometric_in_top5": sum(1 for f in top5 if f in BIOMETRIC_FEATURES),
            "network_in_top5":   sum(1 for f in top5 if f not in BIOMETRIC_FEATURES),
        }
    dae_top5 = [e["feature"] for e in dae_ranked[:5]]
    bio_net["dae"] = {
        "biometric_in_top5": sum(1 for f in dae_top5 if f in BIOMETRIC_FEATURES),
        "network_in_top5":   sum(1 for f in dae_top5 if f not in BIOMETRIC_FEATURES),
    }

    # Alerts by attack category — vectorised
    cat_counts: dict[str, int] = {}
    if attack_cats is not None:
        xgb_preds = all_preds["xgboost"]["y_pred"]
        mask = (xgb_preds == 1) & np.array(
            [c is not None and str(c) != "None" for c in attack_cats]
        )
        if mask.any():
            cats_arr = np.asarray(attack_cats)[mask].astype(str)
            uniq, counts = np.unique(cats_arr, return_counts=True)
            cat_counts = {str(u): int(c) for u, c in zip(uniq, counts)}

    dashboard = {
        "total_samples": int(n_samples),
        "total_alerts": total_alerts,
        "alerts_by_severity": severity_counts,
        "model_agreement": agreement_counts,
        "feature_importance_rankings": rankings,
        "biometric_vs_network": bio_net,
        "alerts_by_attack_category": cat_counts,
    }

    path = out_dir / "admin_dashboard.json"
    write_json_strict(path, dashboard)
    logger.info("  Saved: %s", path)
    return dashboard


__all__ = [
    "_severity",
    "build_analyst_report",
    "build_clinician_summaries",
    "build_admin_dashboard",
]
