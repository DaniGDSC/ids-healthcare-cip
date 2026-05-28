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
from .feature_groups import _feature_to_narrative, observation_phrase
from .io import OUTPUT_DIR, write_json_strict

logger = logging.getLogger(__name__)


# ── 1. Analyst report ────────────────────────────────────────────────


def build_analyst_report(
    all_shap: dict,
    all_preds: dict,
    weighted_err: np.ndarray,
    dae_preds: dict,
    feat_names: list,
    risk_levels: np.ndarray,
    *,
    suffix: str = "",
    output_dir: Path | None = None,
    counterfactuals_by_idx: dict | None = None,
    stability_by_idx: dict | None = None,
) -> list:
    """Build per-alert analyst report.

    Severity is the Module 3 canonical ``risk_level`` (threshold of
    composite risk score), not a function of ``n_flagged``. An entry is
    emitted whenever any detector flagged the sample OR the Module 3
    risk_level is not LOW — so HIGH/CRITICAL alerts driven by D_crit /
    S_data / D_clinical_tier (rather than detector votes) still surface
    in the analyst view.
    """
    logger.info("Building analyst report...")
    alerts = []
    out_dir = output_dir or OUTPUT_DIR

    pred_matrix = np.column_stack(
        [all_preds[name]["y_pred"] for name in TRACK_A_MODELS] + [dae_preds["y_pred"]]
    )
    n_flagged_all = pred_matrix.sum(axis=1)
    n_detectors = int(pred_matrix.shape[1])
    risk_levels = np.asarray(risk_levels).astype(str)
    flagged_indices = np.where((n_flagged_all > 0) | (risk_levels != "LOW"))[0]

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

        entry["consensus"] = (
            f"{int(n_flagged_all[idx])}/{n_detectors} detectors flagged"
        )
        entry["severity"] = str(risk_levels[idx])
        entry["risk_level"] = str(risk_levels[idx])

        # Phase 2 — counterfactual ("what would have to change for the
        # alert NOT to fire"). Only attached when supplied by the caller
        # (the regen tool pre-computes them); legacy callers see the
        # pre-Phase-2 shape with no ``counterfactual`` field.
        if counterfactuals_by_idx is not None and idx in counterfactuals_by_idx:
            entry["counterfactual"] = counterfactuals_by_idx[idx]

        # Phase 4.1 — stability score for the SHAP top-K under input
        # noise. UNSTABLE alerts are flagged to downstream so they get
        # demoted from auto_execute to human review.
        if stability_by_idx is not None and idx in stability_by_idx:
            entry["stability"] = stability_by_idx[idx]

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
    risk_levels: np.ndarray,
    *,
    suffix: str = "",
    output_dir: Path | None = None,
    X_test: np.ndarray | None = None,
    counterfactuals_by_idx: dict | None = None,
    stability_by_idx: dict | None = None,
) -> list:
    """Build plain-language clinician summaries for XGBoost-flagged alerts.

    Severity is the Module 3 canonical ``risk_level``. Confidence-band
    logic: when the top-2 SHAP feature is ≥80% of the top-1's magnitude,
    cite both in the narrative.

    Phase 1.1 — when ``X_test`` is supplied, the top-1 narrative is
    grounded in a baseline-comparison clause computed from
    ``artifacts/feature_baselines.json`` (e.g. "~2.8 IQR-widths above
    benign baseline"). The clause is appended in-line so existing
    template wording stays intact; when ``X_test`` is None or the
    baseline file is missing, the summary falls back to the
    pre-Phase-1.1 category-only form.
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
    risk_levels = np.asarray(risk_levels).astype(str)

    for idx in np.where(xgb_preds["y_pred"] == 1)[0]:
        idx = int(idx)
        severity = str(risk_levels[idx])

        # Skip samples that the formula-fix detection gate downgraded to
        # NORMAL — these are XGBoost-flagged but the composite risk
        # didn't survive the upgraded Module 3 gate, so by definition
        # they're not user-facing alerts and shouldn't get a clinician
        # summary. (Pre-formula-fix this branch never fired because
        # ``assign_risk_levels`` never emitted NORMAL.)
        if severity == "NORMAL":
            continue

        top = _top_features_shap(xgb_shap[idx], feat_names, k=3)
        top1_feat = top[0]["feature"]
        top1_val = abs(top[0]["shap_value"])
        narrative, category = _feature_to_narrative(top1_feat)

        # Phase 1.1 — ground the narrative in an observed-value deviation.
        if X_test is not None and top1_feat in feat_names:
            col = feat_names.index(top1_feat)
            obs = observation_phrase(top1_feat, float(X_test[idx, col]))
            if obs:
                narrative = f"{narrative} {obs}"

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

        record = {"sample_index": int(idx), "severity": severity, "summary": summary}

        # Phase 2 — append a counterfactual clause + remediation hint so
        # the clinician sees what would have prevented the alert.
        if counterfactuals_by_idx is not None and idx in counterfactuals_by_idx:
            cf = counterfactuals_by_idx[idx]
            if cf.get("feasible"):
                # Try the project's canonical narrative formatter; fall back
                # to a minimal join if the counterfactual module isn't
                # importable in the caller's environment.
                try:
                    from .counterfactual import CounterfactualResult, counterfactual_narrative
                    clause = counterfactual_narrative(CounterfactualResult(
                        sparsity=cf["sparsity"], changes=cf["changes"],
                        flips_prediction=cf["flips_prediction"],
                        new_proba=cf["new_proba"],
                        original_proba=cf["original_proba"],
                        remediation_hint=cf.get("remediation_hint", ""),
                        feasible=cf["feasible"],
                    ))
                except Exception:
                    clause = ""
                if clause:
                    record["summary"] = f"{summary} {clause}"
                record["counterfactual"] = cf

        # Phase 4.1 — stability badge. Clinician sees the coloured band
        # only (no number); the analyst report carries the raw score.
        if stability_by_idx is not None and idx in stability_by_idx:
            stab = stability_by_idx[idx]
            try:
                from .stability import stability_badge
                badge = stability_badge(stab.get("band", ""))
                if badge:
                    record["summary"] = f"{record['summary']} {badge}"
            except Exception:
                pass
            record["stability"] = stab

        # Phase 3.1 — append a Markdown decision-tree playbook so the
        # clinician has a step-by-step procedure rather than a single
        # prescribed action. Selection is driven by the top SHAP
        # category and the canonical severity.
        try:
            from module5_responses.playbooks import render_markdown, select_playbook
            playbook = select_playbook(category, severity)
            record["summary"] = (
                f"{record['summary']}\n\n{render_markdown(playbook)}"
            )
            record["playbook"] = playbook.to_dict()
        except Exception:
            # Module 5 isn't importable in some test environments — fall
            # back to the pre-Phase-3 summary shape.
            pass

        summaries.append(record)

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
    risk_levels: np.ndarray,
    *,
    output_dir: Path | None = None,
) -> dict:
    """Build aggregated administrator dashboard data.

    ``alerts_by_severity`` now reflects the Module 3 canonical
    ``risk_level`` (composite-score thresholds), aggregated over the
    same alert population as ``build_analyst_report`` — i.e. samples
    flagged by any detector OR whose risk_level != LOW. ``model_agreement``
    keeps the per-N-of-4 detector-vote distribution, which is now an
    independent ensemble-agreement signal rather than a severity proxy.
    """
    logger.info("Building admin dashboard...")
    out_dir = output_dir or OUTPUT_DIR

    pred_matrix = np.column_stack(
        [all_preds[name]["y_pred"] for name in TRACK_A_MODELS] + [dae_preds["y_pred"]]
    )
    n_flagged_all = pred_matrix.sum(axis=1)
    n_samples = int(pred_matrix.shape[0])
    n_detectors = int(pred_matrix.shape[1])
    risk_levels = np.asarray(risk_levels).astype(str)
    flagged_mask = (n_flagged_all > 0) | (risk_levels != "LOW")

    severity_counts = {tier: 0 for tier in ("CRITICAL", "HIGH", "MEDIUM", "LOW")}
    uniq, counts = np.unique(risk_levels[flagged_mask], return_counts=True)
    for tier, c in zip(uniq, counts):
        if tier in severity_counts:
            severity_counts[tier] = int(c)
    agreement_counts = {
        f"{k}_of_{n_detectors}": int((n_flagged_all == k).sum())
        for k in range(1, n_detectors + 1)
    }
    total_alerts = int(flagged_mask.sum())

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
    "build_analyst_report",
    "build_clinician_summaries",
    "build_admin_dashboard",
]
