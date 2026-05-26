"""Batch simulation + latency profiling for the online explainer.

Renders per-alert explanations for every XGBoost-flagged sample using
batched TreeSHAP / DAE calls, then computes p50/p95/p99 latency stats.
"""

from __future__ import annotations

import logging
import time

import numpy as np

from .compute import _normalise_shap_output, _top_features_dae, _top_features_shap
from .nlg import build_shap_context, clinician_nlg
from .online_explainer import AlertExplainer

logger = logging.getLogger(__name__)


def run_batch_simulation(
    explainer: AlertExplainer,
    X_test: np.ndarray,
    y_pred_xgb: np.ndarray,
    feat_names: list,
) -> tuple[list, list]:
    """Run per-alert explanations for all XGBoost-flagged samples.

    Batch TreeSHAP: computes SHAP values for all flagged samples in one
    call per model. Python→C boundary crossed once instead of k times.
    """
    if tuple(feat_names) != explainer.feat_names:
        raise ValueError(
            "run_batch_simulation: feat_names mismatch with explainer's "
            "constructor feat_names. Construct a new AlertExplainer if "
            "the feature schema changed."
        )

    alert_idx = np.where(y_pred_xgb == 1)[0]
    logger.info(
        "Simulating %d per-alert explanations (batch SHAP)...", len(alert_idx),
    )

    if len(alert_idx) == 0:
        return [], []

    X_alerts = X_test[alert_idx]
    feats = list(feat_names)

    # ── Batch Track A predictions ──
    t_pred = time.perf_counter()
    batch_votes: dict[str, np.ndarray] = {}
    for name, clf in explainer.classifiers.items():
        batch_votes[name] = clf.predict_proba(X_alerts)[:, 1]
    pred_ms = round((time.perf_counter() - t_pred) * 1000, 3)

    # ── Batch SHAP ──
    t_shap = time.perf_counter()
    batch_shap: dict[str, np.ndarray] = {}
    for name, shap_explainer in explainer.explainers.items():
        sv = shap_explainer.shap_values(X_alerts)
        sv = _normalise_shap_output(sv)
        batch_shap[name] = sv
    shap_ms = round((time.perf_counter() - t_shap) * 1000, 3)

    # ── Batch DAE ──
    t_dae = time.perf_counter()
    from detection_engine import DetectionEngine
    X_aug = DetectionEngine().build_augmented(X_alerts)
    dae_errors, dae_per_feature = explainer.dae.reconstruction_error_decomposed(X_aug)
    dae_ms = round((time.perf_counter() - t_dae) * 1000, 3)

    logger.info(
        "  Batch: predict=%.1fms, shap=%.1fms, dae=%.1fms for %d alerts",
        pred_ms, shap_ms, dae_ms, len(alert_idx),
    )

    all_timings = []
    sample_explanations = []

    for i, idx in enumerate(alert_idx):
        votes = {
            name: {
                "prediction": int(batch_votes[name][i] >= explainer.thresholds[name]),
                "confidence": round(float(batch_votes[name][i]), 4),
            }
            for name in explainer.classifiers
        }
        dae_error = float(dae_errors[i])
        votes["dae"] = {
            "prediction": int(dae_error > explainer.dae.threshold),
            "reconstruction_error": round(dae_error, 8),
        }

        n_flagged = sum(1 for v in votes.values() if v["prediction"] == 1)
        severity = explainer._severity(n_flagged)

        timings = {
            "predict_ms":  round(pred_ms / len(alert_idx), 3),
            "treeshap_ms": round(shap_ms / len(alert_idx), 3),
            "dae_ms":      round(dae_ms / len(alert_idx), 3),
            "total_ms":    round((pred_ms + shap_ms + dae_ms) / len(alert_idx), 3),
        }
        all_timings.append(timings)

        if len(sample_explanations) < 20:
            xgb_sv = batch_shap.get("xgboost")
            top_shap = (
                _top_features_shap(xgb_sv[i], feats, k=3)
                if xgb_sv is not None else []
            )
            top_dae = _top_features_dae(dae_per_feature[i], feats, k=3)
            nlg = clinician_nlg(severity, top_shap)
            shap_context = build_shap_context(top_shap)

            sample_explanations.append({
                "sample_index": int(idx),
                "votes": votes,
                "severity": severity,
                "n_flagged": n_flagged,
                "top_shap_features": top_shap,
                "top_dae_features": top_dae,
                "clinician_summary": nlg,
                "shap_context": shap_context,
                "timings_ms": timings,
            })

        if (i + 1) % 100 == 0:
            logger.info("  Assembled %d/%d alert records", i + 1, len(alert_idx))

    return all_timings, sample_explanations


def compute_latency_stats(all_timings: list) -> dict:
    """Compute p50/p95/p99 for each timing component."""
    if not all_timings:
        return {}

    components = list(all_timings[0].keys())
    stats: dict[str, dict] = {}

    for comp in components:
        values = [t[comp] for t in all_timings if comp in t]
        if not values:
            continue
        arr = np.array(values)
        p50, p95, p99 = np.percentile(arr, [50, 95, 99])
        stats[comp] = {
            "n_samples": len(arr),
            "mean": round(float(arr.mean()), 3),
            "p50":  round(float(p50), 3),
            "p95":  round(float(p95), 3),
            "p99":  round(float(p99), 3),
            "min":  round(float(arr.min()), 3),
            "max":  round(float(arr.max()), 3),
        }

    return stats


__all__ = ["run_batch_simulation", "compute_latency_stats"]
