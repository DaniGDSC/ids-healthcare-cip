"""Explanation-faithfulness validation suite.

Three independent checks:
  - ``validate_consistency`` — SHAP rank vs sklearn ``feature_importances_``
  - ``validate_perturbation``  — F1 drop when top-N features are masked
  - ``validate_cross_model``   — Spearman rho + top-5 overlap across models
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import f1_score

from .config import SHAP_MODELS, TRACK_A_MODELS
from .io import OUTPUT_DIR, write_json_strict

logger = logging.getLogger(__name__)


def validate_consistency(
    all_shap: dict,
    feat_names: list,
    *,
    project_root: Path,
    output_dir: Path | None = None,
) -> dict:
    """Compare SHAP feature rankings with native ``feature_importances_``."""
    logger.info("Running consistency check (SHAP vs native importances)...")
    results = {}
    out_dir = output_dir or OUTPUT_DIR

    from common import loads_signed

    for name in SHAP_MODELS:
        cfg = TRACK_A_MODELS[name]
        obj = loads_signed(project_root / cfg["pipeline"])
        clf = obj.named_steps["classifier"] if hasattr(obj, "named_steps") else obj

        if not hasattr(clf, "feature_importances_"):
            continue

        native_imp = clf.feature_importances_
        native_ranked = [feat_names[i] for i in np.argsort(native_imp)[::-1]]

        shap_mean_abs = np.mean(np.abs(all_shap[name]), axis=0)
        shap_ranked = [feat_names[i] for i in np.argsort(shap_mean_abs)[::-1]]

        native_top5 = set(native_ranked[:5])
        shap_top5 = set(shap_ranked[:5])
        overlap = native_top5 & shap_top5

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
        logger.info(
            "  %s: top-5 overlap=%d/5, Spearman rho=%.4f (p=%.4f)",
            name, len(overlap), rho, p_val,
        )

    write_json_strict(out_dir / "validation_consistency.json", results)
    logger.info("  Saved: validation_consistency.json")
    return results


def validate_perturbation(
    all_shap: dict,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feat_names: list,
    *,
    top_n_features: int = 5,
    output_dir: Path | None = None,
) -> dict:
    """Mask top-N SHAP features and measure F1 drop.

    Faithful SHAP explanations should produce a noticeable F1 drop when
    the most important features are masked to their column means.
    """
    logger.info("Running perturbation test (mask top-%d features)...", top_n_features)
    results = {}
    out_dir = output_dir or OUTPUT_DIR

    from common.model_registry import get_track_a_classifiers, get_track_a_thresholds
    classifiers = get_track_a_classifiers()
    thresholds = get_track_a_thresholds()

    for name in SHAP_MODELS:
        clf = classifiers[name]
        y_proba_base = clf.predict_proba(X_test)[:, 1]
        threshold = thresholds[name]
        y_pred_base = (y_proba_base >= threshold).astype(int)
        f1_base = f1_score(y_test, y_pred_base, pos_label=1)

        shap_mean = np.mean(np.abs(all_shap[name]), axis=0)
        top_feat_idx = np.argsort(shap_mean)[-top_n_features:]

        X_masked = X_test.copy()
        X_masked[:, top_feat_idx] = X_test[:, top_feat_idx].mean(axis=0)

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
        logger.info(
            "  %s: F1 %.4f → %.4f (drop=%.1f%%) %s",
            name, f1_base, f1_masked, drop_pct,
            "FAITHFUL" if drop_pct > 5.0 else "WEAK",
        )

    write_json_strict(out_dir / "validation_perturbation.json", results)
    logger.info("  Saved: validation_perturbation.json")
    return results


def validate_cross_model(
    global_importances: dict,
    *,
    output_dir: Path | None = None,
) -> dict:
    """Compare SHAP rankings across models."""
    logger.info("Running cross-model ranking comparison...")
    out_dir = output_dir or OUTPUT_DIR

    model_names = list(global_importances.keys())
    if not model_names:
        return {"pairwise_comparisons": {}, "consensus_top5_all_models": []}

    all_features = [entry["feature"] for entry in global_importances[model_names[0]]]
    feat_to_col = {f: i for i, f in enumerate(all_features)}
    n_models = len(model_names)
    n_feats = len(all_features)

    rank_matrix = np.zeros((n_models, n_feats), dtype=np.int32)
    for mi, name in enumerate(model_names):
        for entry in global_importances[name]:
            col = feat_to_col.get(entry["feature"])
            if col is not None:
                rank_matrix[mi, col] = entry["rank"]

    top5_sets = [
        set(all_features[j] for j in range(n_feats) if rank_matrix[mi, j] <= 5)
        for mi in range(n_models)
    ]

    comparisons = {}
    for i, m1 in enumerate(model_names):
        for j, m2 in enumerate(model_names[i + 1:], start=i + 1):
            rho, p_val = spearmanr(rank_matrix[i], rank_matrix[j])
            overlap = top5_sets[i] & top5_sets[j]
            comparisons[f"{m1}_vs_{m2}"] = {
                "spearman_rho": round(float(rho), 4),
                "spearman_p_value": round(float(p_val), 6),
                "top5_overlap": sorted(overlap),
                "top5_overlap_count": len(overlap),
            }
            logger.info(
                "  %s vs %s: rho=%.4f, top-5 overlap=%d/5",
                m1, m2, rho, len(overlap),
            )

    consensus = sorted(set.intersection(*top5_sets)) if top5_sets else []
    result = {
        "pairwise_comparisons": comparisons,
        "consensus_top5_all_models": consensus,
    }

    write_json_strict(out_dir / "validation_cross_model.json", result)
    logger.info("  Consensus features (top-5 in all models): %s", consensus)
    return result


__all__ = [
    "validate_consistency",
    "validate_perturbation",
    "validate_cross_model",
]
