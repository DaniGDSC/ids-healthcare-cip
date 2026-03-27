"""Parameter importance analysis — identifies which hyperparameters matter.

Uses Optuna fANOVA when available, falls back to grouped mean analysis.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def compute_importance_optuna(study: Any) -> Dict[str, float]:
    """Compute parameter importance using Optuna's fANOVA.

    Args:
        study: Completed Optuna study object.

    Returns:
        Dict of {param_name: importance_score} sorted descending.
    """
    import optuna

    logger.info("── Parameter importance (Optuna fANOVA) ──")
    try:
        importances = optuna.importance.get_param_importances(study)
        for name, score in importances.items():
            logger.info("  %s: %.4f", name, score)
        return dict(importances)
    except Exception as e:
        logger.warning("  fANOVA failed: %s — falling back to grouped analysis", e)
        return {}


def compute_importance_grouped(
    trials: List[Dict[str, Any]],
    metric: str = "f1_score",
) -> Dict[str, float]:
    """Compute parameter importance via grouped mean variance analysis.

    For each parameter, groups completed trials by parameter value and
    computes the variance of group means.  Higher variance = more important.

    Args:
        trials: List of trial result dicts (must have "hyperparameters" and "metrics").
        metric: Metric to analyse.

    Returns:
        Dict of {param_name: normalised_importance} sorted descending.
    """
    logger.info("── Parameter importance (grouped variance) ──")

    completed = [t for t in trials if t.get("status") == "completed"]
    if len(completed) < 3:
        logger.warning("  Too few completed trials (%d) for importance analysis", len(completed))
        return {}

    # Collect all parameter names
    all_params: set = set()
    for t in completed:
        all_params.update(t["hyperparameters"].keys())

    variances: Dict[str, float] = {}
    for param in sorted(all_params):
        groups: Dict[Any, List[float]] = {}
        for t in completed:
            val = t["hyperparameters"].get(param)
            if val is None:
                continue
            # Round floats for grouping
            key = round(val, 6) if isinstance(val, float) else val
            groups.setdefault(key, []).append(t["metrics"][metric])

        if len(groups) < 2:
            variances[param] = 0.0
            continue

        group_means = [np.mean(scores) for scores in groups.values()]
        variances[param] = float(np.var(group_means))

    # Normalise to [0, 1]
    total_var = sum(variances.values())
    if total_var > 0:
        importances = {k: v / total_var for k, v in variances.items()}
    else:
        importances = {k: 0.0 for k in variances}

    # Sort descending
    importances = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))

    for name, score in importances.items():
        logger.info("  %s: %.4f", name, score)

    return importances


def compute_importance(
    tuner: Any,
    trials: List[Dict[str, Any]],
    metric: str = "f1_score",
) -> Dict[str, Any]:
    """Compute parameter importance using best available method.

    Uses Optuna fANOVA if a study is available, otherwise falls back
    to grouped variance analysis.

    Args:
        tuner: HyperparameterTuner instance (may have .optuna_study).
        trials: List of trial result dicts.
        metric: Target metric.

    Returns:
        Dict with method name and importance scores.
    """
    study = getattr(tuner, "optuna_study", None)

    optuna_importances: Dict[str, float] = {}
    if study is not None:
        optuna_importances = compute_importance_optuna(study)

    grouped_importances = compute_importance_grouped(trials, metric)

    # Prefer Optuna if available, include both for comparison
    if optuna_importances:
        primary = optuna_importances
        method = "optuna_fanova"
    else:
        primary = grouped_importances
        method = "grouped_variance"

    return {
        "method": method,
        "importances": primary,
        "grouped_importances": grouped_importances,
        "optuna_importances": optuna_importances,
    }
