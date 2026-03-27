"""Multi-seed validation — retrain top-K configs with multiple seeds.

Provides confidence intervals for the best configurations found during
hyperparameter search.  Essential for publication-grade results.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np

from .config import MultiSeedConfig
from .evaluator import QuickEvaluator

logger = logging.getLogger(__name__)


class MultiSeedValidator:
    """Retrain top-K configurations with multiple seeds for confidence intervals.

    Args:
        config: Multi-seed validation configuration.
        evaluator: QuickEvaluator for train+eval.
    """

    def __init__(
        self,
        config: MultiSeedConfig,
        evaluator: QuickEvaluator,
    ) -> None:
        self._config = config
        self._evaluator = evaluator

    def validate(
        self,
        tuning_results: Dict[str, Any],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, Any]:
        """Retrain top-K configs with multiple seeds and compute statistics.

        Args:
            tuning_results: Results from hyperparameter search.
            X_train: Raw training features.
            y_train: Training labels.
            X_test: Raw test features.
            y_test: Test labels.

        Returns:
            Dict with per-config seed results, mean, std, and rankings.
        """
        if not self._config.enabled:
            logger.info("  Multi-seed validation disabled — skipping")
            return {"enabled": False, "configs": []}

        logger.info("═══ Multi-Seed Validation ═══")
        logger.info("  Top-K: %d", self._config.top_k)
        logger.info("  Seeds: %s", self._config.seeds)
        logger.info("  Full epochs: %d", self._config.full_epochs)

        metric = tuning_results.get("metric", "f1_score")
        direction = tuning_results.get("direction", "maximize")
        trials = tuning_results.get("trials", [])

        # Get top-K completed trials
        completed = [t for t in trials if t.get("status") == "completed"]
        if direction == "maximize":
            top_trials = sorted(
                completed,
                key=lambda t: t["metrics"][metric],
                reverse=True,
            )[:self._config.top_k]
        else:
            top_trials = sorted(
                completed,
                key=lambda t: t["metrics"][metric],
            )[:self._config.top_k]

        configs_results: List[Dict[str, Any]] = []

        for rank, trial in enumerate(top_trials):
            hp = trial["hyperparameters"]
            logger.info("── Config %d/%d (original %s=%.4f) ──",
                        rank + 1, len(top_trials), metric, trial["metrics"][metric])

            seed_results: List[Dict[str, Any]] = []
            for seed in self._config.seeds:
                logger.info("  Seed %d ...", seed)
                try:
                    result = self._evaluator.evaluate_config(
                        hp, X_train, y_train, X_test, y_test,
                        seed_override=seed,
                        epochs_override=self._config.full_epochs,
                    )
                    seed_results.append({
                        "seed": seed,
                        "metrics": result["metrics"],
                        "epochs_run": result["epochs_run"],
                        "duration_seconds": result["duration_seconds"],
                        "status": "completed",
                    })
                    logger.info("    %s=%.4f", metric, result["metrics"][metric])
                except Exception as e:
                    logger.warning("    Seed %d FAILED: %s", seed, e)
                    seed_results.append({
                        "seed": seed,
                        "status": "failed",
                        "error": str(e),
                    })

            # Compute statistics
            completed_seeds = [s for s in seed_results if s["status"] == "completed"]
            stats = self._compute_stats(completed_seeds, metric)

            configs_results.append({
                "rank": rank + 1,
                "hyperparameters": hp,
                "original_score": trial["metrics"][metric],
                "seed_results": seed_results,
                "statistics": stats,
            })

            if stats:
                logger.info(
                    "  %s: %.4f +/- %.4f (n=%d)",
                    metric, stats["mean"], stats["std"], stats["n_seeds"],
                )

        return {
            "enabled": True,
            "metric": metric,
            "top_k": self._config.top_k,
            "seeds": self._config.seeds,
            "full_epochs": self._config.full_epochs,
            "configs": configs_results,
        }

    @staticmethod
    def _compute_stats(
        seed_results: List[Dict[str, Any]],
        metric: str,
    ) -> Dict[str, Any]:
        """Compute mean, std, min, max across seeds for all metrics."""
        if not seed_results:
            return {}

        all_metrics: Dict[str, List[float]] = {}
        for s in seed_results:
            for k, v in s["metrics"].items():
                all_metrics.setdefault(k, []).append(v)

        stats: Dict[str, Any] = {"n_seeds": len(seed_results)}
        for k, values in all_metrics.items():
            arr = np.array(values)
            stats[f"{k}_mean"] = round(float(arr.mean()), 4)
            stats[f"{k}_std"] = round(float(arr.std()), 4)
            stats[f"{k}_min"] = round(float(arr.min()), 4)
            stats[f"{k}_max"] = round(float(arr.max()), 4)

        # Convenience aliases for the target metric
        target_vals = all_metrics.get(metric, [])
        if target_vals:
            stats["mean"] = round(float(np.mean(target_vals)), 4)
            stats["std"] = round(float(np.std(target_vals)), 4)

        return stats
