"""Hyperparameter tuner — systematic search over detection architecture.

Supports three strategies:
  - grid:     exhaustive Cartesian product
  - random:   sampled without replacement
  - bayesian: Optuna TPE with optional Hyperband pruning
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from .config import Phase2_5Config
from .evaluator import QuickEvaluator
from .search_space import SearchSpace

logger = logging.getLogger(__name__)


class HyperparameterTuner:
    """Systematic hyperparameter search for the detection architecture.

    Args:
        config: Phase 2.5 configuration.
        evaluator: QuickEvaluator for fast train+eval.
        search_space: SearchSpace for candidate generation.
    """

    def __init__(
        self,
        config: Phase2_5Config,
        evaluator: QuickEvaluator,
        search_space: SearchSpace,
    ) -> None:
        self._config = config
        self._evaluator = evaluator
        self._search_space = search_space

    def run(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, Any]:
        """Execute the hyperparameter search.

        Args:
            X_train: Raw training features (N, F).
            y_train: Training labels.
            X_test: Raw test features.
            y_test: Test labels.

        Returns:
            Dict with all trial results, best config, and summary.
        """
        strategy = self._config.search_strategy
        metric = self._config.search_metric
        direction = self._config.search_direction

        logger.info("═══ Hyperparameter Search ═══")
        logger.info("  Strategy:  %s", strategy)
        logger.info("  Metric:    %s (%s)", metric, direction)
        logger.info("  Max trials: %d", self._config.max_trials)

        if strategy == "bayesian":
            return self._run_bayesian(X_train, y_train, X_test, y_test)

        return self._run_classic(X_train, y_train, X_test, y_test, strategy)

    # ── Classic grid/random search ─────────────────────────────────

    def _run_classic(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        strategy: str,
    ) -> Dict[str, Any]:
        """Run grid or random search."""
        metric = self._config.search_metric
        direction = self._config.search_direction

        if strategy == "grid":
            candidates = self._search_space.grid()
        else:
            candidates = self._search_space.random(self._config.max_trials)

        logger.info("  Candidates: %d", len(candidates))

        trials: List[Dict[str, Any]] = []
        for i, hp in enumerate(candidates):
            logger.info("── Trial %d/%d ──", i + 1, len(candidates))
            try:
                result = self._evaluator.evaluate_config(
                    hp, X_train, y_train, X_test, y_test
                )
                result["trial_index"] = i
                result["status"] = "completed"
                trials.append(result)

                score = result["metrics"].get(metric, 0.0)
                logger.info(
                    "  Trial %d: %s=%.4f (%.1fs)",
                    i + 1, metric, score, result["duration_seconds"],
                )
            except Exception as e:
                logger.warning("  Trial %d FAILED: %s", i + 1, e)
                trials.append({
                    "trial_index": i,
                    "hyperparameters": hp,
                    "status": "failed",
                    "error": str(e),
                })

        return self._build_result(trials, strategy, metric, direction, len(candidates))

    # ── Bayesian (Optuna TPE) with Hyperband pruning ──────────────

    def _run_bayesian(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, Any]:
        """Run Optuna TPE search with optional Hyperband pruning."""
        import optuna

        metric = self._config.search_metric
        direction = self._config.search_direction
        max_trials = self._config.max_trials

        # Suppress Optuna's verbose trial logging
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        # Build pruner
        pruner: Optional[optuna.pruners.BasePruner] = None
        if self._config.pruning_enabled:
            pruner = optuna.pruners.HyperbandPruner(
                min_resource=self._config.min_resource,
                max_resource=self._config.quick_train.epochs,
                reduction_factor=self._config.reduction_factor,
            )
            logger.info(
                "  Hyperband pruner: min_resource=%d, max_resource=%d, reduction_factor=%d",
                self._config.min_resource,
                self._config.quick_train.epochs,
                self._config.reduction_factor,
            )

        study = optuna.create_study(
            direction=direction,
            sampler=optuna.samplers.TPESampler(seed=self._config.random_state),
            pruner=pruner,
            study_name="phase2_5_tuning",
        )

        # Closure for Optuna objective
        evaluator = self._evaluator
        search_space = self._search_space
        pruning_enabled = self._config.pruning_enabled
        epochs = self._config.quick_train.epochs

        def objective(trial: optuna.Trial) -> float:
            hp = search_space.suggest_optuna(trial)

            if pruning_enabled:
                result = evaluator.evaluate_config_with_pruning(
                    hp, X_train, y_train, X_test, y_test,
                    trial=trial, metric=metric, epochs=epochs,
                )
            else:
                result = evaluator.evaluate_config(
                    hp, X_train, y_train, X_test, y_test,
                )

            trial.set_user_attr("result", result)
            return result["metrics"][metric]

        study.optimize(objective, n_trials=max_trials)

        # Collect trials into our standard format
        trials: List[Dict[str, Any]] = []
        for i, trial in enumerate(study.trials):
            if trial.state == optuna.trial.TrialState.COMPLETE:
                result = trial.user_attrs.get("result", {})
                result["trial_index"] = i
                result["status"] = "completed"
                trials.append(result)
            elif trial.state == optuna.trial.TrialState.PRUNED:
                trials.append({
                    "trial_index": i,
                    "hyperparameters": trial.params,
                    "status": "pruned",
                    "pruned_at_epoch": trial.last_step,
                })
            else:
                trials.append({
                    "trial_index": i,
                    "hyperparameters": trial.params,
                    "status": "failed",
                    "error": str(trial.user_attrs.get("error", "unknown")),
                })

        # Store the Optuna study for importance analysis
        self._optuna_study = study

        result = self._build_result(
            trials, "bayesian", metric, direction, max_trials,
        )
        result["pruned_trials"] = sum(1 for t in trials if t.get("status") == "pruned")
        return result

    @property
    def optuna_study(self) -> Any:
        """Return the Optuna study object for importance analysis."""
        return getattr(self, "_optuna_study", None)

    # ── Shared result builder ─────────────────────────────────────

    @staticmethod
    def _build_result(
        trials: List[Dict[str, Any]],
        strategy: str,
        metric: str,
        direction: str,
        total_candidates: int,
    ) -> Dict[str, Any]:
        """Build standardised tuning result dict."""
        completed = [t for t in trials if t.get("status") == "completed"]
        if not completed:
            raise RuntimeError("All trials failed — no valid configuration found")

        if direction == "maximize":
            best = max(completed, key=lambda t: t["metrics"][metric])
        else:
            best = min(completed, key=lambda t: t["metrics"][metric])

        best_score = best["metrics"][metric]
        logger.info("═══ Best: trial %d, %s=%.4f ═══", best["trial_index"], metric, best_score)

        return {
            "strategy": strategy,
            "metric": metric,
            "direction": direction,
            "total_trials": total_candidates,
            "completed_trials": len(completed),
            "failed_trials": sum(1 for t in trials if t.get("status") == "failed"),
            "trials": trials,
            "best_trial_index": best["trial_index"],
            "best_config": best["hyperparameters"],
            "best_metrics": best["metrics"],
            "best_score": best_score,
        }
