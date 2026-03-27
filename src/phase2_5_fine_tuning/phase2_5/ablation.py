"""Ablation runner — systematic component removal/replacement studies.

Evaluates the contribution of each architecture component by comparing
the full model (baseline) against variants with components removed or
replaced.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np

from .config import AblationVariantConfig, Phase2_5Config
from .evaluator import QuickEvaluator

logger = logging.getLogger(__name__)


class AblationRunner:
    """Run ablation studies on the detection architecture.

    Args:
        config: Phase 2.5 configuration.
        evaluator: QuickEvaluator for fast train+eval.
    """

    def __init__(
        self,
        config: Phase2_5Config,
        evaluator: QuickEvaluator,
    ) -> None:
        self._config = config
        self._evaluator = evaluator

    def run(
        self,
        base_hp: Dict[str, Any],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, Any]:
        """Execute all ablation variants and compare against baseline.

        Args:
            base_hp: Baseline hyperparameters (full model).
            X_train: Raw training features (N, F).
            y_train: Training labels.
            X_test: Raw test features.
            y_test: Test labels.

        Returns:
            Dict with baseline results, variant results, and comparison.
        """
        logger.info("═══ Ablation Study ═══")
        variants = self._config.ablation_variants
        logger.info("  Baseline: %s", self._config.ablation_baseline)
        logger.info("  Variants: %d", len(variants))

        # Evaluate baseline (full model)
        logger.info("── Baseline: full model ──")
        baseline_result = self._evaluator.evaluate_config(
            base_hp, X_train, y_train, X_test, y_test
        )
        baseline_result["variant"] = "baseline (full)"
        logger.info(
            "  Baseline: F1=%.4f, AUC=%.4f",
            baseline_result["metrics"]["f1_score"],
            baseline_result["metrics"]["auc_roc"],
        )

        # Evaluate each variant
        variant_results: List[Dict[str, Any]] = []
        for i, variant in enumerate(variants):
            logger.info("── Variant %d/%d: %s ──", i + 1, len(variants), variant.name)
            try:
                result = self._evaluator.evaluate_ablation_variant(
                    variant_name=variant.name,
                    base_hp=base_hp,
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    remove=variant.remove,
                    replace=variant.replace,
                    override=variant.override,
                )
                result["description"] = variant.description
                result["status"] = "completed"
                variant_results.append(result)

                logger.info(
                    "  %s: F1=%.4f, AUC=%.4f (%.1fs)",
                    variant.name,
                    result["metrics"]["f1_score"],
                    result["metrics"]["auc_roc"],
                    result["duration_seconds"],
                )
            except Exception as e:
                logger.warning("  Variant %s FAILED: %s", variant.name, e)
                variant_results.append({
                    "variant": variant.name,
                    "description": variant.description,
                    "status": "failed",
                    "error": str(e),
                })

        # Build comparison table
        comparison = self._build_comparison(baseline_result, variant_results)

        return {
            "baseline": baseline_result,
            "variants": variant_results,
            "comparison": comparison,
        }

    @staticmethod
    def _build_comparison(
        baseline: Dict[str, Any],
        variants: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Build a comparison table of metric deltas against baseline."""
        base_metrics = baseline["metrics"]
        rows: List[Dict[str, Any]] = []

        rows.append({
            "variant": "baseline (full)",
            "f1_score": base_metrics["f1_score"],
            "auc_roc": base_metrics["auc_roc"],
            "accuracy": base_metrics["accuracy"],
            "total_params": baseline.get("total_params", 0),
            "delta_f1": 0.0,
            "delta_auc": 0.0,
        })

        for v in variants:
            if v.get("status") != "completed":
                rows.append({
                    "variant": v["variant"],
                    "f1_score": None,
                    "auc_roc": None,
                    "accuracy": None,
                    "total_params": None,
                    "delta_f1": None,
                    "delta_auc": None,
                    "status": "failed",
                })
                continue

            vm = v["metrics"]
            rows.append({
                "variant": v["variant"],
                "f1_score": vm["f1_score"],
                "auc_roc": vm["auc_roc"],
                "accuracy": vm["accuracy"],
                "total_params": v.get("total_params", 0),
                "delta_f1": round(vm["f1_score"] - base_metrics["f1_score"], 4),
                "delta_auc": round(vm["auc_roc"] - base_metrics["auc_roc"], 4),
            })

        return rows
