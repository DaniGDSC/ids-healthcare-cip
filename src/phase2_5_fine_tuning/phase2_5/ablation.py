"""Ablation runner — systematic component removal/replacement studies.

Evaluates the contribution of each architecture component by comparing
the full model (baseline) against variants with components removed or
replaced.  Uses SMOTE data for training and imbalanced test for evaluation.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np

from .config import Phase2_5Config
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
        X_train_smote: np.ndarray,
        y_train_smote: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, Any]:
        """Execute all ablation variants and compare against baseline.

        Args:
            base_hp: Baseline hyperparameters.
            X_train_smote: SMOTE-balanced training features.
            y_train_smote: SMOTE-balanced labels.
            X_test: Imbalanced test features.
            y_test: Imbalanced test labels.

        Returns:
            Dict with baseline results, variant results, and comparison.
        """
        logger.info("═══ Ablation Study ═══")
        variants = self._config.ablation_variants
        logger.info("  Variants: %d", len(variants))

        # Baseline (full model, no modifications)
        logger.info("── Baseline: full model ──")
        baseline_result = self._evaluator.evaluate_ablation_variant(
            variant_name="baseline (full)",
            base_hp=base_hp,
            X_train_smote=X_train_smote,
            y_train_smote=y_train_smote,
            X_test=X_test,
            y_test=y_test,
        )
        baseline_result["status"] = "completed"
        logger.info(
            "  Baseline: attack_f1=%.4f, AUC=%.4f",
            baseline_result["metrics"].get("attack_f1", 0),
            baseline_result["metrics"].get("auc_roc", 0),
        )

        # Evaluate each variant
        variant_results: List[Dict[str, Any]] = []
        for i, variant in enumerate(variants):
            logger.info("── Variant %d/%d: %s ──", i + 1, len(variants), variant.name)
            try:
                result = self._evaluator.evaluate_ablation_variant(
                    variant_name=variant.name,
                    base_hp=base_hp,
                    X_train_smote=X_train_smote,
                    y_train_smote=y_train_smote,
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
                    "  %s: attack_f1=%.4f, AUC=%.4f (%.1fs)",
                    variant.name,
                    result["metrics"].get("attack_f1", 0),
                    result["metrics"].get("auc_roc", 0),
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
        """Build comparison table with attack_f1 deltas."""
        bm = baseline["metrics"]
        rows: List[Dict[str, Any]] = []

        rows.append({
            "variant": "baseline (full)",
            "attack_f1": bm.get("attack_f1", 0),
            "auc_roc": bm.get("auc_roc", 0),
            "accuracy": bm.get("accuracy", 0),
            "total_params": baseline.get("total_params", 0),
            "delta_f1": 0.0,
            "delta_auc": 0.0,
        })

        for v in variants:
            if v.get("status") != "completed":
                rows.append({
                    "variant": v["variant"],
                    "attack_f1": None, "auc_roc": None, "accuracy": None,
                    "total_params": None, "delta_f1": None, "delta_auc": None,
                    "status": "failed",
                })
                continue

            vm = v["metrics"]
            rows.append({
                "variant": v["variant"],
                "attack_f1": vm.get("attack_f1", 0),
                "auc_roc": vm.get("auc_roc", 0),
                "accuracy": vm.get("accuracy", 0),
                "total_params": v.get("total_params", 0),
                "delta_f1": round(vm.get("attack_f1", 0) - bm.get("attack_f1", 0), 4),
                "delta_auc": round(vm.get("auc_roc", 0) - bm.get("auc_roc", 0), 4),
            })

        return rows
