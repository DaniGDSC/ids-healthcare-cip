"""ContextEnricher — attach per-sample SHAP context and explanations."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np

from .explanation_generator import ExplanationGenerator

logger = logging.getLogger(__name__)


class ContextEnricher:
    """Enrich filtered samples with SHAP context and explanations.

    Args:
        explanation_generator: ExplanationGenerator for producing text.
    """

    def __init__(
        self,
        explanation_generator: ExplanationGenerator,
    ) -> None:
        self._generator = explanation_generator

    def enrich(
        self,
        filtered_samples: List[Dict[str, Any]],
        shap_values: np.ndarray,
        feature_names: List[str],
    ) -> List[Dict[str, Any]]:
        """Attach SHAP context and explanation to each filtered sample.

        Args:
            filtered_samples: Non-NORMAL sample dicts from risk_report.
            shap_values: SHAP values, shape (N, T, F).
            feature_names: List of feature names.

        Returns:
            Enriched sample dicts with top_features and explanation.
        """
        logger.info("── Context enrichment ──")
        enriched = []

        for i, sample in enumerate(filtered_samples):
            if i >= len(shap_values):
                break

            top3 = self.get_top3_features(shap_values[i], feature_names)
            explanation = self._generator.generate(
                sample["risk_level"],
                sample["sample_index"],
                top3,
            )

            enriched.append(
                {
                    "sample_index": sample["sample_index"],
                    "risk_level": sample["risk_level"],
                    "anomaly_score": sample["anomaly_score"],
                    "threshold": sample["threshold"],
                    "timestamp": f"T={sample['sample_index']}",
                    "top_features": top3,
                    "explanation": explanation,
                }
            )

        logger.info("  Enriched %d samples with explanations", len(enriched))
        return enriched

    @staticmethod
    def get_top3_features(
        sample_shap: np.ndarray,
        feature_names: List[str],
    ) -> List[Dict[str, Any]]:
        """Extract top 3 contributing features for a single sample.

        Args:
            sample_shap: SHAP values for one sample, shape (T, F).
            feature_names: Feature names.

        Returns:
            List of dicts with feature, shap_value, contribution_pct.
        """
        per_feature = np.mean(np.abs(sample_shap), axis=0)
        total = per_feature.sum()
        if total < 1e-12:
            total = 1.0

        ranked_idx = np.argsort(per_feature)[::-1][:3]
        top3 = []
        for idx in ranked_idx:
            top3.append(
                {
                    "feature": feature_names[idx],
                    "shap_value": round(float(per_feature[idx]), 6),
                    "contribution_pct": round(float(per_feature[idx] / total * 100), 2),
                }
            )
        return top3

    def get_config(self) -> Dict[str, Any]:
        """Return enricher configuration."""
        return {"generator": self._generator.get_config()}
