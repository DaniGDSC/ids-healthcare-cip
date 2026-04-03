"""FeatureImportanceRanker — compute global feature importance from SHAP."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FeatureImportanceRanker:
    """Compute global feature importance from SHAP values.

    Aggregates 3D SHAP values (N, T, F) over timesteps then over samples.

    Args:
        top_k: Number of top features to return.
    """

    def __init__(self, top_k: int = 10) -> None:
        self._top_k = top_k

    def rank(
        self,
        shap_values: np.ndarray,
        feature_names: List[str],
    ) -> Tuple[pd.DataFrame, List[Tuple[str, float]]]:
        """Compute global feature importance.

        Args:
            shap_values: SHAP values, shape (N, T, F).
            feature_names: List of feature names.

        Returns:
            Tuple of (full importance DataFrame, top_k list of (name, score)).
        """
        per_sample = np.mean(np.abs(shap_values), axis=1)  # (N, F)
        global_importance = np.mean(per_sample, axis=0)  # (F,)

        df = pd.DataFrame(
            {"feature": feature_names, "mean_abs_shap": global_importance}
        ).sort_values("mean_abs_shap", ascending=False)
        df["rank"] = range(1, len(df) + 1)

        top_features = [
            (row["feature"], row["mean_abs_shap"]) for _, row in df.head(self._top_k).iterrows()
        ]

        logger.info("  Top %d features:", self._top_k)
        for name, score in top_features:
            logger.info("    %s: %.6f", name, score)

        return df, top_features

    def get_config(self) -> Dict[str, Any]:
        """Return ranker configuration."""
        return {"top_k": self._top_k}
