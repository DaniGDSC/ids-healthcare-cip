"""AlertFilter — filter non-NORMAL risk samples for SHAP explanation."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class AlertFilter:
    """Filter non-NORMAL risk samples with stratified sampling.

    Args:
        max_samples: Maximum samples to explain (stratified sampling).
    """

    RISK_LEVELS_NON_NORMAL: Tuple[str, ...] = (
        "LOW",
        "MEDIUM",
        "HIGH",
        "CRITICAL",
    )

    def __init__(self, max_samples: int = 200) -> None:
        self._max_samples = max_samples

    def filter(
        self,
        sample_assessments: List[Dict[str, Any]],
        rng: np.random.Generator,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
        """Filter samples where risk_level != NORMAL.

        Args:
            sample_assessments: Per-sample risk dicts from risk_report.json.
            rng: Random number generator for stratified sampling.

        Returns:
            Tuple of (filtered_samples, level_counts).
        """
        non_normal = [s for s in sample_assessments if s["risk_level"] != "NORMAL"]

        level_counts: Dict[str, int] = {}
        for level in self.RISK_LEVELS_NON_NORMAL:
            count = sum(1 for s in non_normal if s["risk_level"] == level)
            level_counts[level] = count

        if len(non_normal) > self._max_samples:
            sampled: List[Dict[str, Any]] = []
            for level in self.RISK_LEVELS_NON_NORMAL:
                level_samples = [s for s in non_normal if s["risk_level"] == level]
                n_take = max(
                    1,
                    int(self._max_samples * len(level_samples) / len(non_normal)),
                )
                n_take = min(n_take, len(level_samples))
                indices = rng.choice(len(level_samples), n_take, replace=False)
                sampled.extend(level_samples[i] for i in sorted(indices))
            non_normal = sampled

        counts_str = ", ".join(f"{k}={v}" for k, v in level_counts.items())
        logger.info(
            "  Filtered %d samples for explanation (%s)",
            len(non_normal),
            counts_str,
        )
        return non_normal, level_counts

    def get_config(self) -> Dict[str, Any]:
        """Return filter configuration."""
        return {"max_samples": self._max_samples}
