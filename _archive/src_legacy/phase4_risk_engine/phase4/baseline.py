"""Baseline computation — Median + k*MAD from Normal-only attention scores.

Produces an IMMUTABLE baseline configuration used as the reference
point for dynamic thresholding and concept drift detection.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict

import numpy as np
import pandas as pd

from .base import BaseDetector

logger = logging.getLogger(__name__)


class BaselineComputer(BaseDetector):
    """Compute Median + k*MAD baseline from Normal-only attention scores.

    Args:
        mad_multiplier: k multiplier for MAD (default 3.0).
    """

    def __init__(self, mad_multiplier: float = 3.0) -> None:
        self._mad_multiplier = mad_multiplier

    def compute(self, attn_df: pd.DataFrame) -> Dict[str, Any]:
        """Compute baseline threshold from Normal-only training attention.

        Uses Median + k*MAD where MAD = Median(|Xi - Median(X)|).

        Args:
            attn_df: Attention output DataFrame with Label and split columns.

        Returns:
            Baseline config dict with median, mad, threshold, n_normal.
        """
        logger.info("── Baseline computation ──")

        # Filter Normal samples from training split only
        train_normal = attn_df[(attn_df["Label"] == 0) & (attn_df["split"] == "train")]
        attn_cols = [c for c in attn_df.columns if c.startswith("attn_")]

        logger.info("  Normal training samples: %d", len(train_normal))

        # Per-sample attention magnitude (L2 norm across 128 dims)
        attn_values = train_normal[attn_cols].values.astype(np.float64)
        sample_magnitudes = np.linalg.norm(attn_values, axis=1)

        median = float(np.median(sample_magnitudes))
        mad = float(np.median(np.abs(sample_magnitudes - median)))
        baseline_threshold = median + self._mad_multiplier * mad

        logger.info(
            "  Baseline threshold: %.6f, MAD: %.6f",
            baseline_threshold,
            mad,
        )

        return {
            "median": median,
            "mad": mad,
            "baseline_threshold": baseline_threshold,
            "mad_multiplier": self._mad_multiplier,
            "n_normal_samples": len(train_normal),
            "n_attention_dims": len(attn_cols),
            "computed_at": datetime.now(timezone.utc).isoformat(),
        }

    def get_config(self) -> Dict[str, Any]:
        """Return baseline computer configuration."""
        return {"mad_multiplier": self._mad_multiplier}
