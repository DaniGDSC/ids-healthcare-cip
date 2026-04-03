"""WaterfallVisualizer — render waterfall chart of cumulative SHAP contributions."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .base import BaseVisualizer


class WaterfallVisualizer(BaseVisualizer):
    """Render waterfall chart of cumulative SHAP feature contributions.

    Args:
        top_k: Number of top features to display.
        dpi: Chart resolution.
        fig_width: Figure width in inches.
        fig_height: Figure height in inches.
    """

    def __init__(
        self,
        top_k: int = 10,
        dpi: int = 150,
        fig_width: float = 10.0,
        fig_height: float = 6.0,
    ) -> None:
        self._top_k = top_k
        self._dpi = dpi
        self._fig_width = fig_width
        self._fig_height = fig_height

    def plot(self, data: Dict[str, Any], output_path: Path) -> None:
        """Render waterfall chart.

        Expected data keys:
            sample: Enriched sample dict with sample_index, risk_level.
            shap_vals: SHAP values for this sample, shape (T, F).
            feature_names: List of feature names.
            baseline_threshold: Baseline prediction value.
        """
        sample = data["sample"]
        shap_vals = data["shap_vals"]
        feature_names = data["feature_names"]
        baseline_threshold = data["baseline_threshold"]

        per_feature = np.mean(shap_vals, axis=0)
        ranked_idx = np.argsort(np.abs(per_feature))[::-1][: self._top_k]

        names = [feature_names[i] for i in ranked_idx]
        values = [per_feature[i] for i in ranked_idx]

        fig, ax = plt.subplots(figsize=(self._fig_width, self._fig_height))

        cumulative = baseline_threshold
        starts = []
        for v in values:
            starts.append(cumulative)
            cumulative += v

        colors = ["#d62728" if v > 0 else "#1f77b4" for v in values]
        ax.barh(range(len(names)), values, left=starts, color=colors)

        ax.axvline(
            x=baseline_threshold,
            color="gray",
            linestyle="--",
            linewidth=1,
            label=f"Baseline ({baseline_threshold:.4f})",
        )
        ax.axvline(
            x=cumulative,
            color="black",
            linestyle="-",
            linewidth=1.5,
            label=f"Final ({cumulative:.4f})",
        )

        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        ax.invert_yaxis()
        ax.set_xlabel("Model Output")
        ax.set_title(f"Waterfall — Sample {sample['sample_index']} " f"({sample['risk_level']})")
        ax.legend(loc="lower right", fontsize=8)
        plt.tight_layout()
        fig.savefig(output_path, dpi=self._dpi, bbox_inches="tight")
        plt.close(fig)

    def get_config(self) -> Dict[str, Any]:
        """Return waterfall visualizer configuration."""
        return {
            "type": "waterfall",
            "top_k": self._top_k,
            "dpi": self._dpi,
            "fig_width": self._fig_width,
            "fig_height": self._fig_height,
        }
