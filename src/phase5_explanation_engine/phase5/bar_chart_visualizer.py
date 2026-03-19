"""BarChartVisualizer — render horizontal bar chart of feature importance."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, FrozenSet, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch

from .base import BaseVisualizer


class BarChartVisualizer(BaseVisualizer):
    """Render horizontal bar chart of top features by mean |SHAP|.

    Args:
        top_k: Number of top features to display.
        biometric_columns: Set of biometric column names for color coding.
        dpi: Chart resolution.
        fig_width: Figure width in inches.
        fig_height: Figure height in inches.
    """

    def __init__(
        self,
        top_k: int = 10,
        biometric_columns: Optional[FrozenSet[str]] = None,
        dpi: int = 150,
        fig_width: float = 10.0,
        fig_height: float = 6.0,
    ) -> None:
        self._top_k = top_k
        self._biometric_columns = biometric_columns or frozenset()
        self._dpi = dpi
        self._fig_width = fig_width
        self._fig_height = fig_height

    def plot(self, data: Dict[str, Any], output_path: Path) -> None:
        """Render feature importance bar chart.

        Expected data keys:
            importance_df: DataFrame with feature, mean_abs_shap columns.
        """
        importance_df = data["importance_df"]
        top = importance_df.head(self._top_k).copy()
        top = top.sort_values("mean_abs_shap", ascending=True)

        colors = ["#1f77b4" if f in self._biometric_columns else "#d62728" for f in top["feature"]]

        fig, ax = plt.subplots(figsize=(self._fig_width, self._fig_height))
        ax.barh(top["feature"], top["mean_abs_shap"], color=colors)
        ax.set_xlabel("Mean |SHAP Value|")
        ax.set_title(f"Top {self._top_k} Feature Importance (Global)")

        legend_items = [
            Patch(facecolor="#d62728", label="Network features"),
            Patch(facecolor="#1f77b4", label="Biometric features"),
        ]
        ax.legend(handles=legend_items, loc="lower right", fontsize=8)

        plt.tight_layout()
        fig.savefig(output_path, dpi=self._dpi, bbox_inches="tight")
        plt.close(fig)

    def get_config(self) -> Dict[str, Any]:
        """Return bar chart visualizer configuration."""
        return {
            "type": "bar_chart",
            "top_k": self._top_k,
            "biometric_columns": sorted(self._biometric_columns),
            "dpi": self._dpi,
            "fig_width": self._fig_width,
            "fig_height": self._fig_height,
        }
