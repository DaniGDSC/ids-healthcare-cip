"""LineGraphVisualizer — render anomaly score timeline with threshold crossing."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .base import BaseVisualizer


class LineGraphVisualizer(BaseVisualizer):
    """Render anomaly score timeline with threshold crossing markers.

    Args:
        dpi: Chart resolution.
        fig_width: Figure width in inches.
        fig_height: Figure height in inches.
    """

    def __init__(
        self,
        dpi: int = 150,
        fig_width: float = 10.0,
        fig_height: float = 6.0,
    ) -> None:
        self._dpi = dpi
        self._fig_width = fig_width
        self._fig_height = fig_height

    def plot(self, data: Dict[str, Any], output_path: Path) -> None:
        """Render anomaly timeline chart.

        Expected data keys:
            anomaly_scores: List[float] of consecutive anomaly scores.
            baseline_threshold: Threshold for marking crossing point.
            incident_id: Incident identifier for title.
        """
        anomaly_scores = data["anomaly_scores"]
        baseline_threshold = data["baseline_threshold"]
        incident_id = data["incident_id"]

        t = list(range(1, len(anomaly_scores) + 1))

        fig, ax = plt.subplots(figsize=(self._fig_width, self._fig_height * 0.7))
        ax.plot(t, anomaly_scores, "b-o", markersize=4, label="Anomaly Score")
        ax.axhline(
            y=baseline_threshold,
            color="red",
            linestyle="--",
            linewidth=1.5,
            label=f"Threshold ({baseline_threshold:.4f})",
        )

        for i, score in enumerate(anomaly_scores):
            if score > baseline_threshold:
                ax.axvline(
                    x=t[i],
                    color="orange",
                    linestyle=":",
                    alpha=0.7,
                    label=f"Crossing at T={t[i]}",
                )
                break

        ax.set_xlabel("Timestep")
        ax.set_ylabel("Anomaly Score")
        ax.set_title(f"Anomaly Timeline — Incident {incident_id}")
        ax.legend(fontsize=8)
        ax.set_xlim(0.5, len(t) + 0.5)
        plt.tight_layout()
        fig.savefig(output_path, dpi=self._dpi, bbox_inches="tight")
        plt.close(fig)

    def get_config(self) -> Dict[str, Any]:
        """Return line graph visualizer configuration."""
        return {
            "type": "line_graph",
            "dpi": self._dpi,
            "fig_width": self._fig_width,
            "fig_height": self._fig_height,
        }
