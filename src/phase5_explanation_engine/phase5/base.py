"""Abstract base class for Phase 5 visualizers.

New chart types can be added by subclassing BaseVisualizer
without modifying existing visualizer code (Open/Closed Principle).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict


class BaseVisualizer(ABC):
    """Interface for all Phase 5 visualization components.

    Subclasses: WaterfallVisualizer, BarChartVisualizer, LineGraphVisualizer.
    """

    @abstractmethod
    def plot(self, data: Dict[str, Any], output_path: Path) -> None:
        """Render visualization and save to *output_path*.

        Args:
            data: Visualization-specific data dict.
            output_path: Path to save the rendered PNG.
        """

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Return visualizer configuration for the pipeline report."""
