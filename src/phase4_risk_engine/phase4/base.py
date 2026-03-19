"""Abstract base class for Phase 4 detection/computation components.

Subclasses must implement ``get_config()``.
New detector types can be added without modifying existing code
(Open/Closed Principle).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseDetector(ABC):
    """Interface for all Phase 4 detection and computation components.

    Subclasses: BaselineComputer, DynamicThresholdUpdater,
    ConceptDriftDetector, CrossModalFusionDetector.
    """

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Return component configuration for the pipeline report.

        Returns:
            Dict with component-specific configuration.
        """
