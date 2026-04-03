"""Abstract base class for detection/computation components.

Shared across Phase 2 (AttentionAnomalyDetector), Phase 4 (risk engine
components), Phase 5 (ConditionalExplainer), and dashboard
(CognitiveTranslator). Placed here to avoid circular dependencies
between phases.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseDetector(ABC):
    """Interface for detection and computation components.

    Subclasses must implement ``get_config()`` to support
    pipeline reporting and configuration export.
    """

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Return component configuration for the pipeline report.

        Returns:
            Dict with component-specific configuration.
        """
