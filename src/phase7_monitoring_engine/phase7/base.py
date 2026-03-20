"""Abstract base class for Phase 7 monitors.

New monitor types can be added by subclassing BaseMonitor
without modifying existing monitor code (Open/Closed Principle).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class MonitorStatus:
    """Status snapshot of a running monitor."""

    name: str
    running: bool
    details: Dict[str, Any] = field(default_factory=dict)


class BaseMonitor(ABC):
    """Interface for all Phase 7 monitoring components.

    Subclasses: HeartbeatReceiver, PerformanceCollector,
    ArtifactIntegrityChecker, SecurityMonitor, DashboardReporter.
    """

    @abstractmethod
    async def start(self) -> None:
        """Start the monitoring loop (async)."""

    @abstractmethod
    def stop(self) -> None:
        """Stop the monitoring loop."""

    @abstractmethod
    def get_status(self) -> MonitorStatus:
        """Return current monitor status."""

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Return monitor configuration for the pipeline report."""
