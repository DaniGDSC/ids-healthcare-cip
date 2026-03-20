"""Unified alert tracking and dispatch.

All monitors inject AlertDispatcher to report alerts (Dependency Inversion).
"""

from __future__ import annotations

import logging
from typing import Dict, List

from .state_machine import AlertCategory, AlertSeverity, MonitoringAlert

logger = logging.getLogger(__name__)


class AlertDispatcher:
    """Tracks and dispatches monitoring alerts.

    Injected into all monitors to decouple alert handling
    from monitoring logic (Dependency Inversion Principle).
    """

    def __init__(self) -> None:
        self._alerts: List[MonitoringAlert] = []

    def dispatch(self, alert: MonitoringAlert) -> None:
        """Register and log a new alert."""
        self._alerts.append(alert)
        logger.warning(
            "[ALERT] %s | %s | %s | %s",
            alert.severity,
            alert.category,
            alert.engine_id,
            alert.message,
        )

    def get_active(self) -> List[MonitoringAlert]:
        """Return all unresolved alerts."""
        return [a for a in self._alerts if not a.resolved]

    def count_by_severity(self) -> Dict[str, int]:
        """Count active alerts per severity level."""
        counts: Dict[str, int] = {s.value: 0 for s in AlertSeverity}
        for a in self.get_active():
            counts[a.severity] = counts.get(a.severity, 0) + 1
        return counts

    def count_by_category(self) -> Dict[str, int]:
        """Count all alerts per category."""
        counts: Dict[str, int] = {c.value: 0 for c in AlertCategory}
        for a in self._alerts:
            counts[a.category] = counts.get(a.category, 0) + 1
        return counts

    @property
    def all_alerts(self) -> List[MonitoringAlert]:
        """Return all alerts."""
        return list(self._alerts)
