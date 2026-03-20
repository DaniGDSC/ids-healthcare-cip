"""Security monitor -- coordinates integrity checking and generates security alerts."""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List

from .alert_dispatcher import AlertDispatcher
from .base import BaseMonitor, MonitorStatus
from .config import Phase7Config
from .integrity_checker import ArtifactIntegrityChecker
from .state_machine import (
    AlertCategory,
    AlertSeverity,
    MonitoringAlert,
    SecurityEvent,
)

logger = logging.getLogger(__name__)


class SecurityMonitor(BaseMonitor):
    """Coordinates security monitoring and generates CRITICAL alerts.

    Delegates SHA-256 verification to ArtifactIntegrityChecker
    and dispatches alerts for any violations detected.
    """

    def __init__(
        self,
        config: Phase7Config,
        alert_dispatcher: AlertDispatcher,
        integrity_checker: ArtifactIntegrityChecker,
    ) -> None:
        self._config = config
        self._dispatcher = alert_dispatcher
        self._checker = integrity_checker
        self._running = False

    async def start(self) -> None:
        """Mark security monitor as running."""
        self._running = True

    def stop(self) -> None:
        """Mark security monitor as stopped."""
        self._running = False

    def get_status(self) -> MonitorStatus:
        """Return current security monitor status."""
        return MonitorStatus(
            name="SecurityMonitor",
            running=self._running,
            details=self._checker.get_status().details,
        )

    def get_config(self) -> Dict[str, Any]:
        """Return security monitor configuration."""
        return {
            "integrity_check_interval": self._config.artifact_integrity_check_interval,
            "baseline_check_interval": self._config.baseline_check_interval_seconds,
            "baseline_config_path": self._config.baseline_config_path,
        }

    def _handle_security_events(self, events: List[SecurityEvent]) -> None:
        """Generate CRITICAL alerts for security violations."""
        for evt in events:
            if evt.severity == AlertSeverity.CRITICAL.value:
                ts = int(time.time())
                self._dispatcher.dispatch(
                    MonitoringAlert(
                        alert_id=f"SEC-{evt.event_type}-{evt.engine_id}-{ts}",
                        category=AlertCategory.SECURITY.value,
                        severity=AlertSeverity.CRITICAL.value,
                        engine_id=evt.engine_id,
                        timestamp=evt.timestamp,
                        message=f"Security: {evt.detail}",
                    )
                )

    async def process_cycle(self) -> None:
        """Process one security verification cycle."""
        events = self._checker.verify_all_artifacts()
        self._handle_security_events(events)
        bl_event = self._checker.verify_baseline_config()
        if bl_event and bl_event.severity == AlertSeverity.CRITICAL.value:
            self._handle_security_events([bl_event])

    @property
    def all_events(self) -> List[SecurityEvent]:
        """Return all security events."""
        return self._checker.all_events
