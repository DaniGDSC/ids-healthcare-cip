"""Heartbeat receiver -- async, non-blocking heartbeat processing.

Simulates heartbeat reception by checking artifact directory existence.
Adaptive timeout based on grace period configuration.
"""

from __future__ import annotations

import logging
import random
import time
from pathlib import Path
from typing import Any, Dict, Optional

from .alert_dispatcher import AlertDispatcher
from .base import BaseMonitor, MonitorStatus
from .config import EngineEntry, Phase7Config
from .state_change_logger import StateChangeLogger
from .state_machine import (
    AlertCategory,
    AlertSeverity,
    EngineState,
    MonitoringAlert,
    StateChangeEvent,
    StateMachine,
)

logger = logging.getLogger(__name__)


class HeartbeatReceiver(BaseMonitor):
    """Receives and processes engine heartbeats.

    Simulates heartbeat reception by checking if artifact directories exist.
    Uses adaptive timeout based on grace period configuration.
    """

    def __init__(
        self,
        config: Phase7Config,
        state_machines: Dict[str, StateMachine],
        alert_dispatcher: AlertDispatcher,
        state_change_logger: StateChangeLogger,
        project_root: Path,
        rng: random.Random,
    ) -> None:
        self._config = config
        self._machines = state_machines
        self._dispatcher = alert_dispatcher
        self._logger = state_change_logger
        self._root = project_root
        self._rng = rng
        self._running = False

    async def start(self) -> None:
        """Mark heartbeat receiver as running."""
        self._running = True

    def stop(self) -> None:
        """Mark heartbeat receiver as stopped."""
        self._running = False

    def get_status(self) -> MonitorStatus:
        """Return current heartbeat receiver status."""
        return MonitorStatus(
            name="HeartbeatReceiver",
            running=self._running,
            details={
                "engines": len(self._machines),
                "interval": self._config.heartbeat_interval_seconds,
            },
        )

    def get_config(self) -> Dict[str, Any]:
        """Return heartbeat configuration."""
        return {
            "heartbeat_interval_seconds": self._config.heartbeat_interval_seconds,
            "missed_heartbeat_threshold": self._config.missed_heartbeat_threshold,
            "grace_period_seconds": self._config.grace_period_seconds,
            "miss_probability": self._config.heartbeat_miss_probability,
            "spike_probability": self._config.heartbeat_spike_probability,
        }

    def _simulate_heartbeat(self, engine: EngineEntry) -> Optional[float]:
        """Simulate heartbeat by checking artifact existence.

        Returns:
            Latency in ms if heartbeat received, None if missed.
        """
        if engine.artifact_dir:
            artifact_dir = self._root / engine.artifact_dir
            if not artifact_dir.exists():
                return None

        if self._rng.random() < self._config.heartbeat_miss_probability:
            return None

        if self._rng.random() < self._config.heartbeat_spike_probability:
            latency = self._rng.gauss(
                self._config.spike_latency_mean_ms,
                self._config.spike_latency_std_ms,
            )
        else:
            latency = self._rng.gauss(
                self._config.normal_latency_mean_ms,
                self._config.normal_latency_std_ms,
            )
        return max(1.0, latency)

    async def process_cycle(self) -> None:
        """Process one heartbeat cycle for all engines."""
        for engine in self._config.engines:
            machine = self._machines.get(engine.id)
            if machine is None:
                continue

            latency = self._simulate_heartbeat(engine)
            if latency is not None:
                event = await machine.process_heartbeat(
                    latency, self._config.latency_p95_threshold_ms
                )
            else:
                event = await machine.tick_missed(self._config.missed_heartbeat_threshold)

            if event is not None:
                self._logger.log(event)
                self._generate_alert(event)

    def _generate_alert(self, event: StateChangeEvent) -> None:
        """Generate health alert on DOWN or DEGRADED transition."""
        ts = int(time.time())
        if event.new_state == EngineState.DOWN.value:
            self._dispatcher.dispatch(
                MonitoringAlert(
                    alert_id=f"HEALTH-DOWN-{event.engine_id}-{ts}",
                    category=AlertCategory.HEALTH.value,
                    severity=AlertSeverity.CRITICAL.value,
                    engine_id=event.engine_id,
                    timestamp=event.timestamp,
                    message=(
                        f"{event.engine_id} DOWN: {event.reason}. " "Notify IT admin immediately."
                    ),
                )
            )
        elif event.new_state == EngineState.DEGRADED.value:
            self._dispatcher.dispatch(
                MonitoringAlert(
                    alert_id=f"HEALTH-DEG-{event.engine_id}-{ts}",
                    category=AlertCategory.HEALTH.value,
                    severity=AlertSeverity.WARNING.value,
                    engine_id=event.engine_id,
                    timestamp=event.timestamp,
                    message=(
                        f"{event.engine_id} DEGRADED: {event.reason}. "
                        "Notify IT admin within 5 minutes."
                    ),
                )
            )
