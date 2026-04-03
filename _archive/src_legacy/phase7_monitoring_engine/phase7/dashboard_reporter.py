"""Dashboard reporter -- simulated WebSocket push for real-time monitoring."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from .alert_dispatcher import AlertDispatcher
from .base import BaseMonitor, MonitorStatus
from .config import Phase7Config
from .performance_collector import PerformanceCollector
from .state_machine import EngineState, StateMachine

logger = logging.getLogger(__name__)


class DashboardReporter(BaseMonitor):
    """Simulated WebSocket dashboard push (research/CI mode).

    Pushes engine states, metrics, and alert summaries at
    configurable intervals.
    """

    def __init__(
        self,
        config: Phase7Config,
        state_machines: Dict[str, StateMachine],
        performance_collector: PerformanceCollector,
        alert_dispatcher: AlertDispatcher,
    ) -> None:
        self._config = config
        self._machines = state_machines
        self._perf = performance_collector
        self._dispatcher = alert_dispatcher
        self._running = False
        self._push_count: int = 0
        self._last_payload: Optional[Dict[str, Any]] = None

    async def start(self) -> None:
        """Mark dashboard reporter as running."""
        self._running = True

    def stop(self) -> None:
        """Mark dashboard reporter as stopped."""
        self._running = False

    def get_status(self) -> MonitorStatus:
        """Return current dashboard reporter status."""
        return MonitorStatus(
            name="DashboardReporter",
            running=self._running,
            details={"push_count": self._push_count},
        )

    def get_config(self) -> Dict[str, Any]:
        """Return dashboard configuration."""
        return {
            "push_interval_seconds": self._config.dashboard_push_interval_seconds,
            "protocol": "WebSocket (simulated)",
        }

    def push_update(self) -> Dict[str, Any]:
        """Push dashboard update (simulated WebSocket).

        Returns:
            Dashboard payload dict.
        """
        self._push_count += 1
        states = {eid: m.state.value for eid, m in self._machines.items()}
        metrics: Dict[str, Optional[Dict[str, float]]] = {}
        for eid in self._machines:
            latest = self._perf.get_latest(eid)
            if latest:
                metrics[eid] = {
                    "latency_p95_ms": latest.latency_p95_ms,
                    "memory_mb": latest.memory_mb,
                    "cpu_pct": latest.cpu_pct,
                }
            else:
                metrics[eid] = None

        alert_counts = self._dispatcher.count_by_severity()
        payload: Dict[str, Any] = {
            "push_id": self._push_count,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "engines": {},
            "system_overview": {},
            "active_alerts": alert_counts,
        }

        for eid in self._machines:
            payload["engines"][eid] = {
                "state": states.get(eid, EngineState.UNKNOWN.value),
                "metrics": metrics.get(eid),
            }

        for state in EngineState:
            payload["system_overview"][state.value] = sum(
                1 for s in states.values() if s == state.value
            )

        self._last_payload = payload
        logger.debug("Dashboard push #%d", self._push_count)
        return payload

    async def process_cycle(self) -> None:
        """Process one dashboard push cycle."""
        self.push_update()

    @property
    def push_count(self) -> int:
        """Return total number of pushes."""
        return self._push_count
