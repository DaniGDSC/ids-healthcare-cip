"""Performance metrics collector -- rolling 24h window with threshold checking."""

from __future__ import annotations

import logging
import random
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .alert_dispatcher import AlertDispatcher
from .base import BaseMonitor, MonitorStatus
from .config import Phase7Config
from .state_machine import (
    AlertCategory,
    AlertSeverity,
    MonitoringAlert,
    PerformanceSnapshot,
)

logger = logging.getLogger(__name__)

# Simulation constants
PERF_LATENCY_MEAN_MS: float = 20.0
PERF_LATENCY_STD_MS: float = 8.0
PERF_THROUGHPUT_MEAN_SPS: float = 100.0
PERF_THROUGHPUT_STD_SPS: float = 30.0
PERF_MEMORY_MEAN_MB: float = 512.0
PERF_MEMORY_STD_MB: float = 150.0
PERF_CPU_MEAN_PCT: float = 40.0
PERF_CPU_STD_PCT: float = 15.0
MIN_LATENCY_MS: float = 1.0


class PerformanceCollector(BaseMonitor):
    """Collects rolling performance metrics and checks thresholds.

    Maintains a sliding window of metrics per engine and generates
    alerts when configurable thresholds are breached.
    """

    def __init__(
        self,
        config: Phase7Config,
        alert_dispatcher: AlertDispatcher,
        rng: random.Random,
    ) -> None:
        self._config = config
        self._dispatcher = alert_dispatcher
        self._rng = rng
        self._running = False
        self._metrics: Dict[str, List[PerformanceSnapshot]] = {e.id: [] for e in config.engines}

    async def start(self) -> None:
        """Mark performance collector as running."""
        self._running = True

    def stop(self) -> None:
        """Mark performance collector as stopped."""
        self._running = False

    def get_status(self) -> MonitorStatus:
        """Return current collector status."""
        total = sum(len(v) for v in self._metrics.values())
        return MonitorStatus(
            name="PerformanceCollector",
            running=self._running,
            details={"total_snapshots": total},
        )

    def get_config(self) -> Dict[str, Any]:
        """Return performance collector configuration."""
        return {
            "collection_interval": self._config.performance_collection_interval,
            "latency_threshold_ms": self._config.latency_p95_threshold_ms,
            "memory_threshold_pct": self._config.memory_warning_threshold_pct,
            "cpu_threshold_pct": self._config.cpu_warning_threshold_pct,
            "rolling_window_hours": self._config.rolling_window_hours,
        }

    def record(self, snapshot: PerformanceSnapshot) -> None:
        """Record a performance snapshot, pruning beyond rolling window."""
        engine_id = snapshot.engine_id
        if engine_id not in self._metrics:
            self._metrics[engine_id] = []
        self._metrics[engine_id].append(snapshot)
        max_entries = (
            self._config.rolling_window_hours * 3600 // self._config.performance_collection_interval
        )
        if len(self._metrics[engine_id]) > max_entries:
            self._metrics[engine_id] = self._metrics[engine_id][-max_entries:]

    def get_latest(self, engine_id: str) -> Optional[PerformanceSnapshot]:
        """Return the most recent snapshot for an engine."""
        entries = self._metrics.get(engine_id, [])
        return entries[-1] if entries else None

    def get_all(self, engine_id: str) -> List[PerformanceSnapshot]:
        """Return all snapshots within the rolling window."""
        return list(self._metrics.get(engine_id, []))

    def check_thresholds(self, engine_id: str) -> List[MonitoringAlert]:
        """Check latest metrics against configured thresholds."""
        latest = self.get_latest(engine_id)
        if latest is None:
            return []

        alerts: List[MonitoringAlert] = []
        now = datetime.now(timezone.utc).isoformat()
        ts = int(time.time())

        if latest.latency_p95_ms > self._config.latency_p95_threshold_ms:
            alerts.append(
                MonitoringAlert(
                    alert_id=f"PERF-LAT-{engine_id}-{ts}",
                    category=AlertCategory.PERFORMANCE.value,
                    severity=AlertSeverity.WARNING.value,
                    engine_id=engine_id,
                    timestamp=now,
                    message=(
                        f"Latency p95 {latest.latency_p95_ms:.1f}ms "
                        f"> {self._config.latency_p95_threshold_ms:.0f}ms"
                    ),
                )
            )

        mem_pct = (latest.memory_mb / self._config.reference_memory_mb) * 100.0
        if mem_pct > self._config.memory_warning_threshold_pct:
            alerts.append(
                MonitoringAlert(
                    alert_id=f"PERF-MEM-{engine_id}-{ts}",
                    category=AlertCategory.PERFORMANCE.value,
                    severity=AlertSeverity.WARNING.value,
                    engine_id=engine_id,
                    timestamp=now,
                    message=(
                        f"Memory {latest.memory_mb:.0f}MB "
                        f"({mem_pct:.1f}%) "
                        f"> {self._config.memory_warning_threshold_pct:.0f}%"
                    ),
                )
            )

        if latest.cpu_pct > self._config.cpu_warning_threshold_pct:
            alerts.append(
                MonitoringAlert(
                    alert_id=f"PERF-CPU-{engine_id}-{ts}",
                    category=AlertCategory.PERFORMANCE.value,
                    severity=AlertSeverity.WARNING.value,
                    engine_id=engine_id,
                    timestamp=now,
                    message=(
                        f"CPU {latest.cpu_pct:.1f}% "
                        f"> {self._config.cpu_warning_threshold_pct:.0f}%"
                    ),
                )
            )

        return alerts

    def simulate_metrics(self, engine_id: str) -> PerformanceSnapshot:
        """Simulate performance metrics for one engine."""
        base = self._rng.gauss(PERF_LATENCY_MEAN_MS, PERF_LATENCY_STD_MS)
        return PerformanceSnapshot(
            engine_id=engine_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            latency_p50_ms=max(MIN_LATENCY_MS, base),
            latency_p95_ms=max(MIN_LATENCY_MS, base * self._rng.uniform(1.5, 3.0)),
            latency_p99_ms=max(MIN_LATENCY_MS, base * self._rng.uniform(2.5, 5.0)),
            throughput_sps=max(
                MIN_LATENCY_MS,
                self._rng.gauss(PERF_THROUGHPUT_MEAN_SPS, PERF_THROUGHPUT_STD_SPS),
            ),
            memory_mb=max(
                50.0,
                self._rng.gauss(PERF_MEMORY_MEAN_MB, PERF_MEMORY_STD_MB),
            ),
            cpu_pct=max(
                MIN_LATENCY_MS,
                min(
                    100.0,
                    self._rng.gauss(PERF_CPU_MEAN_PCT, PERF_CPU_STD_PCT),
                ),
            ),
        )

    async def process_cycle(self) -> None:
        """Process one performance collection cycle."""
        for engine in self._config.engines:
            snapshot = self.simulate_metrics(engine.id)
            self.record(snapshot)
            alerts = self.check_thresholds(engine.id)
            for alert in alerts:
                self._dispatcher.dispatch(alert)
