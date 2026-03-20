"""Monitoring pipeline -- asyncio.gather() orchestrator for all monitors."""

from __future__ import annotations

import asyncio
import logging
import random
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .alert_dispatcher import AlertDispatcher
from .config import Phase7Config
from .dashboard_reporter import DashboardReporter
from .exporter import MonitoringExporter
from .heartbeat_receiver import HeartbeatReceiver
from .integrity_checker import ArtifactIntegrityChecker
from .performance_collector import PerformanceCollector
from .report import render_monitoring_report
from .security_monitor import SecurityMonitor
from .state_change_logger import StateChangeLogger
from .state_machine import (
    PerformanceSnapshot,
    StateMachine,
    build_transition_table,
)

logger = logging.getLogger(__name__)


def _detect_hardware() -> Dict[str, str]:
    """Detect hardware and environment info."""
    import platform as plat

    hw: Dict[str, str] = {
        "python": plat.python_version(),
        "platform": plat.platform(),
        "device": f"CPU: {plat.machine()}",
    }
    try:
        import tensorflow as tf

        hw["tensorflow"] = tf.__version__
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            hw["device"] = f"GPU: {gpus[0].name}"
            hw["cuda"] = "available"
        else:
            hw["cuda"] = "N/A (CPU execution)"
    except ImportError:
        hw["tensorflow"] = "N/A"
        hw["cuda"] = "N/A"
    return hw


def _get_git_commit(project_root: Path) -> str:
    """Return current git commit hash or 'unknown'."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=str(project_root),
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except (subprocess.SubprocessError, FileNotFoundError):
        return "unknown"


class MonitoringPipeline:
    """Orchestrates all monitoring subsystems via asyncio.gather().

    Creates and wires all SOLID components, runs concurrent loops,
    exports artifacts, and generates reports.

    Args:
        config: Validated Phase7Config instance.
        project_root: Project root directory.
    """

    def __init__(
        self,
        config: Phase7Config,
        project_root: Path,
    ) -> None:
        self._config = config
        self._root = project_root
        self._rng = random.Random(config.random_state)

        # Build transition table from config
        if config.transitions:
            rules = [
                {
                    "from_state": t.from_state,
                    "trigger": t.trigger,
                    "to_state": t.to_state,
                }
                for t in config.transitions
            ]
            transition_table = build_transition_table(rules)
        else:
            transition_table = None  # uses DEFAULT_TRANSITIONS

        # Create state machines (one per engine)
        self._machines: Dict[str, StateMachine] = {
            engine.id: StateMachine(
                engine_id=engine.id,
                transitions=transition_table,
                buffer_size=config.circular_buffer_size,
                consecutive_ok_threshold=config.consecutive_ok_threshold,
            )
            for engine in config.engines
        }

        # Create shared components (Dependency Injection)
        self._alert_dispatcher = AlertDispatcher()
        self._state_logger = StateChangeLogger(config.circular_buffer_size)

        # Create monitors (AlertDispatcher injected into all)
        self._heartbeat = HeartbeatReceiver(
            config=config,
            state_machines=self._machines,
            alert_dispatcher=self._alert_dispatcher,
            state_change_logger=self._state_logger,
            project_root=project_root,
            rng=self._rng,
        )
        self._performance = PerformanceCollector(
            config=config,
            alert_dispatcher=self._alert_dispatcher,
            rng=self._rng,
        )
        self._integrity = ArtifactIntegrityChecker(
            config=config,
            alert_dispatcher=self._alert_dispatcher,
            project_root=project_root,
        )
        self._security = SecurityMonitor(
            config=config,
            alert_dispatcher=self._alert_dispatcher,
            integrity_checker=self._integrity,
        )
        self._dashboard = DashboardReporter(
            config=config,
            state_machines=self._machines,
            performance_collector=self._performance,
            alert_dispatcher=self._alert_dispatcher,
        )
        self._exporter = MonitoringExporter(project_root / config.output_dir)

    def _engine_names(self) -> Dict[str, str]:
        """Return engine name lookup."""
        return {e.id: (e.name or e.id) for e in self._config.engines}

    # ---------------------------------------------------------------
    # Async monitoring loops
    # ---------------------------------------------------------------
    async def _heartbeat_loop(self, n_cycles: int) -> None:
        """Monitor engine heartbeats for n_cycles iterations."""
        for _ in range(n_cycles):
            await self._heartbeat.process_cycle()
            await asyncio.sleep(0.01)

    async def _performance_loop(self, n_cycles: int) -> None:
        """Collect performance metrics for n_cycles iterations."""
        for _ in range(n_cycles):
            await self._performance.process_cycle()
            await asyncio.sleep(0.01)

    async def _security_loop(self, n_cycles: int) -> None:
        """Verify artifact integrity for n_cycles iterations."""
        for _ in range(n_cycles):
            await self._security.process_cycle()
            await asyncio.sleep(0.01)

    async def _dashboard_loop(self, n_cycles: int) -> None:
        """Push dashboard updates for n_cycles iterations."""
        for _ in range(n_cycles):
            await self._dashboard.process_cycle()
            await asyncio.sleep(0.01)

    # ---------------------------------------------------------------
    # Pipeline execution
    # ---------------------------------------------------------------
    async def run_async(self, n_cycles: Optional[int] = None) -> Dict[str, Any]:
        """Run all monitoring loops concurrently.

        Args:
            n_cycles: Override number of monitoring iterations per loop.

        Returns:
            Pipeline summary dict.
        """
        cycles = n_cycles or self._config.n_cycles
        logger.info(
            "Pipeline starting: %d cycles, %d engines",
            cycles,
            len(self._config.engines),
        )
        t0 = time.time()

        # Start all monitors
        await self._heartbeat.start()
        await self._performance.start()
        await self._security.start()
        await self._dashboard.start()

        # Run concurrent loops
        await asyncio.gather(
            self._heartbeat_loop(cycles),
            self._performance_loop(cycles),
            self._security_loop(cycles),
            self._dashboard_loop(cycles),
        )

        # Stop all monitors
        self._heartbeat.stop()
        self._performance.stop()
        self._security.stop()
        self._dashboard.stop()

        duration = time.time() - t0
        engine_states = {eid: m.state.value for eid, m in self._machines.items()}
        state_counts: Dict[str, int] = {}
        for s in engine_states.values():
            state_counts[s] = state_counts.get(s, 0) + 1

        return {
            "duration_seconds": round(duration, 2),
            "n_cycles": cycles,
            "n_engines": len(self._config.engines),
            "engine_states": engine_states,
            "state_summary": state_counts,
            "total_state_changes": len(self._state_logger.all_events),
            "total_alerts": len(self._alert_dispatcher.all_alerts),
            "alert_severity_counts": self._alert_dispatcher.count_by_severity(),
            "alert_category_counts": self._alert_dispatcher.count_by_category(),
            "security_events": len(self._security.all_events),
            "dashboard_pushes": self._dashboard.push_count,
        }

    def run(self, n_cycles: Optional[int] = None) -> Dict[str, Any]:
        """Synchronous wrapper for run_async()."""
        return asyncio.run(self.run_async(n_cycles))

    # ---------------------------------------------------------------
    # Data accessors for export
    # ---------------------------------------------------------------
    @property
    def state_changes(self) -> list:
        """Return all state change events."""
        return self._state_logger.all_events

    @property
    def alerts(self) -> list:
        """Return all generated alerts."""
        return self._alert_dispatcher.all_alerts

    @property
    def security_events(self) -> list:
        """Return all security events."""
        return self._security.all_events

    @property
    def engine_states(self) -> Dict[str, str]:
        """Return current state per engine."""
        return {eid: m.state.value for eid, m in self._machines.items()}

    @property
    def performance_snapshots(self) -> Dict[str, List[PerformanceSnapshot]]:
        """Return all performance snapshots per engine."""
        return {e.id: self._performance.get_all(e.id) for e in self._config.engines}

    @property
    def dashboard(self) -> DashboardReporter:
        """Return the dashboard reporter instance."""
        return self._dashboard

    def export_artifacts(self, summary: Dict[str, Any]) -> Dict[str, str]:
        """Export all monitoring artifacts."""
        return self._exporter.export(
            state_changes=self.state_changes,
            engine_states=self.engine_states,
            engine_names=self._engine_names(),
            performance_data=self.performance_snapshots,
            security_events=self.security_events,
            alerts=self.alerts,
            summary=summary,
        )

    def generate_report(
        self,
        report_dir: Path,
        summary: Dict[str, Any],
        artifact_hashes: Dict[str, str],
        git_commit: str,
        hardware: Dict[str, str],
    ) -> Path:
        """Generate monitoring report."""
        return render_monitoring_report(
            report_dir=report_dir,
            summary=summary,
            state_changes=self.state_changes,
            engine_states=self.engine_states,
            engine_names=self._engine_names(),
            performance_data=self.performance_snapshots,
            security_events=self.security_events,
            alerts=self.alerts,
            artifact_hashes=artifact_hashes,
            git_commit=git_commit,
            hardware=hardware,
            config=self._config,
        )
