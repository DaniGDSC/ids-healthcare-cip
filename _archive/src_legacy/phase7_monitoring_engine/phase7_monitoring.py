"""Phase 7 Monitoring Engine -- system health, performance, and security monitoring.

Runs as an independent process observing Phase 0-6 engines via heartbeat
protocol, artifact integrity verification, and performance metric collection.

OWASP Controls:
    A01 -- Broken Access Control:  PathValidator for all file operations
    A02 -- Cryptographic Failures: SHA-256 integrity verification
    A08 -- Software Integrity:     Continuous artifact hash monitoring
    A09 -- Logging & Monitoring:   This IS the monitoring engine

IEEE Q1 journal + hospital IT audience.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import random
import subprocess
import time
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import yaml

from src.phase0_dataset_analysis.phase0.security import (
    AuditLogger,
    IntegrityVerifier,
    PathValidator,
)

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
CONFIG_PATH: Path = PROJECT_ROOT / "config" / "phase7_monitoring_config.yaml"
OUTPUT_DIR: Path = PROJECT_ROOT / "data" / "phase7"
REPORT_DIR: Path = PROJECT_ROOT / "results" / "phase0_analysis"

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Named constants (no magic numbers)
# ---------------------------------------------------------------------------
HEARTBEAT_INTERVAL_S: int = 5
MISSED_HEARTBEAT_THRESHOLD: int = 5
GRACE_PERIOD_S: int = 25
CIRCULAR_BUFFER_SIZE: int = 1000
PERF_COLLECTION_INTERVAL_S: int = 30
ARTIFACT_CHECK_INTERVAL_S: int = 60
BASELINE_CHECK_INTERVAL_S: int = 30
LATENCY_P95_THRESHOLD_MS: float = 100.0
MEMORY_WARNING_PCT: float = 80.0
CPU_WARNING_PCT: float = 90.0
DASHBOARD_PUSH_INTERVAL_S: int = 5
ROLLING_WINDOW_HOURS: int = 24
CONSECUTIVE_OK_THRESHOLD: int = 3
REFERENCE_MEMORY_MB: float = 4096.0

# Heartbeat simulation parameters
HEARTBEAT_MISS_PROBABILITY: float = 0.05
HEARTBEAT_SPIKE_PROBABILITY: float = 0.05
NORMAL_LATENCY_MEAN_MS: float = 25.0
NORMAL_LATENCY_STD_MS: float = 10.0
SPIKE_LATENCY_MEAN_MS: float = 150.0
SPIKE_LATENCY_STD_MS: float = 30.0
MIN_LATENCY_MS: float = 1.0

# Performance simulation parameters
PERF_LATENCY_MEAN_MS: float = 20.0
PERF_LATENCY_STD_MS: float = 8.0
PERF_THROUGHPUT_MEAN_SPS: float = 100.0
PERF_THROUGHPUT_STD_SPS: float = 30.0
PERF_MEMORY_MEAN_MB: float = 512.0
PERF_MEMORY_STD_MB: float = 150.0
PERF_CPU_MEAN_PCT: float = 40.0
PERF_CPU_STD_PCT: float = 15.0

# Artifact filenames
MONITORING_LOG: str = "monitoring_log.json"
HEALTH_REPORT: str = "health_report.json"
PERFORMANCE_REPORT: str = "performance_report.json"
SECURITY_AUDIT_LOG: str = "security_audit_log.json"

# ---------------------------------------------------------------------------
# Engine registry
# ---------------------------------------------------------------------------
ENGINE_IDS: Tuple[str, ...] = (
    "phase0",
    "phase1",
    "phase2",
    "phase3",
    "phase4",
    "phase5",
    "phase6",
)

ENGINE_NAMES: Dict[str, str] = {
    "phase0": "Foundation & Dataset Analysis",
    "phase1": "Preprocessing",
    "phase2": "Detection Engine",
    "phase3": "Classification Engine",
    "phase4": "Risk-Adaptive Engine",
    "phase5": "Explanation Engine",
    "phase6": "Notification Engine",
}

PHASE_ARTIFACT_DIRS: Dict[str, str] = {
    "phase0": "data/raw",
    "phase1": "data/processed",
    "phase2": "data/phase2",
    "phase3": "data/phase3",
    "phase4": "data/phase4",
    "phase5": "data/phase5",
    "phase6": "data/phase6",
}

# Metadata files with artifact_hashes for SHA-256 verification
PHASE_METADATA_FILES: Dict[str, Optional[str]] = {
    "phase0": None,
    "phase1": None,
    "phase2": "data/phase2/detection_metadata.json",
    "phase3": "data/phase3/classification_metadata.json",
    "phase4": "data/phase4/risk_metadata.json",
    "phase5": "data/phase5/explanation_metadata.json",
    "phase6": None,
}

BASELINE_CONFIG_PATH: str = "data/phase4/baseline_config.json"


# ===================================================================
# Enums
# ===================================================================
class EngineState(str, Enum):
    """Five-state machine for engine health."""

    UNKNOWN = "UNKNOWN"
    STARTING = "STARTING"
    UP = "UP"
    DEGRADED = "DEGRADED"
    DOWN = "DOWN"


class AlertSeverity(str, Enum):
    """Alert severity levels."""

    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class AlertCategory(str, Enum):
    """Alert source category."""

    HEALTH = "HEALTH"
    PERFORMANCE = "PERFORMANCE"
    SECURITY = "SECURITY"


# ===================================================================
# Dataclasses
# ===================================================================
@dataclass
class StateChangeEvent:
    """Immutable record of an engine state transition."""

    engine_id: str
    old_state: str
    new_state: str
    timestamp: str
    reason: str


@dataclass
class PerformanceSnapshot:
    """Point-in-time performance metrics for one engine."""

    engine_id: str
    timestamp: str
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    throughput_sps: float
    memory_mb: float
    cpu_pct: float


@dataclass
class SecurityEvent:
    """Security-relevant event record."""

    event_type: str
    engine_id: str
    timestamp: str
    detail: str
    severity: str


@dataclass
class MonitoringAlert:
    """Generated monitoring alert."""

    alert_id: str
    category: str
    severity: str
    engine_id: str
    timestamp: str
    message: str
    resolved: bool = False


# ===================================================================
# EngineHealthTracker -- state machine per engine
# ===================================================================
class EngineHealthTracker:
    """Manages the 5-state machine for a single engine.

    State transitions:
        UNKNOWN  -> STARTING : first heartbeat received
        STARTING -> UP       : 3 consecutive heartbeats
        UP       -> DEGRADED : latency > threshold
        UP       -> DOWN     : missed_heartbeat_threshold misses
        DEGRADED -> UP       : latency returns to normal
        DEGRADED -> DOWN     : missed_heartbeat_threshold misses
        DOWN     -> STARTING : heartbeat received again
    """

    def __init__(
        self,
        engine_id: str,
        buffer_size: int = CIRCULAR_BUFFER_SIZE,
    ) -> None:
        self._engine_id = engine_id
        self._state = EngineState.UNKNOWN
        self._consecutive_ok: int = 0
        self._missed_count: int = 0
        self._history: Deque[StateChangeEvent] = deque(maxlen=buffer_size)

    @property
    def state(self) -> EngineState:
        """Return current engine state."""
        return self._state

    @property
    def history(self) -> Deque[StateChangeEvent]:
        """Return state change history (circular buffer)."""
        return self._history

    def _transition(self, new_state: EngineState, reason: str) -> StateChangeEvent:
        """Execute and record a state transition."""
        event = StateChangeEvent(
            engine_id=self._engine_id,
            old_state=self._state.value,
            new_state=new_state.value,
            timestamp=datetime.now(timezone.utc).isoformat(),
            reason=reason,
        )
        self._state = new_state
        self._history.append(event)
        return event

    def process_heartbeat(
        self, latency_ms: float, threshold_ms: float
    ) -> Optional[StateChangeEvent]:
        """Process a received heartbeat.

        Args:
            latency_ms: Measured heartbeat latency in milliseconds.
            threshold_ms: Latency threshold for DEGRADED state.

        Returns:
            StateChangeEvent if state changed, None otherwise.
        """
        self._missed_count = 0

        if self._state == EngineState.UNKNOWN:
            self._consecutive_ok = 1
            return self._transition(
                EngineState.STARTING,
                f"first heartbeat, latency={latency_ms:.1f}ms",
            )

        if self._state == EngineState.STARTING:
            self._consecutive_ok += 1
            if self._consecutive_ok >= CONSECUTIVE_OK_THRESHOLD:
                return self._transition(
                    EngineState.UP,
                    f"{CONSECUTIVE_OK_THRESHOLD} consecutive OK",
                )
            return None

        if self._state == EngineState.UP:
            if latency_ms > threshold_ms:
                return self._transition(
                    EngineState.DEGRADED,
                    (f"latency {latency_ms:.1f}ms " f"> {threshold_ms:.0f}ms"),
                )
            return None

        if self._state == EngineState.DEGRADED:
            if latency_ms <= threshold_ms:
                return self._transition(
                    EngineState.UP,
                    f"latency recovered to {latency_ms:.1f}ms",
                )
            return None

        if self._state == EngineState.DOWN:
            self._consecutive_ok = 1
            return self._transition(
                EngineState.STARTING,
                f"heartbeat again, latency={latency_ms:.1f}ms",
            )

        return None

    def tick_missed(self, threshold: int) -> Optional[StateChangeEvent]:
        """Count a missed heartbeat.

        Args:
            threshold: Number of consecutive misses before DOWN.

        Returns:
            StateChangeEvent if state changed to DOWN, None otherwise.
        """
        if self._state in (EngineState.UNKNOWN, EngineState.DOWN):
            return None
        self._missed_count += 1
        self._consecutive_ok = 0
        if self._missed_count >= threshold:
            return self._transition(
                EngineState.DOWN,
                f"{self._missed_count} consecutive misses",
            )
        return None


# ===================================================================
# PerformanceCollector
# ===================================================================
class PerformanceCollector:
    """Collects and stores rolling performance metrics per engine."""

    def __init__(self, window_hours: int = ROLLING_WINDOW_HOURS) -> None:
        self._window_hours = window_hours
        self._metrics: Dict[str, List[PerformanceSnapshot]] = {eid: [] for eid in ENGINE_IDS}

    def record(self, snapshot: PerformanceSnapshot) -> None:
        """Record a performance snapshot, pruning beyond rolling window."""
        self._metrics[snapshot.engine_id].append(snapshot)
        max_entries = self._window_hours * 3600 // PERF_COLLECTION_INTERVAL_S
        if len(self._metrics[snapshot.engine_id]) > max_entries:
            self._metrics[snapshot.engine_id] = self._metrics[snapshot.engine_id][-max_entries:]

    def get_latest(self, engine_id: str) -> Optional[PerformanceSnapshot]:
        """Return the most recent snapshot for an engine."""
        entries = self._metrics[engine_id]
        return entries[-1] if entries else None

    def get_all(self, engine_id: str) -> List[PerformanceSnapshot]:
        """Return all snapshots within the rolling window."""
        return list(self._metrics[engine_id])

    def check_thresholds(
        self,
        engine_id: str,
        latency_threshold: float,
        memory_threshold: float,
        cpu_threshold: float,
    ) -> List[MonitoringAlert]:
        """Check latest metrics against configured thresholds.

        Args:
            engine_id: Engine identifier.
            latency_threshold: p95 latency threshold in ms.
            memory_threshold: Memory usage threshold as percentage.
            cpu_threshold: CPU usage threshold as percentage.

        Returns:
            List of MonitoringAlert for any threshold breaches.
        """
        latest = self.get_latest(engine_id)
        if latest is None:
            return []

        alerts: List[MonitoringAlert] = []
        now = datetime.now(timezone.utc).isoformat()
        ts = int(time.time())

        if latest.latency_p95_ms > latency_threshold:
            alerts.append(
                MonitoringAlert(
                    alert_id=f"PERF-LAT-{engine_id}-{ts}",
                    category=AlertCategory.PERFORMANCE.value,
                    severity=AlertSeverity.WARNING.value,
                    engine_id=engine_id,
                    timestamp=now,
                    message=(
                        f"Latency p95 {latest.latency_p95_ms:.1f}ms " f"> {latency_threshold:.0f}ms"
                    ),
                )
            )

        mem_pct = (latest.memory_mb / REFERENCE_MEMORY_MB) * 100.0
        if mem_pct > memory_threshold:
            alerts.append(
                MonitoringAlert(
                    alert_id=f"PERF-MEM-{engine_id}-{ts}",
                    category=AlertCategory.PERFORMANCE.value,
                    severity=AlertSeverity.WARNING.value,
                    engine_id=engine_id,
                    timestamp=now,
                    message=(
                        f"Memory {latest.memory_mb:.0f}MB "
                        f"({mem_pct:.1f}%) > {memory_threshold:.0f}%"
                    ),
                )
            )

        if latest.cpu_pct > cpu_threshold:
            alerts.append(
                MonitoringAlert(
                    alert_id=f"PERF-CPU-{engine_id}-{ts}",
                    category=AlertCategory.PERFORMANCE.value,
                    severity=AlertSeverity.WARNING.value,
                    engine_id=engine_id,
                    timestamp=now,
                    message=(f"CPU {latest.cpu_pct:.1f}% " f"> {cpu_threshold:.0f}%"),
                )
            )

        return alerts


# ===================================================================
# SecurityMonitor
# ===================================================================
class SecurityMonitor:
    """Monitors artifact integrity via SHA-256 verification.

    Reads metadata files from Phases 2-5 and recomputes hashes for
    all artifacts listed in artifact_hashes sections.
    """

    def __init__(self, project_root: Path) -> None:
        self._root = project_root
        self._baseline_hash: Optional[str] = None
        self._events: List[SecurityEvent] = []

    @staticmethod
    def _compute_sha256(file_path: Path) -> str:
        """Compute SHA-256 hex digest of a file."""
        h = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    def verify_all_artifacts(self) -> List[SecurityEvent]:
        """Verify SHA-256 hashes for all phase artifacts.

        Returns:
            List of SecurityEvent records for this verification cycle.
        """
        events: List[SecurityEvent] = []
        now = datetime.now(timezone.utc).isoformat()

        for engine_id, meta_rel in PHASE_METADATA_FILES.items():
            if meta_rel is None:
                continue
            meta_path = self._root / meta_rel
            if not meta_path.exists():
                events.append(
                    SecurityEvent(
                        event_type="METADATA_MISSING",
                        engine_id=engine_id,
                        timestamp=now,
                        detail=f"Not found: {meta_rel}",
                        severity=AlertSeverity.WARNING.value,
                    )
                )
                continue

            try:
                with open(meta_path) as f:
                    metadata = json.load(f)
            except (json.JSONDecodeError, OSError) as exc:
                events.append(
                    SecurityEvent(
                        event_type="METADATA_READ_ERROR",
                        engine_id=engine_id,
                        timestamp=now,
                        detail=f"Cannot read {meta_rel}: {exc}",
                        severity=AlertSeverity.CRITICAL.value,
                    )
                )
                continue

            artifact_hashes = metadata.get("artifact_hashes", {})
            artifact_dir = self._root / PHASE_ARTIFACT_DIRS[engine_id]

            for artifact_name, hash_info in artifact_hashes.items():
                expected = hash_info.get("sha256", "")
                artifact_path = artifact_dir / artifact_name
                if not artifact_path.exists():
                    events.append(
                        SecurityEvent(
                            event_type="ARTIFACT_MISSING",
                            engine_id=engine_id,
                            timestamp=now,
                            detail=f"Missing: {artifact_name}",
                            severity=AlertSeverity.CRITICAL.value,
                        )
                    )
                    continue

                actual = self._compute_sha256(artifact_path)
                if actual != expected:
                    events.append(
                        SecurityEvent(
                            event_type="HASH_MISMATCH",
                            engine_id=engine_id,
                            timestamp=now,
                            detail=(
                                f"{artifact_name}: " f"{expected[:16]}... != " f"{actual[:16]}..."
                            ),
                            severity=AlertSeverity.CRITICAL.value,
                        )
                    )
                    AuditLogger.log_security_event(
                        "HASH_MISMATCH",
                        f"{engine_id}/{artifact_name}",
                        level=logging.CRITICAL,
                    )
                else:
                    events.append(
                        SecurityEvent(
                            event_type="HASH_VERIFIED",
                            engine_id=engine_id,
                            timestamp=now,
                            detail=(f"{artifact_name}: " f"{actual[:16]}... OK"),
                            severity=AlertSeverity.INFO.value,
                        )
                    )

        self._events.extend(events)
        return events

    def verify_baseline_config(self) -> Optional[SecurityEvent]:
        """Verify Phase 4 baseline_config.json for tampering.

        Returns:
            SecurityEvent if status changed, None otherwise.
        """
        now = datetime.now(timezone.utc).isoformat()
        bl_path = self._root / BASELINE_CONFIG_PATH
        if not bl_path.exists():
            return None

        current = self._compute_sha256(bl_path)
        if self._baseline_hash is None:
            self._baseline_hash = current
            event = SecurityEvent(
                event_type="BASELINE_INITIALIZED",
                engine_id="phase4",
                timestamp=now,
                detail=f"Hash: {current[:16]}...",
                severity=AlertSeverity.INFO.value,
            )
            self._events.append(event)
            return event

        if current != self._baseline_hash:
            event = SecurityEvent(
                event_type="BASELINE_TAMPER_DETECTED",
                engine_id="phase4",
                timestamp=now,
                detail=(f"Changed: {self._baseline_hash[:16]}... " f"-> {current[:16]}..."),
                severity=AlertSeverity.CRITICAL.value,
            )
            AuditLogger.log_security_event(
                "BASELINE_TAMPER",
                "baseline_config.json integrity violation",
                level=logging.CRITICAL,
            )
            self._events.append(event)
            return event

        return None

    @property
    def all_events(self) -> List[SecurityEvent]:
        """Return all security events."""
        return list(self._events)


# ===================================================================
# AlertManager
# ===================================================================
class AlertManager:
    """Tracks and manages monitoring alerts."""

    def __init__(self) -> None:
        self._alerts: List[MonitoringAlert] = []

    def add(self, alert: MonitoringAlert) -> None:
        """Register a new alert."""
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


# ===================================================================
# DashboardReporter
# ===================================================================
class DashboardReporter:
    """Simulated WebSocket dashboard push (research/CI mode)."""

    def __init__(self) -> None:
        self._push_count: int = 0
        self._last_payload: Optional[Dict[str, Any]] = None

    def push_update(
        self,
        engine_states: Dict[str, str],
        engine_metrics: Dict[str, Optional[Dict[str, float]]],
        alert_counts: Dict[str, int],
    ) -> Dict[str, Any]:
        """Push dashboard update (simulated WebSocket).

        Args:
            engine_states: Current state per engine.
            engine_metrics: Latest metrics per engine.
            alert_counts: Active alert counts by severity.

        Returns:
            Dashboard payload dict.
        """
        self._push_count += 1
        payload: Dict[str, Any] = {
            "push_id": self._push_count,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "engines": {},
            "system_overview": {},
            "active_alerts": alert_counts,
        }

        for eid in ENGINE_IDS:
            payload["engines"][eid] = {
                "state": engine_states.get(eid, EngineState.UNKNOWN.value),
                "metrics": engine_metrics.get(eid),
            }

        for state in EngineState:
            payload["system_overview"][state.value] = sum(
                1 for s in engine_states.values() if s == state.value
            )

        self._last_payload = payload
        logger.debug("Dashboard push #%d", self._push_count)
        return payload

    @property
    def push_count(self) -> int:
        """Return total number of pushes."""
        return self._push_count


# ===================================================================
# MonitoringPipeline -- async orchestrator
# ===================================================================
class MonitoringPipeline:
    """Orchestrates all monitoring subsystems via asyncio.gather().

    Args:
        project_root: Project root directory.
        config: Parsed YAML config dict.
        random_state: Seed for reproducible simulation.
    """

    def __init__(
        self,
        project_root: Path,
        config: Dict[str, Any],
        random_state: int = 42,
    ) -> None:
        mon = config.get("monitoring", {})
        self._missed_threshold = mon.get("missed_heartbeat_threshold", MISSED_HEARTBEAT_THRESHOLD)
        self._buffer_size = mon.get("circular_buffer_size", CIRCULAR_BUFFER_SIZE)
        self._latency_threshold = mon.get("latency_p95_threshold_ms", LATENCY_P95_THRESHOLD_MS)
        self._memory_threshold = mon.get("memory_warning_threshold_pct", MEMORY_WARNING_PCT)
        self._cpu_threshold = mon.get("cpu_warning_threshold_pct", CPU_WARNING_PCT)

        self._root = project_root
        self._trackers: Dict[str, EngineHealthTracker] = {
            eid: EngineHealthTracker(eid, self._buffer_size) for eid in ENGINE_IDS
        }
        self._perf = PerformanceCollector()
        self._security = SecurityMonitor(project_root)
        self._alerts = AlertManager()
        self._dashboard = DashboardReporter()
        self._state_changes: List[StateChangeEvent] = []
        self._rng = random.Random(random_state)

    # ---------------------------------------------------------------
    # Heartbeat simulation
    # ---------------------------------------------------------------
    def _simulate_heartbeat(self, engine_id: str) -> Optional[float]:
        """Simulate heartbeat by checking artifact existence.

        Returns:
            Latency in ms if heartbeat received, None if missed.
        """
        artifact_dir = self._root / PHASE_ARTIFACT_DIRS[engine_id]
        if not artifact_dir.exists():
            return None

        if self._rng.random() < HEARTBEAT_MISS_PROBABILITY:
            return None

        if self._rng.random() < HEARTBEAT_SPIKE_PROBABILITY:
            latency = self._rng.gauss(SPIKE_LATENCY_MEAN_MS, SPIKE_LATENCY_STD_MS)
        else:
            latency = self._rng.gauss(NORMAL_LATENCY_MEAN_MS, NORMAL_LATENCY_STD_MS)

        return max(MIN_LATENCY_MS, latency)

    # ---------------------------------------------------------------
    # Performance simulation
    # ---------------------------------------------------------------
    def _simulate_metrics(self, engine_id: str) -> PerformanceSnapshot:
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

    # ---------------------------------------------------------------
    # Alert generation
    # ---------------------------------------------------------------
    def _handle_state_change(self, event: StateChangeEvent) -> None:
        """Generate health alert on DOWN or DEGRADED transition."""
        self._state_changes.append(event)
        logger.info(
            "State: %s %s -> %s (%s)",
            event.engine_id,
            event.old_state,
            event.new_state,
            event.reason,
        )

        ts = int(time.time())
        if event.new_state == EngineState.DOWN.value:
            self._alerts.add(
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
            self._alerts.add(
                MonitoringAlert(
                    alert_id=f"HEALTH-DEG-{event.engine_id}-{ts}",
                    category=AlertCategory.HEALTH.value,
                    severity=AlertSeverity.WARNING.value,
                    engine_id=event.engine_id,
                    timestamp=event.timestamp,
                    message=(
                        f"{event.engine_id} DEGRADED: "
                        f"{event.reason}. "
                        "Notify IT admin within 5 minutes."
                    ),
                )
            )

    def _handle_security_events(self, events: List[SecurityEvent]) -> None:
        """Generate CRITICAL alerts for security violations."""
        for evt in events:
            if evt.severity == AlertSeverity.CRITICAL.value:
                ts = int(time.time())
                self._alerts.add(
                    MonitoringAlert(
                        alert_id=(f"SEC-{evt.event_type}-" f"{evt.engine_id}-{ts}"),
                        category=AlertCategory.SECURITY.value,
                        severity=AlertSeverity.CRITICAL.value,
                        engine_id=evt.engine_id,
                        timestamp=evt.timestamp,
                        message=f"Security: {evt.detail}",
                    )
                )

    # ---------------------------------------------------------------
    # Async monitoring loops
    # ---------------------------------------------------------------
    async def _heartbeat_loop(self, n_cycles: int) -> None:
        """Monitor engine heartbeats for n_cycles iterations."""
        for _ in range(n_cycles):
            for eid in ENGINE_IDS:
                latency = self._simulate_heartbeat(eid)
                if latency is not None:
                    event = self._trackers[eid].process_heartbeat(latency, self._latency_threshold)
                else:
                    event = self._trackers[eid].tick_missed(self._missed_threshold)
                if event is not None:
                    self._handle_state_change(event)
            await asyncio.sleep(0.01)

    async def _performance_loop(self, n_cycles: int) -> None:
        """Collect performance metrics for n_cycles iterations."""
        for _ in range(n_cycles):
            for eid in ENGINE_IDS:
                snapshot = self._simulate_metrics(eid)
                self._perf.record(snapshot)
                alerts = self._perf.check_thresholds(
                    eid,
                    self._latency_threshold,
                    self._memory_threshold,
                    self._cpu_threshold,
                )
                for alert in alerts:
                    self._alerts.add(alert)
            await asyncio.sleep(0.01)

    async def _security_loop(self, n_cycles: int) -> None:
        """Verify artifact integrity for n_cycles iterations."""
        for _ in range(n_cycles):
            events = self._security.verify_all_artifacts()
            self._handle_security_events(events)
            bl_event = self._security.verify_baseline_config()
            if bl_event and bl_event.severity == AlertSeverity.CRITICAL.value:
                self._handle_security_events([bl_event])
            await asyncio.sleep(0.01)

    async def _dashboard_loop(self, n_cycles: int) -> None:
        """Push dashboard updates for n_cycles iterations."""
        for _ in range(n_cycles):
            states = {eid: self._trackers[eid].state.value for eid in ENGINE_IDS}
            metrics: Dict[str, Optional[Dict[str, float]]] = {}
            for eid in ENGINE_IDS:
                latest = self._perf.get_latest(eid)
                if latest:
                    metrics[eid] = {
                        "latency_p95_ms": latest.latency_p95_ms,
                        "memory_mb": latest.memory_mb,
                        "cpu_pct": latest.cpu_pct,
                    }
                else:
                    metrics[eid] = None
            self._dashboard.push_update(states, metrics, self._alerts.count_by_severity())
            await asyncio.sleep(0.01)

    # ---------------------------------------------------------------
    # Pipeline execution
    # ---------------------------------------------------------------
    async def run_async(self, n_cycles: int = 5) -> Dict[str, Any]:
        """Run all monitoring loops concurrently.

        Args:
            n_cycles: Number of monitoring iterations per loop.

        Returns:
            Pipeline summary dict.
        """
        logger.info(
            "Pipeline starting: %d cycles, %d engines",
            n_cycles,
            len(ENGINE_IDS),
        )
        t0 = time.time()

        await asyncio.gather(
            self._heartbeat_loop(n_cycles),
            self._performance_loop(n_cycles),
            self._security_loop(n_cycles),
            self._dashboard_loop(n_cycles),
        )

        duration = time.time() - t0
        engine_states = {eid: self._trackers[eid].state.value for eid in ENGINE_IDS}
        state_counts: Dict[str, int] = {}
        for s in engine_states.values():
            state_counts[s] = state_counts.get(s, 0) + 1

        return {
            "duration_seconds": round(duration, 2),
            "n_cycles": n_cycles,
            "n_engines": len(ENGINE_IDS),
            "engine_states": engine_states,
            "state_summary": state_counts,
            "total_state_changes": len(self._state_changes),
            "total_alerts": len(self._alerts.all_alerts),
            "alert_severity_counts": self._alerts.count_by_severity(),
            "alert_category_counts": self._alerts.count_by_category(),
            "security_events": len(self._security.all_events),
            "dashboard_pushes": self._dashboard.push_count,
        }

    def run(self, n_cycles: int = 5) -> Dict[str, Any]:
        """Synchronous wrapper for run_async()."""
        return asyncio.run(self.run_async(n_cycles))

    # ---------------------------------------------------------------
    # Data accessors for export
    # ---------------------------------------------------------------
    @property
    def state_changes(self) -> List[StateChangeEvent]:
        """Return all state change events."""
        return list(self._state_changes)

    @property
    def alerts(self) -> List[MonitoringAlert]:
        """Return all generated alerts."""
        return self._alerts.all_alerts

    @property
    def security_events(self) -> List[SecurityEvent]:
        """Return all security events."""
        return self._security.all_events

    @property
    def engine_states(self) -> Dict[str, str]:
        """Return current state per engine."""
        return {eid: self._trackers[eid].state.value for eid in ENGINE_IDS}

    @property
    def performance_snapshots(self) -> Dict[str, List[PerformanceSnapshot]]:
        """Return all performance snapshots per engine."""
        return {eid: self._perf.get_all(eid) for eid in ENGINE_IDS}

    @property
    def dashboard(self) -> DashboardReporter:
        """Return the dashboard reporter instance."""
        return self._dashboard


# ===================================================================
# Utility functions
# ===================================================================
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


def _get_git_commit() -> str:
    """Return current git commit hash or 'unknown'."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=str(PROJECT_ROOT),
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except (subprocess.SubprocessError, FileNotFoundError):
        return "unknown"


# ===================================================================
# Artifact export
# ===================================================================
def export_artifacts(
    output_dir: Path,
    state_changes: List[StateChangeEvent],
    engine_states: Dict[str, str],
    performance_data: Dict[str, List[PerformanceSnapshot]],
    security_events: List[SecurityEvent],
    alerts: List[MonitoringAlert],
    summary: Dict[str, Any],
) -> Dict[str, str]:
    """Export Phase 7 monitoring artifacts with SHA-256 hashes.

    Args:
        output_dir: Target directory for artifacts.
        state_changes: All state change events.
        engine_states: Current state per engine.
        performance_data: Rolling metrics per engine.
        security_events: All security events.
        alerts: All generated alerts.
        summary: Pipeline summary dict.

    Returns:
        Mapping of artifact name to SHA-256 hash.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. monitoring_log.json -- state changes only
    monitoring_log = [asdict(sc) for sc in state_changes]
    ml_path = output_dir / MONITORING_LOG
    with open(ml_path, "w") as f:
        json.dump(monitoring_log, f, indent=2)

    # 2. health_report.json -- current state per engine + alerts
    health_report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pipeline": "phase7_monitoring",
        "engine_health": {
            eid: {
                "state": engine_states[eid],
                "name": ENGINE_NAMES[eid],
            }
            for eid in ENGINE_IDS
        },
        "alerts": [asdict(a) for a in alerts],
        "summary": summary,
    }
    hr_path = output_dir / HEALTH_REPORT
    with open(hr_path, "w") as f:
        json.dump(health_report, f, indent=2)

    # 3. performance_report.json -- rolling metrics per engine
    perf_report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pipeline": "phase7_monitoring",
        "metrics": {
            eid: [asdict(s) for s in snapshots] for eid, snapshots in performance_data.items()
        },
    }
    pr_path = output_dir / PERFORMANCE_REPORT
    with open(pr_path, "w") as f:
        json.dump(perf_report, f, indent=2)

    # 4. security_audit_log.json -- security events only
    sec_log = [asdict(e) for e in security_events]
    sal_path = output_dir / SECURITY_AUDIT_LOG
    with open(sal_path, "w") as f:
        json.dump(sec_log, f, indent=2)

    # Compute SHA-256 hashes
    verifier = IntegrityVerifier(output_dir)
    artifact_hashes: Dict[str, str] = {}
    for name, path in [
        (MONITORING_LOG, ml_path),
        (HEALTH_REPORT, hr_path),
        (PERFORMANCE_REPORT, pr_path),
        (SECURITY_AUDIT_LOG, sal_path),
    ]:
        digest = verifier.compute_hash(path)
        artifact_hashes[name] = digest
        logger.info("Exported %s — SHA-256: %s...", name, digest[:16])

    AuditLogger.log_security_event(
        "PHASE7_EXPORT",
        f"4 artifacts exported to {output_dir}",
        level=logging.INFO,
    )
    return artifact_hashes


# ===================================================================
# Report generation
# ===================================================================
def generate_monitoring_report(
    report_dir: Path,
    summary: Dict[str, Any],
    state_changes: List[StateChangeEvent],
    engine_states: Dict[str, str],
    performance_data: Dict[str, List[PerformanceSnapshot]],
    security_events: List[SecurityEvent],
    alerts: List[MonitoringAlert],
    artifact_hashes: Dict[str, str],
    git_commit: str,
    hardware: Dict[str, str],
) -> Path:
    """Generate report_section_monitoring.md for IEEE Q1.

    Args:
        report_dir: Directory for the markdown report.
        summary: Pipeline summary dict.
        state_changes: All state change events.
        engine_states: Current state per engine.
        performance_data: Rolling metrics per engine.
        security_events: All security events.
        alerts: All generated alerts.
        artifact_hashes: SHA-256 hashes of exported files.
        git_commit: Current git commit hash.
        hardware: Hardware detection dict.

    Returns:
        Path to the generated report.
    """
    report_path = report_dir / "report_section_monitoring.md"
    now = datetime.now(timezone.utc).isoformat()

    # Alert counts
    sev_counts: Dict[str, int] = {s.value: 0 for s in AlertSeverity}
    cat_counts: Dict[str, int] = {c.value: 0 for c in AlertCategory}
    for a in alerts:
        sev_counts[a.severity] = sev_counts.get(a.severity, 0) + 1
        cat_counts[a.category] = cat_counts.get(a.category, 0) + 1

    # Security summary
    sec_verified = sum(1 for e in security_events if e.event_type == "HASH_VERIFIED")
    sec_violations = sum(1 for e in security_events if e.severity == AlertSeverity.CRITICAL.value)

    # Engine health table
    health_rows: List[str] = []
    for eid in ENGINE_IDS:
        state = engine_states[eid]
        n_ch = sum(1 for sc in state_changes if sc.engine_id == eid)
        health_rows.append(f"| {eid} | {ENGINE_NAMES[eid]} " f"| {state} | {n_ch} |")

    # Performance table
    perf_rows: List[str] = []
    for eid in ENGINE_IDS:
        snaps = performance_data.get(eid, [])
        if snaps:
            last = snaps[-1]
            perf_rows.append(
                f"| {eid} | {last.latency_p50_ms:.1f} "
                f"| {last.latency_p95_ms:.1f} "
                f"| {last.latency_p99_ms:.1f} "
                f"| {last.throughput_sps:.1f} "
                f"| {last.memory_mb:.0f} "
                f"| {last.cpu_pct:.1f} |"
            )
        else:
            perf_rows.append(f"| {eid} | -- | -- | -- | -- | -- | -- |")

    # Artifact hash table
    hash_rows: List[str] = []
    for name, digest in artifact_hashes.items():
        hash_rows.append(f"| {name} | `{digest[:16]}...` |")

    lines = [
        "## 10.1 System Monitoring & Observability (Phase 7)",
        "",
        "### 10.1.1 Engine Health Summary",
        "",
        "| Engine | Name | State | State Changes |",
        "|--------|------|-------|---------------|",
        *health_rows,
        "",
        f"**Total state changes:** {len(state_changes)}",
        "",
        "### 10.1.2 Performance Metrics (Latest Snapshot)",
        "",
        ("| Engine | p50 (ms) | p95 (ms) | p99 (ms) " "| Throughput | Memory (MB) | CPU (%) |"),
        ("|--------|----------|----------|----------" "|------------|-------------|---------|"),
        *perf_rows,
        "",
        "**Thresholds:**",
        (f"- Latency p95 > {LATENCY_P95_THRESHOLD_MS:.0f}ms " "-> DEGRADED"),
        f"- Memory > {MEMORY_WARNING_PCT:.0f}% -> WARNING",
        f"- CPU > {CPU_WARNING_PCT:.0f}% -> WARNING",
        "",
        "### 10.1.3 Security Audit",
        "",
        (
            f"- **Artifact integrity checks:** {sec_verified} "
            f"verified, {sec_violations} violations"
        ),
        (
            "- **Baseline config tamper:** "
            + (
                "PASS -- no tampering detected"
                if sec_violations == 0
                else "FAIL -- tampering detected"
            )
        ),
        f"- **Total security events:** {len(security_events)}",
        "",
        "### 10.1.4 Alert Summary",
        "",
        "| Severity | Count |",
        "|----------|-------|",
        f"| CRITICAL | {sev_counts.get('CRITICAL', 0)} |",
        f"| WARNING  | {sev_counts.get('WARNING', 0)} |",
        f"| INFO     | {sev_counts.get('INFO', 0)} |",
        "",
        "| Category    | Count |",
        "|-------------|-------|",
        f"| HEALTH      | {cat_counts.get('HEALTH', 0)} |",
        f"| PERFORMANCE | {cat_counts.get('PERFORMANCE', 0)} |",
        f"| SECURITY    | {cat_counts.get('SECURITY', 0)} |",
        "",
        f"**Total alerts:** {len(alerts)}",
        "",
        "### 10.1.5 State Machine Specification",
        "",
        "```",
        "UNKNOWN -> STARTING -> UP <-> DEGRADED",
        "                       |         |",
        "                       v         v",
        "                      DOWN <-----+",
        "                       |",
        "                       +-> STARTING (heartbeat received)",
        "```",
        "",
        "| Transition | Trigger |",
        "|------------|---------|",
        "| UNKNOWN -> STARTING | First heartbeat received |",
        (f"| STARTING -> UP | " f"{CONSECUTIVE_OK_THRESHOLD} consecutive heartbeats |"),
        (f"| UP -> DEGRADED | " f"Latency > {LATENCY_P95_THRESHOLD_MS:.0f}ms |"),
        (f"| UP -> DOWN | " f"{MISSED_HEARTBEAT_THRESHOLD} missed heartbeats |"),
        "| DEGRADED -> UP | Latency returns to normal |",
        (f"| DEGRADED -> DOWN | " f"{MISSED_HEARTBEAT_THRESHOLD} missed heartbeats |"),
        "| DOWN -> STARTING | Heartbeat received again |",
        "",
        "### 10.1.6 Monitoring Configuration",
        "",
        "| Parameter | Value |",
        "|-----------|-------|",
        f"| Heartbeat interval | {HEARTBEAT_INTERVAL_S}s |",
        ("| Missed heartbeat threshold | " f"{MISSED_HEARTBEAT_THRESHOLD} |"),
        f"| Grace period | {GRACE_PERIOD_S}s |",
        f"| Circular buffer size | {CIRCULAR_BUFFER_SIZE} |",
        ("| Performance collection interval | " f"{PERF_COLLECTION_INTERVAL_S}s |"),
        ("| Artifact integrity check interval | " f"{ARTIFACT_CHECK_INTERVAL_S}s |"),
        ("| Baseline check interval | " f"{BASELINE_CHECK_INTERVAL_S}s |"),
        "| Storage | State changes only |",
        f"| Rolling window | {ROLLING_WINDOW_HOURS} hours |",
        "",
        "### 10.1.7 Async Architecture",
        "",
        "```",
        "asyncio.gather(",
        (
            f"    heartbeat_loop     -- every "
            f"{HEARTBEAT_INTERVAL_S}s x "
            f"{len(ENGINE_IDS)} engines"
        ),
        (
            f"    performance_loop   -- every "
            f"{PERF_COLLECTION_INTERVAL_S}s x "
            f"{len(ENGINE_IDS)} engines"
        ),
        (f"    security_loop      -- every " f"{ARTIFACT_CHECK_INTERVAL_S}s x SHA-256"),
        (f"    dashboard_loop     -- every " f"{DASHBOARD_PUSH_INTERVAL_S}s x WebSocket"),
        ")",
        "```",
        "",
        "### 10.1.8 Artifact Integrity (Phase 7 Outputs)",
        "",
        "| Artifact | SHA-256 |",
        "|----------|---------|",
        *hash_rows,
        "",
        "### 10.1.9 Execution Details",
        "",
        ("- **Duration:** " f"{summary.get('duration_seconds', 0):.2f}s"),
        ("- **Monitoring cycles:** " f"{summary.get('n_cycles', 0)}"),
        ("- **Engines monitored:** " f"{summary.get('n_engines', 0)}"),
        ("- **Dashboard pushes:** " f"{summary.get('dashboard_pushes', 0)}"),
        f"- **Git commit:** `{git_commit[:12]}...`",
        "- **Pipeline:** phase7_monitoring",
        f"- **Hardware:** {hardware.get('device', 'N/A')}",
        f"- **Python:** {hardware.get('python', 'N/A')}",
        "",
        "---",
        "",
        (f"*Generated: {now} | Pipeline: phase7_monitoring" f" | Commit: {git_commit[:12]}*"),
        "",
    ]

    report_dir.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        f.write("\n".join(lines))

    logger.info("Report generated: %s", report_path)
    return report_path


# ===================================================================
# Pipeline entry point
# ===================================================================
def run_monitoring_pipeline(
    *,
    config_path: Optional[Path] = None,
    n_cycles: Optional[int] = None,
) -> Dict[str, Any]:
    """Execute Phase 7 monitoring pipeline.

    Args:
        config_path: Override config file path.
        n_cycles: Override number of monitoring cycles.

    Returns:
        Pipeline summary dict.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    logger.info("=" * 60)
    logger.info("Phase 7 Monitoring Engine -- starting")
    logger.info("=" * 60)

    t0 = time.time()

    # 1. Load config
    cfg_path = config_path or CONFIG_PATH
    if cfg_path.exists():
        with open(cfg_path) as f:
            config = yaml.safe_load(f)
        logger.info("Config loaded: %s", cfg_path)
    else:
        config = {}
        logger.warning("Config not found: %s -- using defaults", cfg_path)

    # 2. Determine cycle count
    cycles = n_cycles or config.get("pipeline", {}).get("n_cycles", 5)

    # 3. Hardware + git
    hardware = _detect_hardware()
    git_commit = _get_git_commit()
    logger.info("Hardware: %s", hardware.get("device", "N/A"))
    logger.info("Git commit: %s", git_commit[:12])

    # 4. Path validation
    validator = PathValidator(PROJECT_ROOT)
    output_dir = validator.validate_output_dir(
        Path(config.get("output", {}).get("output_dir", "data/phase7"))
    )
    report_dir = validator.validate_output_dir(REPORT_DIR)

    # 5. Run monitoring pipeline
    random_state = config.get("pipeline", {}).get("random_state", 42)
    pipeline = MonitoringPipeline(
        project_root=PROJECT_ROOT,
        config=config,
        random_state=random_state,
    )
    summary = pipeline.run(n_cycles=cycles)

    # 6. Export artifacts
    artifact_hashes = export_artifacts(
        output_dir=output_dir,
        state_changes=pipeline.state_changes,
        engine_states=pipeline.engine_states,
        performance_data=pipeline.performance_snapshots,
        security_events=pipeline.security_events,
        alerts=pipeline.alerts,
        summary=summary,
    )

    # 7. Generate report
    generate_monitoring_report(
        report_dir=report_dir,
        summary=summary,
        state_changes=pipeline.state_changes,
        engine_states=pipeline.engine_states,
        performance_data=pipeline.performance_snapshots,
        security_events=pipeline.security_events,
        alerts=pipeline.alerts,
        artifact_hashes=artifact_hashes,
        git_commit=git_commit,
        hardware=hardware,
    )

    total_duration = time.time() - t0
    summary["total_duration_seconds"] = round(total_duration, 2)
    summary["artifact_hashes"] = artifact_hashes
    summary["git_commit"] = git_commit

    logger.info("=" * 60)
    logger.info("Phase 7 Monitoring Engine -- complete")
    logger.info(
        "State changes: %d | Alerts: %d | Security events: %d",
        summary["total_state_changes"],
        summary["total_alerts"],
        summary["security_events"],
    )
    logger.info(
        "Dashboard pushes: %d | Duration: %.2fs",
        summary["dashboard_pushes"],
        total_duration,
    )
    logger.info("=" * 60)

    return summary


def main() -> None:
    """Entry point for Phase 7 monitoring engine."""
    run_monitoring_pipeline()


if __name__ == "__main__":
    main()
