"""Five-state machine for engine health tracking.

States: UNKNOWN -> STARTING -> UP <-> DEGRADED -> DOWN
Transitions defined in config, guarded by asyncio.Lock for thread safety.
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Deque, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


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
# Transition table
# ===================================================================
TransitionKey = Tuple[EngineState, str]
TransitionTable = Dict[TransitionKey, EngineState]


def build_transition_table(rules: List[Dict[str, str]]) -> TransitionTable:
    """Build transition table from config rules.

    Args:
        rules: List of {"from_state": ..., "trigger": ..., "to_state": ...}

    Returns:
        Mapping of (current_state, trigger) -> new_state.
    """
    table: TransitionTable = {}
    for rule in rules:
        from_s = EngineState(rule["from_state"])
        to_s = EngineState(rule["to_state"])
        table[(from_s, rule["trigger"])] = to_s
    return table


DEFAULT_TRANSITIONS: TransitionTable = {
    (EngineState.UNKNOWN, "heartbeat_received"): EngineState.STARTING,
    (EngineState.STARTING, "consecutive_ok"): EngineState.UP,
    (EngineState.UP, "latency_exceeded"): EngineState.DEGRADED,
    (EngineState.UP, "missed_threshold"): EngineState.DOWN,
    (EngineState.DEGRADED, "latency_recovered"): EngineState.UP,
    (EngineState.DEGRADED, "missed_threshold"): EngineState.DOWN,
    (EngineState.DOWN, "heartbeat_received"): EngineState.STARTING,
}


# ===================================================================
# StateMachine
# ===================================================================
class StateMachine:
    """Thread-safe state machine for a single engine.

    Uses asyncio.Lock to guard state transitions.
    Transition table is config-driven (not hardcoded).

    Args:
        engine_id: Unique engine identifier.
        transitions: Config-driven transition table. Defaults to DEFAULT_TRANSITIONS.
        buffer_size: Maximum state change history entries.
        consecutive_ok_threshold: Heartbeats needed to transition STARTING -> UP.
    """

    def __init__(
        self,
        engine_id: str,
        transitions: Optional[TransitionTable] = None,
        buffer_size: int = 1000,
        consecutive_ok_threshold: int = 3,
    ) -> None:
        self._engine_id = engine_id
        self._state = EngineState.UNKNOWN
        self._transitions = transitions or DEFAULT_TRANSITIONS
        self._consecutive_ok: int = 0
        self._missed_count: int = 0
        self._consecutive_ok_threshold = consecutive_ok_threshold
        self._history: Deque[StateChangeEvent] = deque(maxlen=buffer_size)
        self._lock = asyncio.Lock()

    @property
    def state(self) -> EngineState:
        """Return current engine state."""
        return self._state

    @property
    def engine_id(self) -> str:
        """Return engine identifier."""
        return self._engine_id

    @property
    def history(self) -> Deque[StateChangeEvent]:
        """Return state change history (circular buffer)."""
        return self._history

    async def _transition(self, trigger: str, reason: str) -> Optional[StateChangeEvent]:
        """Attempt a state transition under asyncio.Lock.

        Returns:
            StateChangeEvent if transition occurred, None if not allowed.
        """
        async with self._lock:
            key = (self._state, trigger)
            new_state = self._transitions.get(key)
            if new_state is None:
                return None
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

    async def process_heartbeat(
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
            return await self._transition(
                "heartbeat_received",
                f"first heartbeat, latency={latency_ms:.1f}ms",
            )

        if self._state == EngineState.STARTING:
            self._consecutive_ok += 1
            if self._consecutive_ok >= self._consecutive_ok_threshold:
                return await self._transition(
                    "consecutive_ok",
                    f"{self._consecutive_ok_threshold} consecutive OK",
                )
            return None

        if self._state == EngineState.UP:
            if latency_ms > threshold_ms:
                return await self._transition(
                    "latency_exceeded",
                    f"latency {latency_ms:.1f}ms > {threshold_ms:.0f}ms",
                )
            return None

        if self._state == EngineState.DEGRADED:
            if latency_ms <= threshold_ms:
                return await self._transition(
                    "latency_recovered",
                    f"latency recovered to {latency_ms:.1f}ms",
                )
            return None

        if self._state == EngineState.DOWN:
            self._consecutive_ok = 1
            return await self._transition(
                "heartbeat_received",
                f"heartbeat again, latency={latency_ms:.1f}ms",
            )

        return None

    async def tick_missed(self, threshold: int) -> Optional[StateChangeEvent]:
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
            return await self._transition(
                "missed_threshold",
                f"{self._missed_count} consecutive misses",
            )
        return None
