"""Pydantic-validated configuration for the Phase 7 monitoring pipeline.

Loads from ``config/phase7_monitoring_config.yaml`` via ``Phase7Config.from_yaml()``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, field_validator


class EngineEntry(BaseModel):
    """Engine registration entry for monitoring (Open/Closed)."""

    id: str
    heartbeat_topic: str
    artifact_dir: Optional[str] = None
    metadata_path: Optional[str] = None
    name: Optional[str] = None


class TransitionRule(BaseModel):
    """State machine transition rule (config-driven)."""

    from_state: str
    trigger: str
    to_state: str


class Phase7Config(BaseModel):
    """Validated configuration for Phase 7 monitoring pipeline."""

    # ── Monitoring parameters ─────────────────────────────────────
    heartbeat_interval_seconds: int = 5
    missed_heartbeat_threshold: int = 5
    grace_period_seconds: int = 25
    storage: str = "state_changes_only"
    circular_buffer_size: int = 1000
    performance_collection_interval: int = 30
    artifact_integrity_check_interval: int = 60
    latency_p95_threshold_ms: float = 100.0
    memory_warning_threshold_pct: float = 80.0
    cpu_warning_threshold_pct: float = 90.0

    # ── Security ──────────────────────────────────────────────────
    baseline_config_path: str = "data/phase4/baseline_config.json"
    baseline_check_interval_seconds: int = 30

    # ── Output ────────────────────────────────────────────────────
    output_dir: Path = Path("data/phase7")

    # ── Pipeline ──────────────────────────────────────────────────
    n_cycles: int = 5
    random_state: int = 42

    # ── Engine registry (Open/Closed — add without code change) ──
    engines: List[EngineEntry] = []

    # ── State machine transitions (config-driven) ─────────────────
    transitions: List[TransitionRule] = []

    # ── Simulation parameters ─────────────────────────────────────
    heartbeat_miss_probability: float = 0.05
    heartbeat_spike_probability: float = 0.05
    normal_latency_mean_ms: float = 25.0
    normal_latency_std_ms: float = 10.0
    spike_latency_mean_ms: float = 150.0
    spike_latency_std_ms: float = 30.0
    consecutive_ok_threshold: int = 3
    rolling_window_hours: int = 24
    reference_memory_mb: float = 4096.0
    dashboard_push_interval_seconds: int = 5

    model_config: Dict[str, Any] = {"arbitrary_types_allowed": True}

    # ── Validators ────────────────────────────────────────────────

    @field_validator("heartbeat_interval_seconds")
    @classmethod
    def _heartbeat_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"heartbeat_interval_seconds must be >= 1, got {v}")
        return v

    @field_validator("missed_heartbeat_threshold")
    @classmethod
    def _missed_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"missed_heartbeat_threshold must be >= 1, got {v}")
        return v

    @field_validator("circular_buffer_size")
    @classmethod
    def _buffer_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"circular_buffer_size must be >= 1, got {v}")
        return v

    @field_validator("n_cycles")
    @classmethod
    def _cycles_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"n_cycles must be >= 1, got {v}")
        return v

    @field_validator("latency_p95_threshold_ms")
    @classmethod
    def _latency_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"latency_p95_threshold_ms must be > 0, got {v}")
        return v

    # ── Default transitions ───────────────────────────────────────

    @classmethod
    def _default_transitions(cls) -> List[TransitionRule]:
        """Default 5-state machine transition rules."""
        return [
            TransitionRule(
                from_state="UNKNOWN",
                trigger="heartbeat_received",
                to_state="STARTING",
            ),
            TransitionRule(
                from_state="STARTING",
                trigger="consecutive_ok",
                to_state="UP",
            ),
            TransitionRule(
                from_state="UP",
                trigger="latency_exceeded",
                to_state="DEGRADED",
            ),
            TransitionRule(
                from_state="UP",
                trigger="missed_threshold",
                to_state="DOWN",
            ),
            TransitionRule(
                from_state="DEGRADED",
                trigger="latency_recovered",
                to_state="UP",
            ),
            TransitionRule(
                from_state="DEGRADED",
                trigger="missed_threshold",
                to_state="DOWN",
            ),
            TransitionRule(
                from_state="DOWN",
                trigger="heartbeat_received",
                to_state="STARTING",
            ),
        ]

    # ── YAML loader ───────────────────────────────────────────────

    @classmethod
    def from_yaml(cls, path: Path) -> Phase7Config:
        """Load and validate configuration from a YAML file."""
        raw: Dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))

        mon = raw.get("monitoring", {})
        sec = raw.get("security", {})
        out = raw.get("output", {})
        pipe = raw.get("pipeline", {})

        engines_raw = raw.get("engines", [])
        engines = [EngineEntry(**e) for e in engines_raw]

        transitions_raw = raw.get("transitions", [])
        transitions = (
            [TransitionRule(**t) for t in transitions_raw]
            if transitions_raw
            else cls._default_transitions()
        )

        return cls(
            heartbeat_interval_seconds=mon.get("heartbeat_interval_seconds", 5),
            missed_heartbeat_threshold=mon.get("missed_heartbeat_threshold", 5),
            grace_period_seconds=mon.get("grace_period_seconds", 25),
            storage=mon.get("storage", "state_changes_only"),
            circular_buffer_size=mon.get("circular_buffer_size", 1000),
            performance_collection_interval=mon.get("performance_collection_interval", 30),
            artifact_integrity_check_interval=mon.get("artifact_integrity_check_interval", 60),
            latency_p95_threshold_ms=mon.get("latency_p95_threshold_ms", 100.0),
            memory_warning_threshold_pct=mon.get("memory_warning_threshold_pct", 80.0),
            cpu_warning_threshold_pct=mon.get("cpu_warning_threshold_pct", 90.0),
            baseline_config_path=sec.get(
                "baseline_config_path", "data/phase4/baseline_config.json"
            ),
            baseline_check_interval_seconds=sec.get("baseline_check_interval_seconds", 30),
            output_dir=Path(out.get("output_dir", "data/phase7")),
            n_cycles=pipe.get("n_cycles", 5),
            random_state=pipe.get("random_state", 42),
            engines=engines,
            transitions=transitions,
        )
