"""Tests for StateMachine, enums, and dataclasses."""

from __future__ import annotations

import asyncio

import pytest

from src.phase7_monitoring_engine.phase7.state_machine import (
    DEFAULT_TRANSITIONS,
    EngineState,
    MonitoringAlert,
    PerformanceSnapshot,
    SecurityEvent,
    StateChangeEvent,
    StateMachine,
    build_transition_table,
)


class TestEngineState:
    """Test EngineState enum."""

    def test_all_states(self):
        assert len(EngineState) == 5
        assert EngineState.UNKNOWN.value == "UNKNOWN"
        assert EngineState.DOWN.value == "DOWN"

    def test_string_enum(self):
        assert str(EngineState.UP) == "EngineState.UP"
        assert EngineState.UP == "UP"


class TestBuildTransitionTable:
    """Test config-driven transition table."""

    def test_build_from_rules(self):
        rules = [
            {"from_state": "UNKNOWN", "trigger": "heartbeat_received", "to_state": "UP"},
        ]
        table = build_transition_table(rules)
        assert (EngineState.UNKNOWN, "heartbeat_received") in table
        assert table[(EngineState.UNKNOWN, "heartbeat_received")] == EngineState.UP

    def test_default_transitions_complete(self):
        assert len(DEFAULT_TRANSITIONS) == 7
        assert (EngineState.UNKNOWN, "heartbeat_received") in DEFAULT_TRANSITIONS
        assert (EngineState.DOWN, "heartbeat_received") in DEFAULT_TRANSITIONS


class TestStateMachine:
    """Test StateMachine state transitions with asyncio.Lock."""

    @pytest.fixture
    def sm(self):
        return StateMachine("test_engine", buffer_size=100, consecutive_ok_threshold=3)

    def test_initial_state(self, sm):
        assert sm.state == EngineState.UNKNOWN
        assert sm.engine_id == "test_engine"
        assert len(sm.history) == 0

    def test_unknown_to_starting(self, sm):
        event = asyncio.run(sm.process_heartbeat(20.0, 100.0))
        assert event is not None
        assert event.old_state == "UNKNOWN"
        assert event.new_state == "STARTING"
        assert sm.state == EngineState.STARTING
        assert len(sm.history) == 1

    def test_starting_to_up(self, sm):
        asyncio.run(sm.process_heartbeat(20.0, 100.0))  # -> STARTING
        asyncio.run(sm.process_heartbeat(20.0, 100.0))  # count=2
        event = asyncio.run(sm.process_heartbeat(20.0, 100.0))  # count=3 -> UP
        assert event is not None
        assert event.new_state == "UP"
        assert sm.state == EngineState.UP

    def test_up_to_degraded(self, sm):
        # Get to UP state
        asyncio.run(sm.process_heartbeat(20.0, 100.0))
        asyncio.run(sm.process_heartbeat(20.0, 100.0))
        asyncio.run(sm.process_heartbeat(20.0, 100.0))
        assert sm.state == EngineState.UP

        # High latency -> DEGRADED
        event = asyncio.run(sm.process_heartbeat(150.0, 100.0))
        assert event is not None
        assert event.new_state == "DEGRADED"

    def test_degraded_to_up(self, sm):
        asyncio.run(sm.process_heartbeat(20.0, 100.0))
        asyncio.run(sm.process_heartbeat(20.0, 100.0))
        asyncio.run(sm.process_heartbeat(20.0, 100.0))
        asyncio.run(sm.process_heartbeat(150.0, 100.0))  # -> DEGRADED
        assert sm.state == EngineState.DEGRADED

        event = asyncio.run(sm.process_heartbeat(50.0, 100.0))  # recover
        assert event is not None
        assert event.new_state == "UP"

    def test_up_to_down_via_missed(self, sm):
        asyncio.run(sm.process_heartbeat(20.0, 100.0))
        asyncio.run(sm.process_heartbeat(20.0, 100.0))
        asyncio.run(sm.process_heartbeat(20.0, 100.0))
        assert sm.state == EngineState.UP

        for i in range(4):
            event = asyncio.run(sm.tick_missed(5))
            assert event is None  # not yet at threshold
        event = asyncio.run(sm.tick_missed(5))
        assert event is not None
        assert event.new_state == "DOWN"

    def test_down_to_starting(self, sm):
        asyncio.run(sm.process_heartbeat(20.0, 100.0))
        asyncio.run(sm.process_heartbeat(20.0, 100.0))
        asyncio.run(sm.process_heartbeat(20.0, 100.0))
        for _ in range(5):
            asyncio.run(sm.tick_missed(5))
        assert sm.state == EngineState.DOWN

        event = asyncio.run(sm.process_heartbeat(20.0, 100.0))
        assert event is not None
        assert event.new_state == "STARTING"

    def test_tick_missed_from_unknown(self, sm):
        event = asyncio.run(sm.tick_missed(5))
        assert event is None

    def test_tick_missed_from_down(self, sm):
        asyncio.run(sm.process_heartbeat(20.0, 100.0))
        asyncio.run(sm.process_heartbeat(20.0, 100.0))
        asyncio.run(sm.process_heartbeat(20.0, 100.0))
        for _ in range(5):
            asyncio.run(sm.tick_missed(5))
        assert sm.state == EngineState.DOWN
        event = asyncio.run(sm.tick_missed(5))
        assert event is None

    def test_no_transition_up_normal_latency(self, sm):
        asyncio.run(sm.process_heartbeat(20.0, 100.0))
        asyncio.run(sm.process_heartbeat(20.0, 100.0))
        asyncio.run(sm.process_heartbeat(20.0, 100.0))
        assert sm.state == EngineState.UP
        event = asyncio.run(sm.process_heartbeat(50.0, 100.0))
        assert event is None

    def test_degraded_stays_degraded_high_latency(self, sm):
        asyncio.run(sm.process_heartbeat(20.0, 100.0))
        asyncio.run(sm.process_heartbeat(20.0, 100.0))
        asyncio.run(sm.process_heartbeat(20.0, 100.0))
        asyncio.run(sm.process_heartbeat(150.0, 100.0))
        assert sm.state == EngineState.DEGRADED
        event = asyncio.run(sm.process_heartbeat(150.0, 100.0))
        assert event is None  # stays degraded

    def test_starting_increments_but_no_transition(self, sm):
        asyncio.run(sm.process_heartbeat(20.0, 100.0))  # -> STARTING
        event = asyncio.run(sm.process_heartbeat(20.0, 100.0))  # count=2
        assert event is None
        assert sm.state == EngineState.STARTING

    def test_circular_buffer(self):
        sm = StateMachine("buf_test", buffer_size=3, consecutive_ok_threshold=2)
        asyncio.run(sm.process_heartbeat(20.0, 100.0))  # -> STARTING
        asyncio.run(sm.process_heartbeat(20.0, 100.0))  # -> UP
        asyncio.run(sm.process_heartbeat(150.0, 100.0))  # -> DEGRADED
        asyncio.run(sm.process_heartbeat(20.0, 100.0))  # -> UP
        assert len(sm.history) == 3  # buffer_size=3, oldest dropped

    def test_custom_transition_table(self):
        custom = {
            (EngineState.UNKNOWN, "heartbeat_received"): EngineState.UP,
        }
        sm = StateMachine("custom", transitions=custom)
        event = asyncio.run(sm.process_heartbeat(20.0, 100.0))
        assert event is not None
        assert event.new_state == "UP"


class TestDataclasses:
    """Test dataclass creation."""

    def test_state_change_event(self):
        e = StateChangeEvent("eng1", "UNKNOWN", "STARTING", "2024-01-01", "test")
        assert e.engine_id == "eng1"

    def test_performance_snapshot(self):
        p = PerformanceSnapshot("eng1", "ts", 10.0, 20.0, 30.0, 100.0, 512.0, 40.0)
        assert p.latency_p50_ms == 10.0

    def test_security_event(self):
        s = SecurityEvent("HASH_VERIFIED", "eng1", "ts", "detail", "INFO")
        assert s.event_type == "HASH_VERIFIED"

    def test_monitoring_alert(self):
        a = MonitoringAlert("id", "HEALTH", "CRITICAL", "eng1", "ts", "msg")
        assert a.resolved is False
