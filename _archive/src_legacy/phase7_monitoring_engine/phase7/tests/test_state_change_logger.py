"""Tests for StateChangeLogger."""

from __future__ import annotations

from src.phase7_monitoring_engine.phase7.state_change_logger import (
    StateChangeLogger,
)
from src.phase7_monitoring_engine.phase7.state_machine import StateChangeEvent


def _event(engine_id="eng1", old="UNKNOWN", new="STARTING"):
    return StateChangeEvent(engine_id, old, new, "ts", "test")


class TestStateChangeLogger:
    """Test circular buffer state change logging."""

    def test_log_and_retrieve(self):
        lg = StateChangeLogger(buffer_size=100)
        lg.log(_event())
        assert len(lg.all_events) == 1

    def test_circular_buffer_overflow(self):
        lg = StateChangeLogger(buffer_size=3)
        for i in range(5):
            lg.log(_event(engine_id=f"eng{i}"))
        assert len(lg.all_events) == 3
        assert lg.all_events[0].engine_id == "eng2"

    def test_get_events_for(self):
        lg = StateChangeLogger()
        lg.log(_event(engine_id="a"))
        lg.log(_event(engine_id="b"))
        lg.log(_event(engine_id="a"))
        assert len(lg.get_events_for("a")) == 2
        assert len(lg.get_events_for("b")) == 1
        assert len(lg.get_events_for("c")) == 0
