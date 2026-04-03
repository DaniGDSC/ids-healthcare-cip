"""Tests for DashboardReporter."""

from __future__ import annotations

import asyncio
import random

from src.phase7_monitoring_engine.phase7.alert_dispatcher import AlertDispatcher
from src.phase7_monitoring_engine.phase7.config import EngineEntry, Phase7Config
from src.phase7_monitoring_engine.phase7.dashboard_reporter import DashboardReporter
from src.phase7_monitoring_engine.phase7.performance_collector import (
    PerformanceCollector,
)
from src.phase7_monitoring_engine.phase7.state_machine import (
    EngineState,
    StateMachine,
)


def _make_config(**overrides):
    defaults = {
        "engines": [EngineEntry(id="eng1", heartbeat_topic="eng1.hb")],
    }
    defaults.update(overrides)
    return Phase7Config(**defaults)


def _make_reporter(config=None):
    cfg = config or _make_config()
    machines = {e.id: StateMachine(e.id) for e in cfg.engines}
    d = AlertDispatcher()
    rng = random.Random(42)
    pc = PerformanceCollector(cfg, d, rng)
    return DashboardReporter(cfg, machines, pc, d), machines, pc, d


class TestDashboardReporter:
    """Test dashboard push functionality."""

    def test_start_stop(self):
        dr, *_ = _make_reporter()
        asyncio.run(dr.start())
        assert dr.get_status().running is True
        dr.stop()
        assert dr.get_status().running is False

    def test_get_config(self):
        dr, *_ = _make_reporter()
        cfg = dr.get_config()
        assert "push_interval_seconds" in cfg
        assert cfg["protocol"] == "WebSocket (simulated)"

    def test_push_update(self):
        dr, *_ = _make_reporter()
        payload = dr.push_update()
        assert payload["push_id"] == 1
        assert "engines" in payload
        assert "system_overview" in payload
        assert "active_alerts" in payload
        assert dr.push_count == 1

    def test_push_includes_engine_states(self):
        dr, machines, *_ = _make_reporter()
        payload = dr.push_update()
        assert "eng1" in payload["engines"]
        assert payload["engines"]["eng1"]["state"] == EngineState.UNKNOWN.value

    def test_push_includes_metrics_when_available(self):
        dr, _, pc, _ = _make_reporter()
        snap = pc.simulate_metrics("eng1")
        pc.record(snap)
        payload = dr.push_update()
        assert payload["engines"]["eng1"]["metrics"] is not None

    def test_push_metrics_none_when_empty(self):
        dr, *_ = _make_reporter()
        payload = dr.push_update()
        assert payload["engines"]["eng1"]["metrics"] is None

    def test_process_cycle(self):
        dr, *_ = _make_reporter()
        asyncio.run(dr.process_cycle())
        assert dr.push_count == 1

    def test_system_overview(self):
        dr, *_ = _make_reporter()
        payload = dr.push_update()
        overview = payload["system_overview"]
        assert overview["UNKNOWN"] == 1
        assert overview["UP"] == 0
