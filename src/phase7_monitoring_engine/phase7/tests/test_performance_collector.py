"""Tests for PerformanceCollector."""

from __future__ import annotations

import asyncio
import random

from src.phase7_monitoring_engine.phase7.alert_dispatcher import AlertDispatcher
from src.phase7_monitoring_engine.phase7.config import EngineEntry, Phase7Config
from src.phase7_monitoring_engine.phase7.performance_collector import (
    PerformanceCollector,
)
from src.phase7_monitoring_engine.phase7.state_machine import PerformanceSnapshot


def _make_config(**overrides):
    defaults = {
        "engines": [EngineEntry(id="eng1", heartbeat_topic="eng1.hb")],
    }
    defaults.update(overrides)
    return Phase7Config(**defaults)


def _make_collector(config=None, rng=None):
    cfg = config or _make_config()
    d = AlertDispatcher()
    r = rng or random.Random(42)
    return PerformanceCollector(cfg, d, r), d


class TestPerformanceCollector:
    """Test performance metric collection."""

    def test_start_stop(self):
        pc, _ = _make_collector()
        asyncio.run(pc.start())
        assert pc.get_status().running is True
        pc.stop()
        assert pc.get_status().running is False

    def test_get_config(self):
        pc, _ = _make_collector()
        cfg = pc.get_config()
        assert "collection_interval" in cfg
        assert "latency_threshold_ms" in cfg

    def test_record_and_get_latest(self):
        pc, _ = _make_collector()
        snap = PerformanceSnapshot("eng1", "ts", 10.0, 20.0, 30.0, 100.0, 512.0, 40.0)
        pc.record(snap)
        assert pc.get_latest("eng1") == snap

    def test_get_latest_empty(self):
        pc, _ = _make_collector()
        assert pc.get_latest("nonexistent") is None

    def test_get_all(self):
        pc, _ = _make_collector()
        for i in range(3):
            pc.record(PerformanceSnapshot("eng1", f"ts{i}", 10.0, 20.0, 30.0, 100.0, 512.0, 40.0))
        assert len(pc.get_all("eng1")) == 3

    def test_threshold_latency_alert(self):
        cfg = _make_config(latency_p95_threshold_ms=10.0)
        pc, dispatcher = _make_collector(config=cfg)
        snap = PerformanceSnapshot("eng1", "ts", 10.0, 50.0, 80.0, 100.0, 512.0, 40.0)
        pc.record(snap)
        alerts = pc.check_thresholds("eng1")
        assert len(alerts) == 1
        assert "Latency" in alerts[0].message

    def test_threshold_cpu_alert(self):
        cfg = _make_config(cpu_warning_threshold_pct=30.0)
        pc, _ = _make_collector(config=cfg)
        snap = PerformanceSnapshot("eng1", "ts", 10.0, 20.0, 30.0, 100.0, 512.0, 50.0)
        pc.record(snap)
        alerts = pc.check_thresholds("eng1")
        assert any("CPU" in a.message for a in alerts)

    def test_no_alerts_within_threshold(self):
        pc, _ = _make_collector()
        snap = PerformanceSnapshot("eng1", "ts", 10.0, 20.0, 30.0, 100.0, 200.0, 30.0)
        pc.record(snap)
        alerts = pc.check_thresholds("eng1")
        assert len(alerts) == 0

    def test_simulate_metrics(self):
        pc, _ = _make_collector()
        snap = pc.simulate_metrics("eng1")
        assert snap.engine_id == "eng1"
        assert snap.latency_p50_ms >= 1.0
        assert snap.cpu_pct >= 1.0

    def test_process_cycle(self):
        pc, dispatcher = _make_collector()
        asyncio.run(pc.process_cycle())
        assert pc.get_latest("eng1") is not None
