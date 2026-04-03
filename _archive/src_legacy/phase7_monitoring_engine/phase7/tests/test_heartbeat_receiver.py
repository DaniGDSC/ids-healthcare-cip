"""Tests for HeartbeatReceiver."""

from __future__ import annotations

import asyncio
import random
from pathlib import Path

from src.phase7_monitoring_engine.phase7.alert_dispatcher import AlertDispatcher
from src.phase7_monitoring_engine.phase7.config import EngineEntry, Phase7Config
from src.phase7_monitoring_engine.phase7.heartbeat_receiver import HeartbeatReceiver
from src.phase7_monitoring_engine.phase7.state_change_logger import StateChangeLogger
from src.phase7_monitoring_engine.phase7.state_machine import (
    EngineState,
    StateMachine,
)


def _make_config(engines=None, **overrides):
    defaults = {
        "engines": engines or [EngineEntry(id="test_eng", heartbeat_topic="test.hb")],
        "heartbeat_miss_probability": 0.0,
        "heartbeat_spike_probability": 0.0,
    }
    defaults.update(overrides)
    return Phase7Config(**defaults)


def _make_receiver(config=None, project_root=None, rng=None, machines=None):
    cfg = config or _make_config()
    root = project_root or Path("/tmp")
    r = rng or random.Random(42)
    m = machines or {e.id: StateMachine(e.id, consecutive_ok_threshold=3) for e in cfg.engines}
    d = AlertDispatcher()
    lg = StateChangeLogger()
    return HeartbeatReceiver(cfg, m, d, lg, root, r), m, d, lg


class TestHeartbeatReceiver:
    """Test heartbeat processing."""

    def test_start_stop(self):
        hb, *_ = _make_receiver()
        asyncio.run(hb.start())
        assert hb.get_status().running is True
        hb.stop()
        assert hb.get_status().running is False

    def test_get_config(self):
        hb, *_ = _make_receiver()
        cfg = hb.get_config()
        assert "heartbeat_interval_seconds" in cfg
        assert "missed_heartbeat_threshold" in cfg

    def test_process_cycle_transitions(self, tmp_path):
        engines = [
            EngineEntry(
                id="eng1",
                heartbeat_topic="eng1.hb",
                artifact_dir=str(tmp_path),
            )
        ]
        cfg = _make_config(engines=engines)
        hb, machines, dispatcher, logger = _make_receiver(config=cfg, project_root=tmp_path.parent)
        asyncio.run(hb.process_cycle())
        assert machines["eng1"].state == EngineState.STARTING
        assert len(logger.all_events) == 1

    def test_missed_heartbeat_no_artifact_dir(self, tmp_path):
        engines = [
            EngineEntry(
                id="eng1",
                heartbeat_topic="eng1.hb",
                artifact_dir=str(tmp_path / "nonexistent"),
            )
        ]
        cfg = _make_config(engines=engines)
        hb, machines, _, _ = _make_receiver(config=cfg, project_root=tmp_path.parent)
        # With no artifact dir, heartbeat returns None -> tick_missed
        asyncio.run(hb.process_cycle())
        # Still UNKNOWN since missed from UNKNOWN does nothing
        assert machines["eng1"].state == EngineState.UNKNOWN

    def test_alert_on_degraded(self, tmp_path):
        engines = [
            EngineEntry(
                id="eng1",
                heartbeat_topic="eng1.hb",
                artifact_dir=str(tmp_path),
            )
        ]
        cfg = _make_config(
            engines=engines,
            latency_p95_threshold_ms=10.0,
            heartbeat_spike_probability=0.0,
            normal_latency_mean_ms=50.0,
            normal_latency_std_ms=0.1,
        )
        hb, machines, dispatcher, _ = _make_receiver(config=cfg, project_root=tmp_path.parent)
        # 3 cycles to get to UP, then one with high latency -> DEGRADED
        for _ in range(4):
            asyncio.run(hb.process_cycle())
        # Should have generated a DEGRADED alert
        degraded_alerts = [a for a in dispatcher.all_alerts if "DEGRADED" in a.message]
        assert len(degraded_alerts) >= 1
