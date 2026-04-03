"""Tests for SecurityMonitor."""

from __future__ import annotations

import asyncio
import json

from src.phase7_monitoring_engine.phase7.alert_dispatcher import AlertDispatcher
from src.phase7_monitoring_engine.phase7.config import EngineEntry, Phase7Config
from src.phase7_monitoring_engine.phase7.integrity_checker import (
    ArtifactIntegrityChecker,
)
from src.phase7_monitoring_engine.phase7.security_monitor import SecurityMonitor


def _make_config(**overrides):
    defaults = {
        "engines": [
            EngineEntry(
                id="test_eng",
                heartbeat_topic="test.hb",
                artifact_dir="artifacts",
                metadata_path="metadata.json",
            )
        ],
    }
    defaults.update(overrides)
    return Phase7Config(**defaults)


class TestSecurityMonitor:
    """Test security monitoring coordination."""

    def test_start_stop(self, tmp_path):
        cfg = _make_config()
        d = AlertDispatcher()
        checker = ArtifactIntegrityChecker(cfg, d, tmp_path)
        sm = SecurityMonitor(cfg, d, checker)
        asyncio.run(sm.start())
        assert sm.get_status().running is True
        sm.stop()
        assert sm.get_status().running is False

    def test_get_config(self, tmp_path):
        cfg = _make_config()
        d = AlertDispatcher()
        checker = ArtifactIntegrityChecker(cfg, d, tmp_path)
        sm = SecurityMonitor(cfg, d, checker)
        config = sm.get_config()
        assert "integrity_check_interval" in config

    def test_process_cycle_no_violations(self, tmp_path):
        # No metadata files -> METADATA_MISSING (WARNING, not CRITICAL)
        cfg = _make_config()
        d = AlertDispatcher()
        checker = ArtifactIntegrityChecker(cfg, d, tmp_path)
        sm = SecurityMonitor(cfg, d, checker)
        asyncio.run(sm.process_cycle())
        # No CRITICAL events -> no alerts dispatched
        assert len(d.all_alerts) == 0

    def test_process_cycle_hash_mismatch_generates_alert(self, tmp_path):
        art_dir = tmp_path / "artifacts"
        art_dir.mkdir()
        artifact = art_dir / "model.h5"
        artifact.write_text("model data")
        metadata = {"artifact_hashes": {"model.h5": {"sha256": "wrong_hash"}}}
        meta_path = tmp_path / "metadata.json"
        meta_path.write_text(json.dumps(metadata))

        cfg = _make_config()
        d = AlertDispatcher()
        checker = ArtifactIntegrityChecker(cfg, d, tmp_path)
        sm = SecurityMonitor(cfg, d, checker)
        asyncio.run(sm.process_cycle())
        # Should dispatch CRITICAL alert
        assert len(d.all_alerts) == 1
        assert d.all_alerts[0].severity == "CRITICAL"
        assert d.all_alerts[0].category == "SECURITY"

    def test_all_events(self, tmp_path):
        cfg = _make_config()
        d = AlertDispatcher()
        checker = ArtifactIntegrityChecker(cfg, d, tmp_path)
        sm = SecurityMonitor(cfg, d, checker)
        asyncio.run(sm.process_cycle())
        assert len(sm.all_events) >= 1
