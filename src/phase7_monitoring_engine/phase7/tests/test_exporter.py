"""Tests for MonitoringExporter."""

from __future__ import annotations

import json

from src.phase7_monitoring_engine.phase7.exporter import (
    HEALTH_REPORT,
    MONITORING_LOG,
    PERFORMANCE_REPORT,
    SECURITY_AUDIT_LOG,
    MonitoringExporter,
)
from src.phase7_monitoring_engine.phase7.state_machine import (
    MonitoringAlert,
    PerformanceSnapshot,
    SecurityEvent,
    StateChangeEvent,
)


def _make_data():
    sc = [StateChangeEvent("eng1", "UNKNOWN", "STARTING", "ts", "test")]
    states = {"eng1": "STARTING"}
    names = {"eng1": "Test Engine"}
    perf = {"eng1": [PerformanceSnapshot("eng1", "ts", 10.0, 20.0, 30.0, 100.0, 512.0, 40.0)]}
    sec = [SecurityEvent("HASH_VERIFIED", "eng1", "ts", "ok", "INFO")]
    alerts = [MonitoringAlert("a1", "HEALTH", "WARNING", "eng1", "ts", "msg")]
    summary = {"n_cycles": 5, "n_engines": 1}
    return sc, states, names, perf, sec, alerts, summary


class TestMonitoringExporter:
    """Test artifact export and SHA-256 hashing."""

    def test_export_creates_files(self, tmp_path):
        sc, states, names, perf, sec, alerts, summary = _make_data()
        exporter = MonitoringExporter(tmp_path)
        exporter.export(sc, states, names, perf, sec, alerts, summary)

        assert (tmp_path / MONITORING_LOG).exists()
        assert (tmp_path / HEALTH_REPORT).exists()
        assert (tmp_path / PERFORMANCE_REPORT).exists()
        assert (tmp_path / SECURITY_AUDIT_LOG).exists()

    def test_export_returns_hashes(self, tmp_path):
        sc, states, names, perf, sec, alerts, summary = _make_data()
        exporter = MonitoringExporter(tmp_path)
        hashes = exporter.export(sc, states, names, perf, sec, alerts, summary)

        assert len(hashes) == 4
        for name, digest in hashes.items():
            assert len(digest) == 64

    def test_monitoring_log_content(self, tmp_path):
        sc, states, names, perf, sec, alerts, summary = _make_data()
        exporter = MonitoringExporter(tmp_path)
        exporter.export(sc, states, names, perf, sec, alerts, summary)

        data = json.loads((tmp_path / MONITORING_LOG).read_text())
        assert len(data) == 1
        assert data[0]["engine_id"] == "eng1"

    def test_health_report_content(self, tmp_path):
        sc, states, names, perf, sec, alerts, summary = _make_data()
        exporter = MonitoringExporter(tmp_path)
        exporter.export(sc, states, names, perf, sec, alerts, summary)

        data = json.loads((tmp_path / HEALTH_REPORT).read_text())
        assert data["pipeline"] == "phase7_monitoring"
        assert "eng1" in data["engine_health"]
        assert data["engine_health"]["eng1"]["name"] == "Test Engine"

    def test_creates_output_dir(self, tmp_path):
        out = tmp_path / "nested" / "output"
        sc, states, names, perf, sec, alerts, summary = _make_data()
        exporter = MonitoringExporter(out)
        exporter.export(sc, states, names, perf, sec, alerts, summary)
        assert out.exists()
