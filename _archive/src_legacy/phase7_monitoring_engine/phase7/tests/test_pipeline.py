"""Tests for MonitoringPipeline integration."""

from __future__ import annotations

from src.phase7_monitoring_engine.phase7.config import EngineEntry, Phase7Config
from src.phase7_monitoring_engine.phase7.pipeline import (
    MonitoringPipeline,
    _detect_hardware,
    _get_git_commit,
)


def _make_config(project_root, **overrides):
    engines = overrides.pop(
        "engines",
        [
            EngineEntry(
                id="eng1",
                heartbeat_topic="eng1.hb",
                artifact_dir=str(project_root / "artifacts"),
                metadata_path="metadata.json",
                name="Test Engine",
            )
        ],
    )
    defaults = {
        "engines": engines,
        "n_cycles": 2,
        "heartbeat_miss_probability": 0.0,
        "heartbeat_spike_probability": 0.0,
    }
    defaults.update(overrides)
    return Phase7Config(**defaults)


class TestMonitoringPipeline:
    """Test full pipeline integration."""

    def test_run_pipeline(self, tmp_path):
        art_dir = tmp_path / "artifacts"
        art_dir.mkdir()
        cfg = _make_config(tmp_path)
        pipeline = MonitoringPipeline(cfg, tmp_path)
        summary = pipeline.run(n_cycles=2)

        assert summary["n_cycles"] == 2
        assert summary["n_engines"] == 1
        assert summary["total_state_changes"] >= 1
        assert summary["dashboard_pushes"] == 2

    def test_engine_states(self, tmp_path):
        art_dir = tmp_path / "artifacts"
        art_dir.mkdir()
        cfg = _make_config(tmp_path)
        pipeline = MonitoringPipeline(cfg, tmp_path)
        pipeline.run(n_cycles=3)

        states = pipeline.engine_states
        assert "eng1" in states

    def test_export_artifacts(self, tmp_path):
        art_dir = tmp_path / "artifacts"
        art_dir.mkdir()
        out_dir = tmp_path / "data" / "phase7"
        cfg = _make_config(tmp_path, output_dir=out_dir)
        pipeline = MonitoringPipeline(cfg, tmp_path)
        summary = pipeline.run(n_cycles=1)
        hashes = pipeline.export_artifacts(summary)

        assert len(hashes) == 4
        assert (tmp_path / out_dir / "monitoring_log.json").exists()

    def test_generate_report(self, tmp_path):
        art_dir = tmp_path / "artifacts"
        art_dir.mkdir()
        report_dir = tmp_path / "reports"
        cfg = _make_config(tmp_path)
        pipeline = MonitoringPipeline(cfg, tmp_path)
        summary = pipeline.run(n_cycles=1)
        hashes = pipeline.export_artifacts(summary)

        report_path = pipeline.generate_report(
            report_dir=report_dir,
            summary=summary,
            artifact_hashes=hashes,
            git_commit="abc123def456",
            hardware={"device": "CPU", "python": "3.11"},
        )
        assert report_path.exists()
        content = report_path.read_text()
        assert "10.1 System Monitoring" in content
        assert "eng1" in content

    def test_data_accessors(self, tmp_path):
        art_dir = tmp_path / "artifacts"
        art_dir.mkdir()
        cfg = _make_config(tmp_path)
        pipeline = MonitoringPipeline(cfg, tmp_path)
        pipeline.run(n_cycles=2)

        assert isinstance(pipeline.state_changes, list)
        assert isinstance(pipeline.alerts, list)
        assert isinstance(pipeline.security_events, list)
        assert isinstance(pipeline.performance_snapshots, dict)
        assert pipeline.dashboard.push_count == 2


class TestUtilFunctions:
    """Test hardware detection and git commit."""

    def test_detect_hardware(self):
        hw = _detect_hardware()
        assert "python" in hw
        assert "platform" in hw
        assert "device" in hw

    def test_get_git_commit(self, tmp_path):
        commit = _get_git_commit(tmp_path)
        # May return "unknown" in tmp_path (no git repo)
        assert isinstance(commit, str)
