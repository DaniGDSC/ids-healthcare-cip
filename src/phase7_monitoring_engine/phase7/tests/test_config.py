"""Tests for Phase7Config Pydantic model."""

from __future__ import annotations

import textwrap

import pytest

from src.phase7_monitoring_engine.phase7.config import (
    EngineEntry,
    Phase7Config,
    TransitionRule,
)


def _make_config(**overrides):
    """Factory helper for Phase7Config with minimal defaults."""
    defaults = {
        "engines": [EngineEntry(id="test_engine", heartbeat_topic="test.heartbeat")],
    }
    defaults.update(overrides)
    return Phase7Config(**defaults)


class TestPhase7Config:
    """Test Phase7Config creation and validation."""

    def test_default_values(self):
        cfg = _make_config()
        assert cfg.heartbeat_interval_seconds == 5
        assert cfg.missed_heartbeat_threshold == 5
        assert cfg.circular_buffer_size == 1000
        assert cfg.n_cycles == 5
        assert cfg.random_state == 42
        assert cfg.latency_p95_threshold_ms == 100.0

    def test_custom_values(self):
        cfg = _make_config(heartbeat_interval_seconds=10, n_cycles=3)
        assert cfg.heartbeat_interval_seconds == 10
        assert cfg.n_cycles == 3

    def test_engine_entries(self):
        engines = [
            EngineEntry(
                id="phase2",
                heartbeat_topic="detection.heartbeat",
                artifact_dir="data/phase2",
                name="Detection",
            ),
            EngineEntry(
                id="phase3",
                heartbeat_topic="classification.heartbeat",
            ),
        ]
        cfg = _make_config(engines=engines)
        assert len(cfg.engines) == 2
        assert cfg.engines[0].name == "Detection"
        assert cfg.engines[1].artifact_dir is None

    def test_heartbeat_interval_validator(self):
        with pytest.raises(ValueError, match="heartbeat_interval_seconds"):
            _make_config(heartbeat_interval_seconds=0)

    def test_missed_threshold_validator(self):
        with pytest.raises(ValueError, match="missed_heartbeat_threshold"):
            _make_config(missed_heartbeat_threshold=0)

    def test_buffer_size_validator(self):
        with pytest.raises(ValueError, match="circular_buffer_size"):
            _make_config(circular_buffer_size=0)

    def test_n_cycles_validator(self):
        with pytest.raises(ValueError, match="n_cycles"):
            _make_config(n_cycles=0)

    def test_latency_threshold_validator(self):
        with pytest.raises(ValueError, match="latency_p95_threshold_ms"):
            _make_config(latency_p95_threshold_ms=-1.0)

    def test_default_transitions(self):
        rules = Phase7Config._default_transitions()
        assert len(rules) == 7
        assert all(isinstance(r, TransitionRule) for r in rules)


class TestPhase7ConfigFromYaml:
    """Test YAML loading."""

    def test_from_yaml(self, tmp_path):
        yaml_content = textwrap.dedent("""\
            monitoring:
              heartbeat_interval_seconds: 3
              missed_heartbeat_threshold: 4
              circular_buffer_size: 500
            security:
              baseline_config_path: "data/phase4/baseline.json"
            output:
              output_dir: "data/test"
            pipeline:
              n_cycles: 2
              random_state: 99
            engines:
              - id: eng1
                heartbeat_topic: eng1.hb
                artifact_dir: "data/eng1"
                name: "Engine 1"
              - id: eng2
                heartbeat_topic: eng2.hb
        """)
        cfg_path = tmp_path / "test_config.yaml"
        cfg_path.write_text(yaml_content)

        cfg = Phase7Config.from_yaml(cfg_path)
        assert cfg.heartbeat_interval_seconds == 3
        assert cfg.missed_heartbeat_threshold == 4
        assert cfg.circular_buffer_size == 500
        assert cfg.n_cycles == 2
        assert cfg.random_state == 99
        assert len(cfg.engines) == 2
        assert cfg.engines[0].id == "eng1"
        assert cfg.engines[0].name == "Engine 1"
        assert cfg.engines[1].artifact_dir is None
        assert len(cfg.transitions) == 7  # defaults

    def test_from_yaml_with_transitions(self, tmp_path):
        yaml_content = textwrap.dedent("""\
            pipeline:
              n_cycles: 1
            engines:
              - id: eng1
                heartbeat_topic: eng1.hb
            transitions:
              - from_state: UNKNOWN
                trigger: heartbeat_received
                to_state: UP
        """)
        cfg_path = tmp_path / "test_config.yaml"
        cfg_path.write_text(yaml_content)

        cfg = Phase7Config.from_yaml(cfg_path)
        assert len(cfg.transitions) == 1
        assert cfg.transitions[0].from_state == "UNKNOWN"
