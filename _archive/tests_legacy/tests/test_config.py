"""Tests for production configuration loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from config.production_loader import cfg, reload


@pytest.fixture(autouse=True)
def _reload_config():
    """Ensure fresh config for each test."""
    reload()
    yield


class TestConfigLoader:
    def test_loads_yaml(self):
        val = cfg("streaming.window_size")
        assert val == 20

    def test_nested_access(self):
        assert cfg("calibration.target_fpr") == 0.10

    def test_deep_nested(self):
        assert cfg("calibration.benign_thresholds.normal") == 30.0

    def test_missing_key_returns_default(self):
        assert cfg("nonexistent.key", 42) == 42

    def test_missing_key_returns_none(self):
        assert cfg("nonexistent.key") is None

    def test_partial_path_returns_default(self):
        assert cfg("streaming.nonexistent", "fallback") == "fallback"

    def test_all_sections_present(self):
        for section in [
            "streaming", "inference", "calibration", "baseline",
            "risk_scoring", "cia", "clinical", "alert_fatigue", "simulation",
        ]:
            assert cfg(section) is not None, f"Section '{section}' missing"

    def test_config_values_types(self):
        assert isinstance(cfg("streaming.window_size"), int)
        assert isinstance(cfg("calibration.target_fpr"), float)
        assert isinstance(cfg("clinical.safety_critical_devices"), list)

    def test_reload_works(self):
        val1 = cfg("streaming.window_size")
        reload()
        val2 = cfg("streaming.window_size")
        assert val1 == val2

    def test_missing_file_returns_defaults(self, tmp_path):
        reload(tmp_path / "nonexistent.yaml")
        assert cfg("streaming.window_size", 20) == 20
