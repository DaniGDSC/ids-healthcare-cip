"""Unit tests for Phase4Config (pydantic validation + from_yaml)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pytest
import yaml

from src.phase4_risk_engine.phase4.config import KScheduleEntry, Phase4Config


def _make_k_schedule() -> list:
    return [
        KScheduleEntry(start_hour=0, end_hour=6, k=2.5),
        KScheduleEntry(start_hour=6, end_hour=22, k=3.0),
        KScheduleEntry(start_hour=22, end_hour=24, k=3.5),
    ]


def _make_config(**overrides: Any) -> Phase4Config:
    """Create a Phase4Config with sensible defaults."""
    defaults: Dict[str, Any] = {
        "phase3_dir": Path("data/phase3"),
        "phase3_metadata": Path("data/phase3/classification_metadata.json"),
        "phase2_dir": Path("data/phase2"),
        "phase2_metadata": Path("data/phase2/detection_metadata.json"),
        "phase1_train": Path("data/processed/train_phase1.parquet"),
        "phase1_test": Path("data/processed/test_phase1.parquet"),
        "k_schedule": _make_k_schedule(),
        "biometric_columns": ["Temp", "SpO2", "Pulse_Rate"],
        "output_dir": Path("data/phase4"),
    }
    defaults.update(overrides)
    return Phase4Config(**defaults)


class TestPhase4Config:
    """Validate Phase4Config construction and field validators."""

    def test_valid_defaults(self) -> None:
        cfg = _make_config()
        assert cfg.mad_multiplier == 3.0
        assert cfg.window_size == 100
        assert cfg.drift_threshold == 0.20
        assert cfg.recovery_threshold == 0.10
        assert cfg.recovery_windows == 3
        assert cfg.low_upper == 0.5
        assert cfg.random_state == 42

    def test_mad_multiplier_positive(self) -> None:
        with pytest.raises(Exception, match="mad_multiplier"):
            _make_config(mad_multiplier=0.0)

    def test_mad_multiplier_negative(self) -> None:
        with pytest.raises(Exception, match="mad_multiplier"):
            _make_config(mad_multiplier=-1.0)

    def test_window_size_positive(self) -> None:
        with pytest.raises(Exception, match="window_size"):
            _make_config(window_size=0)

    def test_drift_threshold_zero(self) -> None:
        with pytest.raises(Exception, match="drift_threshold"):
            _make_config(drift_threshold=0.0)

    def test_drift_threshold_one(self) -> None:
        with pytest.raises(Exception, match="drift_threshold"):
            _make_config(drift_threshold=1.0)

    def test_recovery_threshold_zero(self) -> None:
        with pytest.raises(Exception, match="recovery_threshold"):
            _make_config(recovery_threshold=0.0)

    def test_recovery_windows_zero(self) -> None:
        with pytest.raises(Exception, match="recovery_windows"):
            _make_config(recovery_windows=0)

    def test_empty_k_schedule(self) -> None:
        with pytest.raises(Exception, match="k_schedule"):
            _make_config(k_schedule=[])

    def test_empty_biometric_columns(self) -> None:
        with pytest.raises(Exception, match="biometric"):
            _make_config(biometric_columns=[])

    def test_valid_boundary_window_size_one(self) -> None:
        cfg = _make_config(window_size=1)
        assert cfg.window_size == 1


class TestPhase4ConfigFromYaml:
    """Test YAML loading and mapping."""

    def test_from_yaml(self, tmp_path: Path) -> None:
        yaml_content = {
            "data": {
                "phase3_dir": "data/phase3",
                "phase3_metadata": "data/phase3/classification_metadata.json",
                "phase2_dir": "data/phase2",
                "phase2_metadata": "data/phase2/detection_metadata.json",
                "phase1_train": "data/processed/train_phase1.parquet",
                "phase1_test": "data/processed/test_phase1.parquet",
                "label_column": "Label",
            },
            "baseline": {"mad_multiplier": 3.0},
            "dynamic_threshold": {
                "window_size": 100,
                "k_schedule": [
                    {"start_hour": 0, "end_hour": 6, "k": 2.5},
                    {"start_hour": 6, "end_hour": 22, "k": 3.0},
                    {"start_hour": 22, "end_hour": 24, "k": 3.5},
                ],
            },
            "concept_drift": {
                "drift_threshold": 0.20,
                "recovery_threshold": 0.10,
                "recovery_windows": 3,
            },
            "risk_levels": {
                "low_upper": 0.5,
                "medium_upper": 1.0,
                "high_upper": 2.0,
            },
            "biometric_columns": ["Temp", "SpO2"],
            "output": {
                "output_dir": "data/phase4",
                "baseline_file": "baseline_config.json",
                "threshold_file": "threshold_config.json",
                "risk_report_file": "risk_report.json",
                "drift_log_file": "drift_log.csv",
            },
            "random_state": 42,
        }
        yaml_path = tmp_path / "phase4_config.yaml"
        yaml_path.write_text(yaml.dump(yaml_content), encoding="utf-8")

        cfg = Phase4Config.from_yaml(yaml_path)
        assert cfg.phase3_dir == Path("data/phase3")
        assert cfg.mad_multiplier == 3.0
        assert cfg.window_size == 100
        assert len(cfg.k_schedule) == 3
        assert cfg.k_schedule[0].k == 2.5
        assert cfg.drift_threshold == 0.20
        assert cfg.low_upper == 0.5
        assert cfg.biometric_columns == ["Temp", "SpO2"]
        assert cfg.random_state == 42
