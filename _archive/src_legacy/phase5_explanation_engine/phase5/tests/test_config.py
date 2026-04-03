"""Unit tests for Phase5Config (Pydantic-validated configuration)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.phase5_explanation_engine.phase5.config import (
    ExplanationTemplates,
    Phase5Config,
)


def _make_templates() -> ExplanationTemplates:
    return ExplanationTemplates(
        CRITICAL="CRITICAL: {idx}",
        HIGH="HIGH: {idx}",
        MEDIUM="MEDIUM: {idx}",
        LOW="LOW: {idx}",
    )


def _make_config(**overrides) -> Phase5Config:
    defaults = {
        "phase4_dir": Path("data/phase4"),
        "phase4_metadata": Path("data/phase4/risk_metadata.json"),
        "phase3_dir": Path("data/phase3"),
        "phase3_metadata": Path("data/phase3/classification_metadata.json"),
        "phase2_dir": Path("data/phase2"),
        "phase2_metadata": Path("data/phase2/detection_metadata.json"),
        "phase1_train": Path("data/processed/train_phase1.parquet"),
        "phase1_test": Path("data/processed/test_phase1.parquet"),
        "explanation_templates": _make_templates(),
        "biometric_columns": ["Temp", "SpO2"],
    }
    defaults.update(overrides)
    return Phase5Config(**defaults)


class TestExplanationTemplates:
    def test_valid_templates(self) -> None:
        t = _make_templates()
        assert t.CRITICAL == "CRITICAL: {idx}"
        assert t.HIGH == "HIGH: {idx}"
        assert t.MEDIUM == "MEDIUM: {idx}"
        assert t.LOW == "LOW: {idx}"

    def test_missing_template_key(self) -> None:
        with pytest.raises(Exception):
            ExplanationTemplates(CRITICAL="c", HIGH="h", MEDIUM="m")


class TestPhase5Config:
    def test_valid_defaults(self) -> None:
        cfg = _make_config()
        assert cfg.background_samples == 100
        assert cfg.max_explain_samples == 200
        assert cfg.top_features == 10
        assert cfg.random_state == 42

    def test_background_samples_zero(self) -> None:
        with pytest.raises(ValueError, match="background_samples"):
            _make_config(background_samples=0)

    def test_max_explain_samples_zero(self) -> None:
        with pytest.raises(ValueError, match="max_explain_samples"):
            _make_config(max_explain_samples=0)

    def test_top_features_zero(self) -> None:
        with pytest.raises(ValueError, match="top_features"):
            _make_config(top_features=0)

    def test_max_waterfall_negative(self) -> None:
        with pytest.raises(ValueError, match="max_waterfall_charts"):
            _make_config(max_waterfall_charts=-1)

    def test_max_timeline_negative(self) -> None:
        with pytest.raises(ValueError, match="max_timeline_charts"):
            _make_config(max_timeline_charts=-1)

    def test_empty_biometric_columns(self) -> None:
        with pytest.raises(ValueError, match="biometric column"):
            _make_config(biometric_columns=[])

    def test_valid_boundary_values(self) -> None:
        cfg = _make_config(
            background_samples=1,
            max_explain_samples=1,
            top_features=1,
            max_waterfall_charts=0,
            max_timeline_charts=0,
        )
        assert cfg.background_samples == 1
        assert cfg.max_waterfall_charts == 0


class TestPhase5ConfigFromYaml:
    def test_from_yaml(self, tmp_path: Path) -> None:
        yaml_content = """\
data:
  phase4_dir: "data/phase4"
  phase4_metadata: "data/phase4/risk_metadata.json"
  phase3_dir: "data/phase3"
  phase3_metadata: "data/phase3/classification_metadata.json"
  phase2_dir: "data/phase2"
  phase2_metadata: "data/phase2/detection_metadata.json"
  phase1_train: "data/processed/train_phase1.parquet"
  phase1_test: "data/processed/test_phase1.parquet"
  label_column: "Label"
shap:
  background_samples: 50
  max_explain_samples: 100
  top_features: 5
  max_waterfall_charts: 3
  max_timeline_charts: 2
output:
  output_dir: "data/phase5"
  shap_values_file: "shap_values.parquet"
  explanation_report_file: "explanation_report.json"
  metadata_file: "explanation_metadata.json"
  charts_dir: "charts"
explanation_templates:
  CRITICAL: "C: {idx}"
  HIGH: "H: {idx}"
  MEDIUM: "M: {idx}"
  LOW: "L: {idx}"
biometric_columns:
  - Temp
  - SpO2
random_state: 123
"""
        yaml_path = tmp_path / "test_config.yaml"
        yaml_path.write_text(yaml_content)
        cfg = Phase5Config.from_yaml(yaml_path)
        assert cfg.background_samples == 50
        assert cfg.random_state == 123
        assert cfg.explanation_templates.CRITICAL == "C: {idx}"

    def test_templates_loaded_from_yaml(self, tmp_path: Path) -> None:
        yaml_content = """\
data:
  phase4_dir: "data/phase4"
  phase4_metadata: "data/phase4/m.json"
  phase3_dir: "data/phase3"
  phase3_metadata: "data/phase3/m.json"
  phase2_dir: "data/phase2"
  phase2_metadata: "data/phase2/m.json"
  phase1_train: "data/processed/train.parquet"
  phase1_test: "data/processed/test.parquet"
explanation_templates:
  CRITICAL: "CRIT {idx}"
  HIGH: "HIGH {idx}"
  MEDIUM: "MED {idx}"
  LOW: "LOW {idx}"
biometric_columns:
  - Temp
"""
        yaml_path = tmp_path / "test_config2.yaml"
        yaml_path.write_text(yaml_content)
        cfg = Phase5Config.from_yaml(yaml_path)
        assert cfg.explanation_templates.HIGH == "HIGH {idx}"
        assert cfg.biometric_columns == ["Temp"]
