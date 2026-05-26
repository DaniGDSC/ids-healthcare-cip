"""Phase0Config validation tests.

Covers:
  - YAML structure errors (missing sections, malformed file) → ConfigError
  - Value-range validation in __post_init__ → ValueError
  - Path containment via PathValidator at load time → PermissionError
  - Backwards-compat for optional fields (feature counts default to 0)
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from module0_analysis import ConfigError, Phase0Config


MINIMAL_VALID_YAML = {
    "dataset": {
        "data_path": "data/raw/WUSTL-EHMS/wustl-ehms-2020_with_attacks_categories.csv",
        "label_column": "Label",
        "required_columns": ["Label", "Attack Category"],
    },
    "analysis": {
        "correlation_threshold": 0.95,
    },
    "output": {
        "output_dir": "results/phase0_analysis",
        "stats_report_file": "stats.json",
        "high_correlations_file": "corr.csv",
        "correlation_matrix_file": "matrix.parquet",
    },
}


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """Create a workspace with required data/output dirs."""
    (tmp_path / "data" / "raw" / "WUSTL-EHMS").mkdir(parents=True)
    (tmp_path / "data" / "raw" / "WUSTL-EHMS" / "wustl-ehms-2020_with_attacks_categories.csv").touch()
    return tmp_path


def _write_yaml(path: Path, data: dict) -> None:
    path.write_text(yaml.safe_dump(data))


def test_minimal_yaml_loads(workspace, tmp_path):
    yaml_path = workspace / "config.yaml"
    _write_yaml(yaml_path, MINIMAL_VALID_YAML)
    cfg = Phase0Config.from_yaml(yaml_path, workspace_root=workspace)
    assert cfg.label_column == "Label"
    assert cfg.correlation_threshold == 0.95
    assert cfg.network_feature_count == 0  # default when absent
    assert cfg.biometric_feature_count == 0


def test_missing_dataset_section_raises_configerror(workspace):
    yaml_path = workspace / "config.yaml"
    bad = {k: v for k, v in MINIMAL_VALID_YAML.items() if k != "dataset"}
    _write_yaml(yaml_path, bad)
    with pytest.raises(ConfigError, match="missing required top-level section 'dataset'"):
        Phase0Config.from_yaml(yaml_path, workspace_root=workspace)


def test_missing_analysis_section_raises_configerror(workspace):
    yaml_path = workspace / "config.yaml"
    bad = {k: v for k, v in MINIMAL_VALID_YAML.items() if k != "analysis"}
    _write_yaml(yaml_path, bad)
    with pytest.raises(ConfigError, match="missing required top-level section 'analysis'"):
        Phase0Config.from_yaml(yaml_path, workspace_root=workspace)


def test_yaml_not_a_mapping_raises_configerror(workspace):
    yaml_path = workspace / "config.yaml"
    yaml_path.write_text("- a\n- b\n")  # top-level list
    with pytest.raises(ConfigError, match="must contain a YAML mapping"):
        Phase0Config.from_yaml(yaml_path, workspace_root=workspace)


def test_unparseable_yaml_raises_configerror(workspace):
    yaml_path = workspace / "config.yaml"
    yaml_path.write_text("dataset:\n  data_path: 'unterminated string\n")
    with pytest.raises(ConfigError, match="Failed to parse YAML"):
        Phase0Config.from_yaml(yaml_path, workspace_root=workspace)


def test_config_file_not_found_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        Phase0Config.from_yaml(tmp_path / "nope.yaml")


def test_correlation_threshold_out_of_range(workspace):
    yaml_path = workspace / "config.yaml"
    bad = {**MINIMAL_VALID_YAML, "analysis": {"correlation_threshold": 1.5}}
    _write_yaml(yaml_path, bad)
    with pytest.raises(ValueError, match="correlation_threshold must be in"):
        Phase0Config.from_yaml(yaml_path, workspace_root=workspace)


def test_negative_missing_warn_pct(workspace):
    yaml_path = workspace / "config.yaml"
    bad = {**MINIMAL_VALID_YAML,
           "analysis": {"correlation_threshold": 0.9, "missing_value_warn_pct": -1.0}}
    _write_yaml(yaml_path, bad)
    with pytest.raises(ValueError, match="missing_value_warn_pct must be"):
        Phase0Config.from_yaml(yaml_path, workspace_root=workspace)


def test_outlier_multiplier_must_be_positive(workspace):
    yaml_path = workspace / "config.yaml"
    bad = {**MINIMAL_VALID_YAML,
           "analysis": {"correlation_threshold": 0.9, "outlier_iqr_multiplier": 0}}
    _write_yaml(yaml_path, bad)
    with pytest.raises(ValueError, match="outlier_iqr_multiplier must be > 0"):
        Phase0Config.from_yaml(yaml_path, workspace_root=workspace)


def test_data_path_escape_rejected(workspace, tmp_path):
    """Resolved data_path outside workspace → PermissionError."""
    yaml_path = workspace / "config.yaml"
    bad = {**MINIMAL_VALID_YAML}
    bad["dataset"] = {**bad["dataset"], "data_path": "../escape.csv"}
    _write_yaml(yaml_path, bad)
    with pytest.raises(PermissionError, match="Path escapes workspace"):
        Phase0Config.from_yaml(yaml_path, workspace_root=workspace)
