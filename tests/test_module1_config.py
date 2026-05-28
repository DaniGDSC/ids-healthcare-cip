"""Phase1Config tests — strict YAML loader + pydantic validators.

Critical invariants:
  - Unknown top-level YAML section rejected (silent-fallback defense)
  - Ratios MUST sum to 1.0 (model validator)
  - Threshold range, smote k≥1 enforced (field validators)
  - data_path escape rejected via PathValidator.validate_path_containment
"""
from __future__ import annotations


import pytest
import yaml
from pydantic import ValidationError

from module1_preprocessing.config import (
    ALLOWED_TOP_LEVEL,
    ConfigError,
    Phase1Config,
)


MINIMAL_YAML = {
    "data": {
        "input_dir": "data/raw/WUSTL-EHMS",
        "output_dir": "data/processed",
        "label_column": "Label",
    },
    "identifier_removal": {"enabled": True, "remove_columns": ["SrcAddr"]},
    "cleaning": {
        "biometric_strategy": "median",
        "biometric_columns": ["Pulse_Rate"],
        "network_strategy": "dropna",
    },
    "correlation_removal": {
        "enabled": True,
        "threshold": 0.95,
        "phase0_corr_file": "results/phase0_analysis/high_correlations.csv",
    },
    "splitting": {
        "train_ratio": 0.60,
        "val_ratio": 0.15,
        "test_ratio": 0.15,
        "demo_ratio": 0.10,
        "random_state": 42,
    },
    "normalization": {"method": "robust"},
    "track_a": {"smote": {"enabled": True, "k_neighbors": 5}},
    "track_b": {"enabled": True},
    "output": {},
}


@pytest.fixture
def workspace(tmp_path):
    (tmp_path / "data" / "raw" / "WUSTL-EHMS").mkdir(parents=True)
    (tmp_path / "data" / "processed").mkdir(parents=True)
    (tmp_path / "results" / "phase0_analysis").mkdir(parents=True)
    return tmp_path


def _write(path, data):
    path.write_text(yaml.safe_dump(data))


# ── Happy path ──────────────────────────────────────────────────────────


def test_minimal_yaml_loads(workspace):
    p = workspace / "phase1.yaml"
    _write(p, MINIMAL_YAML)
    cfg = Phase1Config.from_yaml(p, workspace_root=workspace)
    assert cfg.train_ratio == 0.60
    assert cfg.random_state == 42


# ── Strict mode: unknown section rejected ──────────────────────────────


def test_unknown_top_level_section_rejected(workspace):
    bad = {**MINIMAL_YAML, "hipaa": {"foo": 1}}  # 'hipaa' is the legacy typo
    p = workspace / "phase1.yaml"
    _write(p, bad)
    with pytest.raises(ConfigError, match="unknown top-level section"):
        Phase1Config.from_yaml(p, workspace_root=workspace)


def test_allowed_top_level_set_includes_canonical_sections():
    """Sanity: ALLOWED_TOP_LEVEL contains the sections the loader reads."""
    needed = {"data", "identifier_removal", "encoding", "cleaning",
              "variance_filtering", "correlation_removal", "splitting",
              "normalization", "track_a", "track_b", "output"}
    assert needed.issubset(ALLOWED_TOP_LEVEL)


def test_yaml_not_a_mapping_raises_configerror(workspace):
    p = workspace / "phase1.yaml"
    p.write_text("- a\n- b\n")
    with pytest.raises(ConfigError, match="must contain a YAML mapping"):
        Phase1Config.from_yaml(p, workspace_root=workspace)


def test_unparseable_yaml_raises_configerror(workspace):
    p = workspace / "phase1.yaml"
    p.write_text("data:\n  input_dir: 'unterminated\n")
    with pytest.raises(ConfigError, match="Failed to parse YAML"):
        Phase1Config.from_yaml(p, workspace_root=workspace)


def test_file_not_found_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        Phase1Config.from_yaml(tmp_path / "nope.yaml")


# ── Ratio sum validator ──────────────────────────────────────────────────


def test_ratios_must_sum_to_one(workspace):
    bad = {**MINIMAL_YAML}
    bad["splitting"] = {**bad["splitting"], "demo_ratio": 0.20}  # sum=1.1
    p = workspace / "phase1.yaml"
    _write(p, bad)
    with pytest.raises(ValidationError, match="must equal 1.0"):
        Phase1Config.from_yaml(p, workspace_root=workspace)


def test_threshold_zero_rejected(workspace):
    bad = {**MINIMAL_YAML}
    bad["correlation_removal"] = {**bad["correlation_removal"], "threshold": 0.0}
    p = workspace / "phase1.yaml"
    _write(p, bad)
    with pytest.raises(ValidationError, match="correlation_threshold must be"):
        Phase1Config.from_yaml(p, workspace_root=workspace)


def test_threshold_above_one_rejected(workspace):
    bad = {**MINIMAL_YAML}
    bad["correlation_removal"] = {**bad["correlation_removal"], "threshold": 1.5}
    p = workspace / "phase1.yaml"
    _write(p, bad)
    with pytest.raises(ValidationError, match="correlation_threshold must be"):
        Phase1Config.from_yaml(p, workspace_root=workspace)


def test_smote_k_below_one_rejected(workspace):
    bad = {**MINIMAL_YAML}
    bad["track_a"] = {"smote": {"enabled": True, "k_neighbors": 0}}
    p = workspace / "phase1.yaml"
    _write(p, bad)
    with pytest.raises(ValidationError, match="smote_k_neighbors must be"):
        Phase1Config.from_yaml(p, workspace_root=workspace)


# ── Non-canonical random_state warning ──────────────────────────────────


def test_non_canonical_random_state_warns(workspace, caplog):
    import logging
    bad = {**MINIMAL_YAML}
    bad["splitting"] = {**bad["splitting"], "random_state": 1337}  # not in {0,7,42}
    p = workspace / "phase1.yaml"
    _write(p, bad)
    caplog.set_level(logging.WARNING)
    cfg = Phase1Config.from_yaml(p, workspace_root=workspace)
    assert cfg.random_state == 1337  # accepted, just warned
    msgs = " ".join(r.message for r in caplog.records)
    assert "research-integrity smell" in msgs or "1337" in msgs


# ── Path escape rejected ────────────────────────────────────────────────


def test_input_dir_escape_rejected(workspace):
    bad = {**MINIMAL_YAML}
    bad["data"] = {**bad["data"], "input_dir": "../etc"}
    p = workspace / "phase1.yaml"
    _write(p, bad)
    with pytest.raises(PermissionError, match="Path escapes workspace"):
        Phase1Config.from_yaml(p, workspace_root=workspace)


def test_phase0_integrity_file_no_longer_in_model():
    """Y2 fix: dead field must be gone from the pydantic model."""
    assert "phase0_integrity_file" not in Phase1Config.model_fields
