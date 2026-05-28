"""Phase0ArtifactReader tests — stats + correlations consumers."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from module1_preprocessing.artifact_reader import Phase0ArtifactReader


def test_read_stats_success(tmp_path):
    stats_path = tmp_path / "stats.json"
    stats_path.write_text(json.dumps({
        "descriptive_statistics": {"Dur": {"mean": 0.5, "std": 0.1}},
        "missing_values": {},
        "class_distribution": {"Normal": {"count": 100}, "Attack": {"count": 50}},
    }))
    reader = Phase0ArtifactReader(
        project_root=tmp_path,
        stats_file=Path("stats.json"),
        corr_file=Path("corr.csv"),
    )
    out = reader.read_stats()
    assert "descriptive_statistics" in out
    assert out["descriptive_statistics"]["Dur"]["mean"] == 0.5


def test_read_stats_missing_file_raises(tmp_path):
    reader = Phase0ArtifactReader(
        project_root=tmp_path,
        stats_file=Path("missing.json"),
        corr_file=Path("corr.csv"),
    )
    with pytest.raises(FileNotFoundError, match="Phase 0 stats"):
        reader.read_stats()


def test_read_correlations_success(tmp_path):
    corr = pd.DataFrame({
        "feature_a": ["A"], "feature_b": ["B"], "correlation": [0.99],
    })
    corr_path = tmp_path / "corr.csv"
    corr.to_csv(corr_path, index=False)
    reader = Phase0ArtifactReader(
        project_root=tmp_path,
        stats_file=Path("stats.json"),
        corr_file=Path("corr.csv"),
    )
    out = reader.read_correlations()
    assert list(out.columns) == ["feature_a", "feature_b", "correlation"]
    assert len(out) == 1


def test_read_correlations_missing_file_raises(tmp_path):
    reader = Phase0ArtifactReader(
        project_root=tmp_path,
        stats_file=Path("stats.json"),
        corr_file=Path("missing.csv"),
    )
    with pytest.raises(FileNotFoundError, match="Phase 0 correlations"):
        reader.read_correlations()


def test_constructor_no_longer_accepts_integrity_file():
    """Y1 fix: dead integrity_file param must be gone."""
    import inspect
    sig = inspect.signature(Phase0ArtifactReader.__init__)
    assert "integrity_file" not in sig.parameters
