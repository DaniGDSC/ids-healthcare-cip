"""Unit tests for Phase4ArtifactReader (SHA-256 verification)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.phase5_explanation_engine.phase5.artifact_reader import (
    Phase4ArtifactReader,
)


def _write_artifact(path: Path, content: bytes = b"test content") -> str:
    """Write a fake artifact and return its SHA-256 hash."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return hashlib.sha256(content).hexdigest()


def _make_metadata(
    meta_path: Path,
    artifact_dir: Path,
    filenames: list[str],
) -> None:
    """Write a fake metadata JSON with correct hashes."""
    hashes = {}
    for name in filenames:
        sha = _write_artifact(artifact_dir / name, name.encode())
        hashes[name] = {"sha256": sha}
    meta = {"artifact_hashes": hashes, "hyperparameters": {}}
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta))


class TestPhase4ArtifactReader:
    def test_verify_all_valid(self, tmp_path: Path) -> None:
        root = tmp_path / "project"
        p2_dir = root / "data" / "phase2"
        p3_dir = root / "data" / "phase3"
        p4_dir = root / "data" / "phase4"

        p2_meta = root / "data" / "phase2" / "detection_metadata.json"
        p3_meta = root / "data" / "phase3" / "classification_metadata.json"
        p4_meta = root / "data" / "phase4" / "risk_metadata.json"

        _make_metadata(p2_meta, p2_dir, ["attn.parquet", "model.h5"])
        _make_metadata(p3_meta, p3_dir, ["weights.h5", "metrics.json"])
        _make_metadata(p4_meta, p4_dir, ["risk_report.json"])

        reader = Phase4ArtifactReader(
            project_root=root,
            phase4_dir=Path("data/phase4"),
            phase4_metadata=Path("data/phase4/risk_metadata.json"),
            phase3_dir=Path("data/phase3"),
            phase3_metadata=Path("data/phase3/classification_metadata.json"),
            phase2_dir=Path("data/phase2"),
            phase2_metadata=Path("data/phase2/detection_metadata.json"),
        )
        p2, p3, p4 = reader.verify_all()
        assert "artifact_hashes" in p2
        assert "artifact_hashes" in p3
        assert "artifact_hashes" in p4

    def test_sha256_mismatch(self, tmp_path: Path) -> None:
        root = tmp_path / "project"
        p2_dir = root / "data" / "phase2"
        p2_meta = root / "data" / "phase2" / "meta.json"

        p2_dir.mkdir(parents=True, exist_ok=True)
        (p2_dir / "file.bin").write_bytes(b"real content")
        meta = {"artifact_hashes": {"file.bin": {"sha256": "wrong_hash_value"}}}
        p2_meta.write_text(json.dumps(meta))

        reader = Phase4ArtifactReader(
            project_root=root,
            phase4_dir=Path("data/phase4"),
            phase4_metadata=Path("data/phase4/m.json"),
            phase3_dir=Path("data/phase3"),
            phase3_metadata=Path("data/phase3/m.json"),
            phase2_dir=Path("data/phase2"),
            phase2_metadata=Path("data/phase2/meta.json"),
        )
        with pytest.raises(ValueError, match="SHA-256 mismatch"):
            reader.verify_all()

    def test_missing_artifact(self, tmp_path: Path) -> None:
        root = tmp_path / "project"
        p2_meta = root / "data" / "phase2" / "meta.json"
        p2_meta.parent.mkdir(parents=True, exist_ok=True)
        meta = {"artifact_hashes": {"missing.bin": {"sha256": "abc123"}}}
        p2_meta.write_text(json.dumps(meta))

        reader = Phase4ArtifactReader(
            project_root=root,
            phase4_dir=Path("data/phase4"),
            phase4_metadata=Path("data/phase4/m.json"),
            phase3_dir=Path("data/phase3"),
            phase3_metadata=Path("data/phase3/m.json"),
            phase2_dir=Path("data/phase2"),
            phase2_metadata=Path("data/phase2/meta.json"),
        )
        with pytest.raises(FileNotFoundError):
            reader.verify_all()

    def test_load_risk_report(self, tmp_path: Path) -> None:
        root = tmp_path / "project"
        p4_dir = root / "data" / "phase4"
        p4_dir.mkdir(parents=True, exist_ok=True)
        report = {
            "total_samples": 100,
            "risk_distribution": {"NORMAL": 80, "HIGH": 20},
            "sample_assessments": [],
        }
        (p4_dir / "risk_report.json").write_text(json.dumps(report))

        reader = Phase4ArtifactReader(
            project_root=root,
            phase4_dir=Path("data/phase4"),
            phase4_metadata=Path("data/phase4/m.json"),
            phase3_dir=Path("data/phase3"),
            phase3_metadata=Path("data/phase3/m.json"),
            phase2_dir=Path("data/phase2"),
            phase2_metadata=Path("data/phase2/m.json"),
        )
        result = reader.load_risk_report()
        assert result["total_samples"] == 100

    def test_load_baseline(self, tmp_path: Path) -> None:
        root = tmp_path / "project"
        p4_dir = root / "data" / "phase4"
        p4_dir.mkdir(parents=True, exist_ok=True)
        baseline = {
            "baseline_threshold": 0.204,
            "mad": 0.025,
            "median": 0.129,
        }
        (p4_dir / "baseline_config.json").write_text(json.dumps(baseline))

        reader = Phase4ArtifactReader(
            project_root=root,
            phase4_dir=Path("data/phase4"),
            phase4_metadata=Path("data/phase4/m.json"),
            phase3_dir=Path("data/phase3"),
            phase3_metadata=Path("data/phase3/m.json"),
            phase2_dir=Path("data/phase2"),
            phase2_metadata=Path("data/phase2/m.json"),
        )
        result = reader.load_baseline()
        assert result["baseline_threshold"] == 0.204

    def test_compute_sha256(self, tmp_path: Path) -> None:
        path = tmp_path / "test.bin"
        path.write_bytes(b"hello world")
        expected = hashlib.sha256(b"hello world").hexdigest()
        assert Phase4ArtifactReader._compute_sha256(path) == expected
