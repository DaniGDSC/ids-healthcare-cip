"""Tests for Phase2ArtifactReader — SHA-256 verification and data loading."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.phase3_classification_engine.phase3.artifact_reader import Phase2ArtifactReader


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _create_phase2_artifacts(tmp_path: Path) -> Path:
    """Create minimal Phase 2 artifacts for testing."""
    phase2_dir = tmp_path / "phase2"
    phase2_dir.mkdir()

    # Create dummy weights file
    weights_path = phase2_dir / "detection_model.weights.h5"
    weights_path.write_bytes(b"dummy_weights_data")

    # Create dummy attention parquet
    attn_path = phase2_dir / "attention_output.parquet"
    df = pd.DataFrame({"attn_0": [1.0, 2.0], "Label": [0, 1]})
    df.to_parquet(attn_path, index=False)

    # Create metadata with correct hashes
    metadata = {
        "hyperparameters": {"timesteps": 20, "stride": 1},
        "artifact_hashes": {
            "detection_model.weights.h5": {
                "sha256": _sha256(weights_path),
                "algorithm": "SHA-256",
            },
            "attention_output.parquet": {
                "sha256": _sha256(attn_path),
                "algorithm": "SHA-256",
            },
        },
    }
    meta_path = phase2_dir / "detection_metadata.json"
    meta_path.write_text(json.dumps(metadata))

    return phase2_dir


class TestPhase2ArtifactReader:
    """Validate artifact loading and SHA-256 verification."""

    def test_load_and_verify_success(self, tmp_path: Path) -> None:
        _create_phase2_artifacts(tmp_path)
        reader = Phase2ArtifactReader(
            project_root=tmp_path,
            phase2_dir=Path("phase2"),
            metadata_file=Path("phase2/detection_metadata.json"),
        )
        weights_path, metadata = reader.load_and_verify()
        assert weights_path.exists()
        assert "artifact_hashes" in metadata

    def test_sha256_mismatch(self, tmp_path: Path) -> None:
        _create_phase2_artifacts(tmp_path)
        meta_path = tmp_path / "phase2" / "detection_metadata.json"
        meta = json.loads(meta_path.read_text())
        meta["artifact_hashes"]["detection_model.weights.h5"]["sha256"] = "bad_hash"
        meta_path.write_text(json.dumps(meta))

        reader = Phase2ArtifactReader(
            project_root=tmp_path,
            phase2_dir=Path("phase2"),
            metadata_file=Path("phase2/detection_metadata.json"),
        )
        with pytest.raises(ValueError, match="SHA-256 mismatch"):
            reader.load_and_verify()

    def test_missing_metadata(self, tmp_path: Path) -> None:
        reader = Phase2ArtifactReader(
            project_root=tmp_path,
            phase2_dir=Path("phase2"),
            metadata_file=Path("nonexistent.json"),
        )
        with pytest.raises(FileNotFoundError):
            reader.load_and_verify()

    def test_missing_artifact(self, tmp_path: Path) -> None:
        phase2_dir = tmp_path / "phase2"
        phase2_dir.mkdir()
        metadata = {
            "artifact_hashes": {
                "detection_model.weights.h5": {
                    "sha256": "abc123",
                    "algorithm": "SHA-256",
                },
            },
        }
        meta_path = phase2_dir / "detection_metadata.json"
        meta_path.write_text(json.dumps(metadata))

        reader = Phase2ArtifactReader(
            project_root=tmp_path,
            phase2_dir=Path("phase2"),
            metadata_file=Path("phase2/detection_metadata.json"),
        )
        with pytest.raises(FileNotFoundError):
            reader.load_and_verify()

    def test_load_phase1_data(self, tmp_path: Path) -> None:
        rng = np.random.RandomState(42)
        features = ["f1", "f2", "f3"]

        train_df = pd.DataFrame(rng.randn(50, 3).astype(np.float32), columns=features)
        train_df["Label"] = np.array([0] * 40 + [1] * 10)
        train_path = tmp_path / "train.parquet"
        train_df.to_parquet(train_path, index=False)

        test_df = pd.DataFrame(rng.randn(20, 3).astype(np.float32), columns=features)
        test_df["Label"] = np.array([0] * 15 + [1] * 5)
        test_path = tmp_path / "test.parquet"
        test_df.to_parquet(test_path, index=False)

        reader = Phase2ArtifactReader(
            project_root=tmp_path,
            phase2_dir=Path("phase2"),
            metadata_file=Path("meta.json"),
        )
        X_train, y_train, X_test, y_test, names = reader.load_phase1_data(train_path, test_path)
        assert X_train.shape == (50, 3)
        assert X_test.shape == (20, 3)
        assert y_train.shape == (50,)
        assert y_test.shape == (20,)
        assert names == features
