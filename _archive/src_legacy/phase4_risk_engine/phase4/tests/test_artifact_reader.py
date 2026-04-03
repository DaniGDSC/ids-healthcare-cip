"""Unit tests for Phase3ArtifactReader (SHA-256 verification)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.phase4_risk_engine.phase4.artifact_reader import Phase3ArtifactReader


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _create_phase3_artifacts(tmp_path: Path) -> Path:
    """Create fake Phase 3 artifacts with metadata."""
    phase3_dir = tmp_path / "phase3"
    phase3_dir.mkdir()

    # Fake weights file
    weights_path = phase3_dir / "classification_model.weights.h5"
    weights_path.write_bytes(b"fake_weights_data_phase3")

    # Fake metrics
    metrics_path = phase3_dir / "metrics_report.json"
    metrics_path.write_text(json.dumps({"metrics": {"accuracy": 0.83}}))

    # Metadata with hashes
    metadata = {
        "hyperparameters": {
            "dense_units": 64,
            "dense_activation": "relu",
            "head_dropout_rate": 0.3,
        },
        "artifact_hashes": {
            "classification_model.weights.h5": {
                "sha256": _sha256(weights_path),
                "algorithm": "SHA-256",
            },
            "metrics_report.json": {
                "sha256": _sha256(metrics_path),
                "algorithm": "SHA-256",
            },
        },
    }
    meta_path = phase3_dir / "classification_metadata.json"
    meta_path.write_text(json.dumps(metadata))
    return phase3_dir


def _create_phase2_artifacts(tmp_path: Path) -> Path:
    """Create fake Phase 2 artifacts with metadata."""
    phase2_dir = tmp_path / "phase2"
    phase2_dir.mkdir()

    # Fake weights
    weights_path = phase2_dir / "detection_model.weights.h5"
    weights_path.write_bytes(b"fake_weights_data_phase2")

    # Fake attention output
    attn_data = {f"attn_{i}": np.random.randn(10).tolist() for i in range(128)}
    attn_data["Label"] = [0] * 5 + [1] * 5
    attn_data["split"] = ["train"] * 5 + ["test"] * 5
    attn_df = pd.DataFrame(attn_data)
    attn_path = phase2_dir / "attention_output.parquet"
    attn_df.to_parquet(attn_path)

    metadata = {
        "hyperparameters": {
            "timesteps": 20,
            "stride": 1,
            "cnn_filters_1": 64,
            "cnn_filters_2": 128,
            "cnn_kernel_size": 3,
            "cnn_activation": "relu",
            "cnn_pool_size": 2,
            "bilstm_units_1": 128,
            "bilstm_units_2": 64,
            "dropout_rate": 0.3,
            "attention_units": 128,
        },
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


class TestPhase3ArtifactReader:
    """Test SHA-256 verification of Phase 3 artifacts."""

    def test_verify_phase3_success(self, tmp_path: Path) -> None:
        _create_phase3_artifacts(tmp_path)
        reader = Phase3ArtifactReader(
            project_root=tmp_path,
            phase3_dir=Path("phase3"),
            phase3_metadata=Path("phase3/classification_metadata.json"),
            phase2_dir=Path("phase2"),
            phase2_metadata=Path("phase2/detection_metadata.json"),
        )
        weights_path, metadata = reader.verify_phase3()
        assert weights_path.exists()
        assert "artifact_hashes" in metadata

    def test_sha256_mismatch(self, tmp_path: Path) -> None:
        _create_phase3_artifacts(tmp_path)
        # Tamper with metadata hash
        meta_path = tmp_path / "phase3" / "classification_metadata.json"
        metadata = json.loads(meta_path.read_text())
        first_key = list(metadata["artifact_hashes"].keys())[0]
        metadata["artifact_hashes"][first_key]["sha256"] = "deadbeef" * 8
        meta_path.write_text(json.dumps(metadata))

        reader = Phase3ArtifactReader(
            project_root=tmp_path,
            phase3_dir=Path("phase3"),
            phase3_metadata=Path("phase3/classification_metadata.json"),
            phase2_dir=Path("phase2"),
            phase2_metadata=Path("phase2/detection_metadata.json"),
        )
        with pytest.raises(ValueError, match="SHA-256 mismatch"):
            reader.verify_phase3()

    def test_missing_artifact(self, tmp_path: Path) -> None:
        _create_phase3_artifacts(tmp_path)
        # Delete the weights file
        (tmp_path / "phase3" / "classification_model.weights.h5").unlink()

        reader = Phase3ArtifactReader(
            project_root=tmp_path,
            phase3_dir=Path("phase3"),
            phase3_metadata=Path("phase3/classification_metadata.json"),
            phase2_dir=Path("phase2"),
            phase2_metadata=Path("phase2/detection_metadata.json"),
        )
        with pytest.raises(FileNotFoundError):
            reader.verify_phase3()

    def test_verify_phase2_success(self, tmp_path: Path) -> None:
        _create_phase2_artifacts(tmp_path)
        reader = Phase3ArtifactReader(
            project_root=tmp_path,
            phase3_dir=Path("phase3"),
            phase3_metadata=Path("phase3/classification_metadata.json"),
            phase2_dir=Path("phase2"),
            phase2_metadata=Path("phase2/detection_metadata.json"),
        )
        attn_path, metadata = reader.verify_phase2()
        assert "artifact_hashes" in metadata

    def test_load_attention_output(self, tmp_path: Path) -> None:
        _create_phase2_artifacts(tmp_path)
        reader = Phase3ArtifactReader(
            project_root=tmp_path,
            phase3_dir=Path("phase3"),
            phase3_metadata=Path("phase3/classification_metadata.json"),
            phase2_dir=Path("phase2"),
            phase2_metadata=Path("phase2/detection_metadata.json"),
        )
        attn_df = reader.load_attention_output()
        assert attn_df.shape[0] == 10
        assert "attn_0" in attn_df.columns
        assert "Label" in attn_df.columns

    def test_load_phase1_data(self, tmp_path: Path) -> None:
        # Create fake Phase 1 parquets
        n_train, n_test, n_feat = 20, 10, 5
        cols = [f"feat_{i}" for i in range(n_feat)] + ["Label"]
        train_df = pd.DataFrame(
            np.random.randn(n_train, n_feat + 1),
            columns=cols,
        )
        train_df["Label"] = np.random.randint(0, 2, n_train)
        test_df = pd.DataFrame(
            np.random.randn(n_test, n_feat + 1),
            columns=cols,
        )
        test_df["Label"] = np.random.randint(0, 2, n_test)

        train_path = tmp_path / "train.parquet"
        test_path = tmp_path / "test.parquet"
        train_df.to_parquet(train_path)
        test_df.to_parquet(test_path)

        reader = Phase3ArtifactReader(
            project_root=tmp_path,
            phase3_dir=Path("phase3"),
            phase3_metadata=Path("phase3/classification_metadata.json"),
            phase2_dir=Path("phase2"),
            phase2_metadata=Path("phase2/detection_metadata.json"),
        )
        X_train, y_train, X_test, y_test, feature_names = reader.load_phase1_data(
            train_path, test_path
        )
        assert X_train.shape == (n_train, n_feat)
        assert X_test.shape == (n_test, n_feat)
        assert len(feature_names) == n_feat
