"""Unit tests for Phase1ArtifactReader (SHA-256, missing files, shapes)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.phase2_detection_engine.phase2.artifact_reader import Phase1ArtifactReader


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65_536):
            h.update(chunk)
    return h.hexdigest()


def _create_artifacts(tmp_path: Path, n_train: int = 50, n_test: int = 20):
    """Create minimal Phase 1 artifacts for testing."""
    rng = np.random.RandomState(42)
    features = ["f1", "f2", "f3"]

    train_df = pd.DataFrame(rng.randn(n_train, 3), columns=features)
    train_df["Label"] = np.array([0] * (n_train - 10) + [1] * 10)
    train_path = tmp_path / "train.parquet"
    train_df.to_parquet(train_path, index=False)

    test_df = pd.DataFrame(rng.randn(n_test, 3), columns=features)
    test_df["Label"] = np.array([0] * (n_test - 5) + [1] * 5)
    test_path = tmp_path / "test.parquet"
    test_df.to_parquet(test_path, index=False)

    metadata = {
        "artifact_hashes": {
            "train_phase1.parquet": {"sha256": _sha256(train_path)},
            "test_phase1.parquet": {"sha256": _sha256(test_path)},
        }
    }
    meta_path = tmp_path / "metadata.json"
    meta_path.write_text(json.dumps(metadata), encoding="utf-8")

    report = {"output": {"feature_names": features}}
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    return train_path, test_path, meta_path, report_path, features


class TestPhase1ArtifactReader:
    """Test loading and SHA-256 verification."""

    def test_load_and_verify_shapes(self, tmp_path: Path) -> None:
        train_p, test_p, meta_p, rep_p, feats = _create_artifacts(tmp_path)
        reader = Phase1ArtifactReader(
            project_root=tmp_path,
            train_parquet=train_p.relative_to(tmp_path),
            test_parquet=test_p.relative_to(tmp_path),
            metadata_file=meta_p.relative_to(tmp_path),
            report_file=rep_p.relative_to(tmp_path),
        )
        X_train, y_train, X_test, y_test, names = reader.load_and_verify()
        assert X_train.shape == (50, 3)
        assert X_test.shape == (20, 3)
        assert y_train.shape == (50,)
        assert y_test.shape == (20,)
        assert names == feats

    def test_sha256_mismatch(self, tmp_path: Path) -> None:
        train_p, test_p, meta_p, rep_p, _ = _create_artifacts(tmp_path)
        # Corrupt hash
        meta = json.loads(meta_p.read_text())
        meta["artifact_hashes"]["train_phase1.parquet"]["sha256"] = "bad"
        meta_p.write_text(json.dumps(meta))

        reader = Phase1ArtifactReader(
            project_root=tmp_path,
            train_parquet=train_p.relative_to(tmp_path),
            test_parquet=test_p.relative_to(tmp_path),
            metadata_file=meta_p.relative_to(tmp_path),
            report_file=rep_p.relative_to(tmp_path),
        )
        with pytest.raises(ValueError, match="SHA-256"):
            reader.load_and_verify()

    def test_missing_file(self, tmp_path: Path) -> None:
        reader = Phase1ArtifactReader(
            project_root=tmp_path,
            train_parquet=Path("missing_train.parquet"),
            test_parquet=Path("missing_test.parquet"),
            metadata_file=Path("missing_meta.json"),
            report_file=Path("missing_report.json"),
        )
        with pytest.raises(FileNotFoundError):
            reader.load_and_verify()

    def test_feature_names_from_report(self, tmp_path: Path) -> None:
        _, _, _, _, feats = _create_artifacts(tmp_path, n_train=30, n_test=10)
        assert feats == ["f1", "f2", "f3"]

    def test_dtypes(self, tmp_path: Path) -> None:
        train_p, test_p, meta_p, rep_p, _ = _create_artifacts(tmp_path)
        reader = Phase1ArtifactReader(
            project_root=tmp_path,
            train_parquet=train_p.relative_to(tmp_path),
            test_parquet=test_p.relative_to(tmp_path),
            metadata_file=meta_p.relative_to(tmp_path),
            report_file=rep_p.relative_to(tmp_path),
        )
        X_train, y_train, _, _, _ = reader.load_and_verify()
        assert X_train.dtype == np.float32
        assert y_train.dtype == np.int32

    def test_compute_sha256_static(self, tmp_path: Path) -> None:
        path = tmp_path / "test.bin"
        path.write_bytes(b"hello world")
        expected = hashlib.sha256(b"hello world").hexdigest()
        assert Phase1ArtifactReader._compute_sha256(path) == expected
