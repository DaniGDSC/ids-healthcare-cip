"""Unit tests for ExplanationExporter (artifact export)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.phase5_explanation_engine.phase5.exporter import (
    ExplanationExporter,
)


class TestExplanationExporter:
    def test_export_shap_values(self, tmp_path: Path) -> None:
        exporter = ExplanationExporter(output_dir=tmp_path)
        shap = np.random.randn(5, 3, 4).astype(np.float32)
        names = ["f1", "f2", "f3", "f4"]
        path, sha = exporter.export_shap_values(shap, names, "shap.parquet")
        assert path.exists()
        df = pd.read_parquet(path)
        assert df.shape == (5, 4)
        assert len(sha) == 64  # SHA-256 hex

    def test_export_explanation_report(self, tmp_path: Path) -> None:
        exporter = ExplanationExporter(output_dir=tmp_path)
        enriched = [
            {
                "sample_index": 0,
                "risk_level": "HIGH",
                "explanation": "test",
            }
        ]
        importance_df = pd.DataFrame(
            {
                "feature": ["f1", "f2"],
                "mean_abs_shap": [0.1, 0.05],
                "rank": [1, 2],
            }
        )
        path, sha = exporter.export_explanation_report(
            enriched, importance_df, {"HIGH": 1}, 2, "abc123", "report.json"
        )
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["total_explained"] == 1
        assert len(sha) == 64

    def test_export_metadata(self, tmp_path: Path) -> None:
        exporter = ExplanationExporter(output_dir=tmp_path)
        path = exporter.export_metadata(
            enriched_samples=[{"sample_index": 0}],
            level_counts={"HIGH": 1},
            chart_files=["chart.png"],
            hw_info={"device": "CPU"},
            duration_s=1.5,
            git_commit="abc123",
            artifact_hashes={"shap.parquet": "sha_hash"},
            background_samples=100,
            filename="meta.json",
        )
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["samples_explained"] == 1
        assert "artifact_hashes" in data

    def test_compute_sha256(self, tmp_path: Path) -> None:
        path = tmp_path / "test.bin"
        path.write_bytes(b"test data")
        expected = hashlib.sha256(b"test data").hexdigest()
        assert ExplanationExporter._compute_sha256(path) == expected

    def test_mkdir_parents(self, tmp_path: Path) -> None:
        nested = tmp_path / "a" / "b" / "c"
        exporter = ExplanationExporter(output_dir=nested)
        shap = np.random.randn(2, 3, 2).astype(np.float32)
        path, _ = exporter.export_shap_values(shap, ["x", "y"], "shap.parquet")
        assert path.exists()

    def test_get_config(self, tmp_path: Path) -> None:
        exporter = ExplanationExporter(output_dir=tmp_path)
        cfg = exporter.get_config()
        assert "output_dir" in cfg
