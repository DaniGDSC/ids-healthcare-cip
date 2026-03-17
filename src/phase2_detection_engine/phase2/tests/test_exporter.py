"""Unit tests for DetectionExporter (weights, parquet, JSON, report)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

from src.phase2_detection_engine.phase2.exporter import DetectionExporter


class TestDetectionExporter:
    """Test artifact export to tmp_path."""

    @pytest.fixture()
    def exporter(self, tmp_path: Path) -> DetectionExporter:
        return DetectionExporter(tmp_path)

    @pytest.fixture()
    def simple_model(self) -> tf.keras.Model:
        inp = tf.keras.Input(shape=(5, 3))
        x = tf.keras.layers.Dense(8)(inp)
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        return tf.keras.Model(inp, x)

    def test_export_model_weights(
        self, exporter: DetectionExporter, simple_model: tf.keras.Model,
        tmp_path: Path,
    ) -> None:
        path = exporter.export_model_weights(
            simple_model, "test.weights.h5",
        )
        assert path.exists()
        assert path.stat().st_size > 0

    def test_export_attention_vectors(
        self, exporter: DetectionExporter, tmp_path: Path,
    ) -> None:
        rng = np.random.RandomState(42)
        train_ctx = rng.randn(20, 8).astype(np.float32)
        test_ctx = rng.randn(10, 8).astype(np.float32)
        y_train = np.zeros(20, dtype=np.int32)
        y_test = np.ones(10, dtype=np.int32)

        path = exporter.export_attention_vectors(
            train_ctx, test_ctx, y_train, y_test, "attn.parquet",
        )
        assert path.exists()

        df = pd.read_parquet(path)
        # 20 train + 10 test = 30 rows
        assert len(df) == 30
        # 8 attn cols + Label + split = 10
        assert "Label" in df.columns
        assert "split" in df.columns
        assert "attn_0" in df.columns
        assert "attn_7" in df.columns
        # Split values
        assert set(df["split"].unique()) == {"train", "test"}

    def test_export_report_json(
        self, exporter: DetectionExporter, tmp_path: Path,
    ) -> None:
        report = {"phase": "test", "params": 1000}
        path = exporter.export_report(report, "report.json")
        assert path.exists()

        loaded = json.loads(path.read_text(encoding="utf-8"))
        assert loaded["phase"] == "test"
        assert loaded["params"] == 1000

    def test_build_report_keys(self, simple_model: tf.keras.Model) -> None:
        rng = np.random.RandomState(42)
        train_ctx = rng.randn(20, 8).astype(np.float32)
        test_ctx = rng.randn(10, 8).astype(np.float32)

        report = DetectionExporter.build_report(
            model=simple_model,
            config_dict={"timesteps": 5, "stride": 1},
            feature_names=["f1", "f2", "f3"],
            train_context=train_ctx,
            test_context=test_ctx,
            train_windows_shape=(20, 5, 3),
            test_windows_shape=(10, 5, 3),
            elapsed=1.23,
        )
        assert "phase" in report
        assert "architecture" in report
        assert "total_parameters" in report
        assert "layers" in report
        assert "hyperparameters" in report
        assert "environment" in report
        assert report["output_dim"] == 8
        assert report["n_features"] == 3
        assert report["elapsed_seconds"] == 1.23

    def test_build_report_environment(
        self, simple_model: tf.keras.Model,
    ) -> None:
        rng = np.random.RandomState(42)
        ctx = rng.randn(5, 8).astype(np.float32)
        report = DetectionExporter.build_report(
            model=simple_model,
            config_dict={},
            feature_names=["f1"],
            train_context=ctx,
            test_context=ctx,
            train_windows_shape=(5, 5, 1),
            test_windows_shape=(5, 5, 1),
            elapsed=0.5,
        )
        env = report["environment"]
        assert "python" in env
        assert "tensorflow" in env
        assert "numpy" in env
        assert "pandas" in env
