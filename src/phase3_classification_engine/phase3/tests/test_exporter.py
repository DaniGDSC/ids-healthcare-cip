"""Tests for ClassificationExporter — artifact export verification."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import tensorflow as tf

from src.phase3_classification_engine.phase3.exporter import ClassificationExporter


@pytest.fixture()
def exporter(tmp_path: Path) -> ClassificationExporter:
    return ClassificationExporter(tmp_path)


@pytest.fixture()
def simple_model() -> tf.keras.Model:
    inp = tf.keras.Input(shape=(5,))
    x = tf.keras.layers.Dense(1, activation="sigmoid")(inp)
    return tf.keras.Model(inp, x)


class TestClassificationExporter:
    """Validate artifact export methods."""

    def test_export_model_weights(
        self,
        exporter: ClassificationExporter,
        simple_model: tf.keras.Model,
        tmp_path: Path,
    ) -> None:
        path = exporter.export_model_weights(simple_model, "test.weights.h5")
        assert path.exists()
        assert path.stat().st_size > 0

    def test_export_metrics(self, exporter: ClassificationExporter, tmp_path: Path) -> None:
        metrics_report = {
            "pipeline": "test",
            "metrics": {"accuracy": 0.95},
        }
        path = exporter.export_metrics(metrics_report, "metrics.json")
        assert path.exists()
        loaded = json.loads(path.read_text())
        assert loaded["metrics"]["accuracy"] == 0.95

    def test_export_confusion_matrix(
        self, exporter: ClassificationExporter, tmp_path: Path
    ) -> None:
        cm = [[90, 10], [5, 95]]
        labels = ["Normal", "Attack"]
        path = exporter.export_confusion_matrix(cm, labels, "cm.csv")
        assert path.exists()
        df = pd.read_csv(path, index_col=0)
        assert df.shape == (2, 2)
        assert list(df.columns) == ["Normal", "Attack"]

    def test_export_history(self, exporter: ClassificationExporter, tmp_path: Path) -> None:
        histories = [
            {"phase": "A", "epochs_run": 5, "final_val_loss": 0.1},
            {"phase": "B", "epochs_run": 3, "final_val_loss": 0.05},
        ]
        path = exporter.export_history(histories, "history.json")
        assert path.exists()
        loaded = json.loads(path.read_text())
        assert len(loaded) == 2
        assert loaded[0]["phase"] == "A"

    def test_build_metrics_report(self, simple_model: tf.keras.Model) -> None:
        metrics = {"accuracy": 0.9}
        hw_info = {
            "device": "CPU",
            "tensorflow": "2.20.0",
            "cuda": "N/A",
            "python": "3.12",
            "platform": "Linux",
        }
        report = ClassificationExporter.build_metrics_report(
            metrics, simple_model, hw_info, 10.5, "abc123"
        )
        assert report["pipeline"] == "phase3_classification"
        assert report["git_commit"] == "abc123"
        assert report["duration_seconds"] == 10.5
        assert report["metrics"]["accuracy"] == 0.9
