"""Unit tests for WaterfallVisualizer (BaseVisualizer)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.phase5_explanation_engine.phase5.base import BaseVisualizer
from src.phase5_explanation_engine.phase5.waterfall_visualizer import (
    WaterfallVisualizer,
)


class TestWaterfallVisualizer:
    def test_implements_base(self) -> None:
        assert issubclass(WaterfallVisualizer, BaseVisualizer)

    def test_plot_creates_file(self, tmp_path: Path) -> None:
        viz = WaterfallVisualizer(top_k=3)
        output = tmp_path / "waterfall.png"
        data = {
            "sample": {"sample_index": 42, "risk_level": "HIGH"},
            "shap_vals": np.random.randn(5, 4).astype(np.float32),
            "feature_names": ["f1", "f2", "f3", "f4"],
            "baseline_threshold": 0.2,
        }
        viz.plot(data, output)
        assert output.exists()

    def test_plot_file_nonzero(self, tmp_path: Path) -> None:
        viz = WaterfallVisualizer(top_k=3)
        output = tmp_path / "waterfall2.png"
        data = {
            "sample": {"sample_index": 1, "risk_level": "CRITICAL"},
            "shap_vals": np.random.randn(5, 4).astype(np.float32),
            "feature_names": ["a", "b", "c", "d"],
            "baseline_threshold": 0.15,
        }
        viz.plot(data, output)
        assert output.stat().st_size > 0

    def test_get_config(self) -> None:
        viz = WaterfallVisualizer(top_k=5, dpi=100)
        cfg = viz.get_config()
        assert cfg["type"] == "waterfall"
        assert cfg["top_k"] == 5
        assert cfg["dpi"] == 100

    def test_plot_with_negative_shap(self, tmp_path: Path) -> None:
        viz = WaterfallVisualizer(top_k=3)
        output = tmp_path / "waterfall_neg.png"
        shap_vals = np.array([[-1.0, 0.5, -0.3, 0.2]] * 5, dtype=np.float32)
        data = {
            "sample": {"sample_index": 10, "risk_level": "HIGH"},
            "shap_vals": shap_vals,
            "feature_names": ["f1", "f2", "f3", "f4"],
            "baseline_threshold": 0.2,
        }
        viz.plot(data, output)
        assert output.exists()
