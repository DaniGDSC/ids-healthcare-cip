"""Unit tests for LineGraphVisualizer (BaseVisualizer)."""

from __future__ import annotations

from pathlib import Path

from src.phase5_explanation_engine.phase5.base import BaseVisualizer
from src.phase5_explanation_engine.phase5.line_graph_visualizer import (
    LineGraphVisualizer,
)


class TestLineGraphVisualizer:
    def test_implements_base(self) -> None:
        assert issubclass(LineGraphVisualizer, BaseVisualizer)

    def test_plot_creates_file(self, tmp_path: Path) -> None:
        viz = LineGraphVisualizer()
        output = tmp_path / "timeline.png"
        data = {
            "anomaly_scores": [0.1, 0.15, 0.3, 0.5, 0.4],
            "baseline_threshold": 0.2,
            "incident_id": 42,
        }
        viz.plot(data, output)
        assert output.exists()

    def test_plot_with_crossing(self, tmp_path: Path) -> None:
        viz = LineGraphVisualizer()
        output = tmp_path / "timeline_cross.png"
        data = {
            "anomaly_scores": [0.1, 0.15, 0.25, 0.4],
            "baseline_threshold": 0.2,
            "incident_id": 10,
        }
        viz.plot(data, output)
        assert output.exists()
        assert output.stat().st_size > 0

    def test_plot_without_crossing(self, tmp_path: Path) -> None:
        viz = LineGraphVisualizer()
        output = tmp_path / "timeline_no_cross.png"
        data = {
            "anomaly_scores": [0.05, 0.08, 0.1, 0.12],
            "baseline_threshold": 0.5,
            "incident_id": 99,
        }
        viz.plot(data, output)
        assert output.exists()

    def test_get_config(self) -> None:
        viz = LineGraphVisualizer(dpi=200, fig_width=12.0)
        cfg = viz.get_config()
        assert cfg["type"] == "line_graph"
        assert cfg["dpi"] == 200
        assert cfg["fig_width"] == 12.0
