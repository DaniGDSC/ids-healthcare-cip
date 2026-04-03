"""Unit tests for BarChartVisualizer (BaseVisualizer)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.phase5_explanation_engine.phase5.bar_chart_visualizer import (
    BarChartVisualizer,
)
from src.phase5_explanation_engine.phase5.base import BaseVisualizer


def _make_importance_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature": ["DIntPkt", "TotBytes", "SpO2", "Heart_rate"],
            "mean_abs_shap": [0.016, 0.006, 0.002, 0.001],
            "rank": [1, 2, 3, 4],
        }
    )


class TestBarChartVisualizer:
    def test_implements_base(self) -> None:
        assert issubclass(BarChartVisualizer, BaseVisualizer)

    def test_plot_creates_file(self, tmp_path: Path) -> None:
        viz = BarChartVisualizer(top_k=3)
        output = tmp_path / "bar.png"
        viz.plot({"importance_df": _make_importance_df()}, output)
        assert output.exists()

    def test_biometric_color_coding(self, tmp_path: Path) -> None:
        viz = BarChartVisualizer(
            top_k=4,
            biometric_columns=frozenset(["SpO2", "Heart_rate"]),
        )
        output = tmp_path / "bar_bio.png"
        viz.plot({"importance_df": _make_importance_df()}, output)
        assert output.exists()
        assert output.stat().st_size > 0

    def test_get_config(self) -> None:
        viz = BarChartVisualizer(
            top_k=5,
            biometric_columns=frozenset(["SpO2"]),
            dpi=200,
        )
        cfg = viz.get_config()
        assert cfg["type"] == "bar_chart"
        assert cfg["top_k"] == 5
        assert cfg["dpi"] == 200
        assert "SpO2" in cfg["biometric_columns"]

    def test_plot_with_empty_biometrics(self, tmp_path: Path) -> None:
        viz = BarChartVisualizer(top_k=3, biometric_columns=frozenset())
        output = tmp_path / "bar_no_bio.png"
        viz.plot({"importance_df": _make_importance_df()}, output)
        assert output.exists()
