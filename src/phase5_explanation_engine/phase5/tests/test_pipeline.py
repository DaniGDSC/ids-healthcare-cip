"""Unit tests for ExplanationPipeline utility functions and chart generation."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from src.phase5_explanation_engine.phase5.config import (
    ExplanationTemplates,
    Phase5Config,
)
from src.phase5_explanation_engine.phase5.pipeline import (
    ExplanationPipeline,
    _detect_hardware,
    _get_git_commit,
)


def _make_config() -> Phase5Config:
    return Phase5Config(
        phase4_dir=Path("data/phase4"),
        phase4_metadata=Path("data/phase4/m.json"),
        phase3_dir=Path("data/phase3"),
        phase3_metadata=Path("data/phase3/m.json"),
        phase2_dir=Path("data/phase2"),
        phase2_metadata=Path("data/phase2/m.json"),
        phase1_train=Path("data/processed/train.parquet"),
        phase1_test=Path("data/processed/test.parquet"),
        explanation_templates=ExplanationTemplates(CRITICAL="C", HIGH="H", MEDIUM="M", LOW="L"),
        biometric_columns=["SpO2", "Heart_rate"],
        max_waterfall_charts=2,
        max_timeline_charts=1,
    )


class TestGetGitCommit:
    def test_returns_string(self) -> None:
        result = _get_git_commit()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_returns_unknown_on_failure(self) -> None:
        with patch(
            "src.phase5_explanation_engine.phase5.pipeline.subprocess.run",
            side_effect=FileNotFoundError,
        ):
            assert _get_git_commit() == "unknown"


class TestDetectHardware:
    def test_returns_dict(self) -> None:
        hw = _detect_hardware()
        assert isinstance(hw, dict)
        assert "device" in hw
        assert "tensorflow" in hw
        assert "python" in hw
        assert "platform" in hw

    def test_has_device_info(self) -> None:
        hw = _detect_hardware()
        assert "CPU" in hw["device"] or "GPU" in hw["device"]


class TestExplanationPipelineConstruction:
    def test_constructor_stores_components(self) -> None:
        config = _make_config()
        reader = MagicMock()
        alert_filter = MagicMock()
        shap_computer = MagicMock()
        feature_ranker = MagicMock()
        context_enricher = MagicMock()
        waterfall_viz = MagicMock()
        bar_chart_viz = MagicMock()
        line_graph_viz = MagicMock()
        exporter = MagicMock()

        pipeline = ExplanationPipeline(
            config=config,
            artifact_reader=reader,
            alert_filter=alert_filter,
            shap_computer=shap_computer,
            feature_ranker=feature_ranker,
            context_enricher=context_enricher,
            waterfall_viz=waterfall_viz,
            bar_chart_viz=bar_chart_viz,
            line_graph_viz=line_graph_viz,
            exporter=exporter,
            project_root=Path("/tmp"),
        )
        assert pipeline._config is config
        assert pipeline._reader is reader
        assert pipeline._filter is alert_filter


class TestGenerateAllCharts:
    def test_generates_charts(self, tmp_path: Path) -> None:
        config = _make_config()
        waterfall = MagicMock()
        bar_chart = MagicMock()
        timeline = MagicMock()

        pipeline = ExplanationPipeline(
            config=config,
            artifact_reader=MagicMock(),
            alert_filter=MagicMock(),
            shap_computer=MagicMock(),
            feature_ranker=MagicMock(),
            context_enricher=MagicMock(),
            waterfall_viz=waterfall,
            bar_chart_viz=bar_chart,
            line_graph_viz=timeline,
            exporter=MagicMock(),
            project_root=Path("/tmp"),
        )

        enriched = [
            {
                "sample_index": 0,
                "risk_level": "HIGH",
                "anomaly_score": 0.8,
                "threshold": 0.2,
            },
            {
                "sample_index": 1,
                "risk_level": "LOW",
                "anomaly_score": 0.3,
                "threshold": 0.2,
            },
        ]
        shap_vals = np.random.randn(2, 5, 4).astype(np.float32)
        importance_df = pd.DataFrame({"feature": ["a", "b"], "mean_abs_shap": [0.1, 0.05]})

        charts_dir = tmp_path / "charts"
        chart_files = pipeline._generate_all_charts(
            enriched, shap_vals, ["a", "b", "c", "d"], importance_df, 0.2, charts_dir
        )

        assert "feature_importance.png" in chart_files
        bar_chart.plot.assert_called_once()
        waterfall.plot.assert_called_once()  # 1 HIGH sample
        timeline.plot.assert_called_once()  # max_timeline_charts=1

    def test_no_crit_high_skips_waterfall(self, tmp_path: Path) -> None:
        config = _make_config()
        waterfall = MagicMock()
        bar_chart = MagicMock()
        timeline = MagicMock()

        pipeline = ExplanationPipeline(
            config=config,
            artifact_reader=MagicMock(),
            alert_filter=MagicMock(),
            shap_computer=MagicMock(),
            feature_ranker=MagicMock(),
            context_enricher=MagicMock(),
            waterfall_viz=waterfall,
            bar_chart_viz=bar_chart,
            line_graph_viz=timeline,
            exporter=MagicMock(),
            project_root=Path("/tmp"),
        )

        enriched = [
            {
                "sample_index": 0,
                "risk_level": "LOW",
                "anomaly_score": 0.3,
                "threshold": 0.2,
            },
        ]
        shap_vals = np.random.randn(1, 5, 4).astype(np.float32)
        importance_df = pd.DataFrame({"feature": ["a"], "mean_abs_shap": [0.1]})

        charts_dir = tmp_path / "charts2"
        chart_files = pipeline._generate_all_charts(
            enriched, shap_vals, ["a", "b", "c", "d"], importance_df, 0.2, charts_dir
        )

        assert "feature_importance.png" in chart_files
        waterfall.plot.assert_not_called()
        timeline.plot.assert_not_called()
