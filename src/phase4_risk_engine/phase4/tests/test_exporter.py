"""Unit tests for RiskAdaptiveExporter (artifact export)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import pytest

from src.phase4_risk_engine.phase4.exporter import RiskAdaptiveExporter


@pytest.fixture()
def exporter(tmp_path: Path) -> RiskAdaptiveExporter:
    return RiskAdaptiveExporter(tmp_path)


def _make_baseline() -> Dict[str, Any]:
    return {
        "median": 0.128979,
        "mad": 0.024865,
        "baseline_threshold": 0.203575,
        "mad_multiplier": 3.0,
        "n_normal_samples": 9972,
        "n_attention_dims": 128,
        "computed_at": "2025-01-01T00:00:00+00:00",
    }


def _make_risk_results() -> List[Dict[str, Any]]:
    return [
        {
            "sample_index": 0,
            "anomaly_score": 0.15,
            "threshold": 0.20,
            "distance": -0.05,
            "risk_level": "NORMAL",
        },
        {
            "sample_index": 1,
            "anomaly_score": 0.30,
            "threshold": 0.20,
            "distance": 0.10,
            "risk_level": "HIGH",
        },
    ]


class TestRiskAdaptiveExporter:
    """Test artifact export methods."""

    def test_export_baseline(self, exporter: RiskAdaptiveExporter) -> None:
        baseline = _make_baseline()
        path = exporter.export_baseline(baseline, "baseline_config.json")

        assert path.exists()
        data = json.loads(path.read_text())
        assert data["median"] == 0.128979
        assert data["mad"] == 0.024865

    def test_export_threshold_config(self, exporter: RiskAdaptiveExporter) -> None:
        baseline = _make_baseline()
        window_log = [
            {
                "sample_index": 100,
                "hour": 0,
                "k_t": 2.5,
                "window_median": 0.17,
                "dynamic_threshold": 0.39,
            }
        ]
        config = {
            "k_schedule": [{"start_hour": 0, "end_hour": 6, "k": 2.5}],
            "window_size": 100,
        }
        path = exporter.export_threshold_config(
            baseline, window_log, config, "threshold_config.json"
        )

        assert path.exists()
        data = json.loads(path.read_text())
        assert data["baseline_threshold"] == 0.203575
        assert data["window_size"] == 100

    def test_export_risk_report(self, exporter: RiskAdaptiveExporter) -> None:
        path = exporter.export_risk_report(
            risk_results=_make_risk_results(),
            baseline=_make_baseline(),
            metrics={"accuracy": 0.83, "f1_score": 0.81},
            hw_info={"device": "CPU", "tensorflow": "2.20.0"},
            duration_s=3.5,
            git_commit="abc123",
            filename="risk_report.json",
        )

        assert path.exists()
        data = json.loads(path.read_text())
        assert data["total_samples"] == 2
        assert "sample_assessments" in data

    def test_export_drift_log(self, exporter: RiskAdaptiveExporter) -> None:
        events = [
            {
                "sample_index": 100,
                "drift_ratio": 0.94,
                "action": "FALLBACK_LOCKED",
                "dynamic_threshold": 0.40,
                "baseline_threshold": 0.20,
            }
        ]
        path = exporter.export_drift_log(events, "drift_log.csv")

        assert path.exists()
        df = pd.read_csv(path)
        assert len(df) == 1
        assert "action" in df.columns

    def test_export_drift_log_empty(self, exporter: RiskAdaptiveExporter) -> None:
        path = exporter.export_drift_log([], "drift_log.csv")
        assert path.exists()
        df = pd.read_csv(path)
        assert len(df) == 0
        assert "sample_index" in df.columns

    def test_build_risk_summary(self) -> None:
        results = _make_risk_results()
        summary = RiskAdaptiveExporter.build_risk_summary(results)
        assert summary["NORMAL"] == 1
        assert summary["HIGH"] == 1
        assert summary["CRITICAL"] == 0
