"""Tests for TuningExporter — export tuning and ablation artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from src.phase2_5_fine_tuning.phase2_5.exporter import TuningExporter


class TestTuningExporter:
    """Validate TuningExporter export methods."""

    def test_export_tuning_results(self, tmp_path: Path) -> None:
        exporter = TuningExporter(tmp_path)
        data = {"best_score": 0.95, "trials": []}
        path = exporter.export_tuning_results(data, "tuning.json")
        assert path.exists()
        loaded = json.loads(path.read_text())
        assert loaded["best_score"] == 0.95

    def test_export_ablation_results(self, tmp_path: Path) -> None:
        exporter = TuningExporter(tmp_path)
        data = {"baseline": {"f1": 0.9}, "variants": []}
        path = exporter.export_ablation_results(data, "ablation.json")
        assert path.exists()
        loaded = json.loads(path.read_text())
        assert loaded["baseline"]["f1"] == 0.9

    def test_export_best_config(self, tmp_path: Path) -> None:
        exporter = TuningExporter(tmp_path)
        config = {"cnn_filters_1": 64, "dropout_rate": 0.3}
        path = exporter.export_best_config(config, "best.json")
        assert path.exists()
        loaded = json.loads(path.read_text())
        assert loaded["cnn_filters_1"] == 64

    def test_export_report(self, tmp_path: Path) -> None:
        exporter = TuningExporter(tmp_path)
        report = {"pipeline": "phase2_5", "duration_seconds": 100}
        path = exporter.export_report(report, "report.json")
        assert path.exists()

    def test_build_report_structure(self) -> None:
        report = TuningExporter.build_report(
            tuning_results={
                "strategy": "random",
                "metric": "f1_score",
                "total_trials": 10,
                "completed_trials": 8,
                "best_score": 0.95,
                "best_config": {"lr": 0.001},
            },
            ablation_results={
                "variants": [{"name": "v1"}, {"name": "v2"}],
                "comparison": [],
            },
            hw_info={"device": "CPU", "tensorflow": "2.20.0"},
            duration_s=120.5,
            git_commit="abc123",
        )

        assert report["pipeline"] == "phase2_5_fine_tuning"
        assert report["tuning_summary"]["strategy"] == "random"
        assert report["tuning_summary"]["best_score"] == 0.95
        assert report["ablation_summary"]["n_variants"] == 2
