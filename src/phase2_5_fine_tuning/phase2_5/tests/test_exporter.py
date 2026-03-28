"""Tests for TuningExporter."""

from __future__ import annotations

import json
from pathlib import Path

from src.phase2_5_fine_tuning.phase2_5.exporter import TuningExporter


class TestTuningExporter:
    def test_export_tuning_results(self, tmp_path: Path) -> None:
        exporter = TuningExporter(tmp_path)
        path = exporter.export_tuning_results({"best_score": 0.95}, "tuning.json")
        assert path.exists()
        assert json.loads(path.read_text())["best_score"] == 0.95

    def test_export_best_config(self, tmp_path: Path) -> None:
        exporter = TuningExporter(tmp_path)
        path = exporter.export_best_config({"head_lr": 0.003}, "best.json")
        assert json.loads(path.read_text())["head_lr"] == 0.003

    def test_build_report_structure(self) -> None:
        report = TuningExporter.build_report(
            tuning_results={"strategy": "bayesian_tpe", "metric": "attack_f1",
                            "total_trials": 10, "completed_trials": 8,
                            "best_score": 0.45, "best_config": {"head_lr": 0.003}},
            ablation_results={"variants": [{"name": "v1"}], "comparison": []},
            importance_results={"method": "optuna_fanova", "importances": {"head_lr": 0.65}},
            multi_seed_results={"enabled": False, "configs": []},
            hw_info={"device": "CPU"},
            duration_s=120.5,
            git_commit="abc123",
        )

        assert report["pipeline"] == "phase2_5_fine_tuning"
        assert report["tuning_summary"]["strategy"] == "bayesian_tpe"
        assert report["tuning_summary"]["best_score"] == 0.45
        assert report["importance_summary"]["method"] == "optuna_fanova"
