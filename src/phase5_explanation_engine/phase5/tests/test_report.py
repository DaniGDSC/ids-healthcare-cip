"""Unit tests for render_explanation_report (enhanced §8.1 markdown)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.phase5_explanation_engine.phase5.config import (
    ExplanationTemplates,
    Phase5Config,
)
from src.phase5_explanation_engine.phase5.report import (
    render_explanation_report,
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
    )


def _make_enriched_samples():
    return [
        {
            "sample_index": 1,
            "risk_level": "HIGH",
            "anomaly_score": 0.5,
            "threshold": 0.2,
            "timestamp": "T=1",
            "top_features": [{"feature": "DIntPkt", "shap_value": 0.05, "contribution_pct": 50.0}],
            "explanation": "HIGH ALERT: test explanation for sample 1",
        },
        {
            "sample_index": 2,
            "risk_level": "LOW",
            "anomaly_score": 0.3,
            "threshold": 0.2,
            "timestamp": "T=2",
            "top_features": [],
            "explanation": "LOW: minor anomaly",
        },
    ]


def _make_importance_df():
    return pd.DataFrame(
        {
            "feature": [
                "DIntPkt",
                "TotBytes",
                "SpO2",
                "Heart_rate",
                "SrcBytes",
            ],
            "mean_abs_shap": [0.016, 0.006, 0.002, 0.001, 0.0005],
            "rank": [1, 2, 3, 4, 5],
        }
    )


def _render() -> str:
    return render_explanation_report(
        enriched_samples=_make_enriched_samples(),
        importance_df=_make_importance_df(),
        level_counts={"LOW": 1, "MEDIUM": 0, "HIGH": 1, "CRITICAL": 0},
        chart_files=[
            "feature_importance.png",
            "waterfall_1.png",
            "timeline_1.png",
        ],
        baseline_threshold=0.204,
        hw_info={"device": "CPU: x86_64", "tensorflow": "2.20.0"},
        duration_s=5.5,
        git_commit="abc123def456",
        config=_make_config(),
    )


class TestRenderExplanationReport:
    def test_contains_shap_methodology(self) -> None:
        report = _render()
        assert "8.1.5" in report
        assert "GradientExplainer" in report

    def test_contains_feature_importance_table(self) -> None:
        report = _render()
        assert "| Rank | Feature |" in report
        assert "DIntPkt" in report

    def test_contains_explanation_examples(self) -> None:
        report = _render()
        assert "| Level | Sample | Explanation |" in report

    def test_contains_shap_justification(self) -> None:
        report = _render()
        assert "Method Justification" in report
        assert "KernelExplainer" in report

    def test_contains_human_centric_section(self) -> None:
        report = _render()
        assert "8.1.7" in report
        assert "Human-Centric" in report

    def test_contains_biometric_interpretation(self) -> None:
        report = _render()
        assert "Feature Interpretation" in report
        assert "biometric" in report.lower()

    def test_contains_execution_details(self) -> None:
        report = _render()
        assert "5.50s" in report
        assert "abc123def456"[:12] in report

    def test_biometric_dominated(self) -> None:
        """When biometric features dominate importance, report reflects it."""
        bio_config = Phase5Config(
            phase4_dir=Path("data/phase4"),
            phase4_metadata=Path("data/phase4/m.json"),
            phase3_dir=Path("data/phase3"),
            phase3_metadata=Path("data/phase3/m.json"),
            phase2_dir=Path("data/phase2"),
            phase2_metadata=Path("data/phase2/m.json"),
            phase1_train=Path("data/processed/train.parquet"),
            phase1_test=Path("data/processed/test.parquet"),
            explanation_templates=ExplanationTemplates(CRITICAL="C", HIGH="H", MEDIUM="M", LOW="L"),
            biometric_columns=["SpO2", "Heart_rate", "Temp", "SYS", "DIA"],
        )
        bio_importance = pd.DataFrame(
            {
                "feature": ["SpO2", "Heart_rate", "Temp", "SYS", "DIntPkt"],
                "mean_abs_shap": [0.02, 0.015, 0.01, 0.008, 0.005],
                "rank": [1, 2, 3, 4, 5],
            }
        )
        report = render_explanation_report(
            enriched_samples=_make_enriched_samples(),
            importance_df=bio_importance,
            level_counts={"LOW": 1, "MEDIUM": 0, "HIGH": 1, "CRITICAL": 0},
            chart_files=[],
            baseline_threshold=0.2,
            hw_info={"device": "CPU: x86_64", "tensorflow": "2.20.0"},
            duration_s=1.0,
            git_commit="abc",
            config=bio_config,
        )
        assert "Biometric features dominate" in report

    def test_equal_modalities(self) -> None:
        """When biometric and network counts are equal."""
        eq_config = Phase5Config(
            phase4_dir=Path("data/phase4"),
            phase4_metadata=Path("data/phase4/m.json"),
            phase3_dir=Path("data/phase3"),
            phase3_metadata=Path("data/phase3/m.json"),
            phase2_dir=Path("data/phase2"),
            phase2_metadata=Path("data/phase2/m.json"),
            phase1_train=Path("data/processed/train.parquet"),
            phase1_test=Path("data/processed/test.parquet"),
            explanation_templates=ExplanationTemplates(CRITICAL="C", HIGH="H", MEDIUM="M", LOW="L"),
            biometric_columns=["SpO2"],
        )
        eq_importance = pd.DataFrame(
            {
                "feature": ["SpO2", "DIntPkt"],
                "mean_abs_shap": [0.02, 0.015],
                "rank": [1, 2],
            }
        )
        report = render_explanation_report(
            enriched_samples=_make_enriched_samples(),
            importance_df=eq_importance,
            level_counts={"LOW": 1, "MEDIUM": 0, "HIGH": 1, "CRITICAL": 0},
            chart_files=[],
            baseline_threshold=0.2,
            hw_info={"device": "CPU: x86_64", "tensorflow": "2.20.0"},
            duration_s=1.0,
            git_commit="abc",
            config=eq_config,
        )
        assert "equally" in report
