"""Unit tests for ExplanationGenerator (template-based explanations)."""

from __future__ import annotations

from src.phase5_explanation_engine.phase5.config import ExplanationTemplates
from src.phase5_explanation_engine.phase5.explanation_generator import (
    ExplanationGenerator,
)


def _make_templates() -> ExplanationTemplates:
    return ExplanationTemplates(
        CRITICAL=(
            "CRITICAL ALERT: Sample {idx} at T={time}. "
            "Top factors: {f1}={v1:.4f} ({p1:.1f}%), "
            "{f2}={v2:.4f} ({p2:.1f}%), "
            "{f3}={v3:.4f} ({p3:.1f}%). "
            "Immediate action required."
        ),
        HIGH=(
            "HIGH ALERT: Suspicious activity detected at sample {idx}. "
            "Primary indicator: {f1} contributing {p1:.1f}%."
        ),
        MEDIUM=("MEDIUM: Anomaly detected at sample {idx}. " "Monitor closely. Key feature: {f1}."),
        LOW="LOW: Minor anomaly at sample {idx}. No immediate action needed.",
    )


def _make_top3():
    return [
        {"feature": "DIntPkt", "shap_value": 0.05, "contribution_pct": 50.0},
        {"feature": "TotBytes", "shap_value": 0.02, "contribution_pct": 20.0},
        {"feature": "SpO2", "shap_value": 0.01, "contribution_pct": 10.0},
    ]


class TestExplanationGenerator:
    def test_critical_with_3_features(self) -> None:
        gen = ExplanationGenerator(templates=_make_templates())
        result = gen.generate("CRITICAL", 42, _make_top3())
        assert "CRITICAL ALERT" in result
        assert "DIntPkt" in result
        assert "TotBytes" in result
        assert "SpO2" in result
        assert "42" in result

    def test_high_with_1_feature(self) -> None:
        gen = ExplanationGenerator(templates=_make_templates())
        result = gen.generate("HIGH", 10, _make_top3())
        assert "HIGH ALERT" in result
        assert "DIntPkt" in result
        assert "50.0%" in result

    def test_medium_with_1_feature(self) -> None:
        gen = ExplanationGenerator(templates=_make_templates())
        result = gen.generate("MEDIUM", 5, _make_top3())
        assert "MEDIUM" in result
        assert "DIntPkt" in result

    def test_low_no_features(self) -> None:
        gen = ExplanationGenerator(templates=_make_templates())
        result = gen.generate("LOW", 99, [])
        assert "LOW" in result
        assert "99" in result

    def test_unknown_level_falls_back(self) -> None:
        gen = ExplanationGenerator(templates=_make_templates())
        result = gen.generate("UNKNOWN", 1, [])
        # Falls back to LOW template
        assert "1" in result

    def test_get_config(self) -> None:
        gen = ExplanationGenerator(templates=_make_templates())
        cfg = gen.get_config()
        assert "template_levels" in cfg
        assert set(cfg["template_levels"]) == {"CRITICAL", "HIGH", "MEDIUM", "LOW"}
