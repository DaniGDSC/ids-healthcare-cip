"""Unit tests for ContextEnricher (per-sample SHAP context)."""

from __future__ import annotations

import numpy as np

from src.phase5_explanation_engine.phase5.config import ExplanationTemplates
from src.phase5_explanation_engine.phase5.context_enricher import (
    ContextEnricher,
)
from src.phase5_explanation_engine.phase5.explanation_generator import (
    ExplanationGenerator,
)


def _make_enricher() -> ContextEnricher:
    templates = ExplanationTemplates(
        CRITICAL="C: {idx}",
        HIGH="H: {idx} {f1}",
        MEDIUM="M: {idx} {f1}",
        LOW="L: {idx}",
    )
    gen = ExplanationGenerator(templates=templates)
    return ContextEnricher(explanation_generator=gen)


def _make_filtered_samples(n: int = 3) -> list:
    return [
        {
            "sample_index": i,
            "risk_level": "HIGH",
            "anomaly_score": 0.5 + i * 0.1,
            "threshold": 0.2,
        }
        for i in range(n)
    ]


class TestContextEnricher:
    def test_enrich_attaches_top_features(self) -> None:
        enricher = _make_enricher()
        shap = np.random.randn(3, 5, 4).astype(np.float32)
        names = ["f1", "f2", "f3", "f4"]
        enriched = enricher.enrich(_make_filtered_samples(), shap, names)
        assert "top_features" in enriched[0]
        assert len(enriched[0]["top_features"]) == 3

    def test_enrich_attaches_explanation(self) -> None:
        enricher = _make_enricher()
        shap = np.random.randn(3, 5, 4).astype(np.float32)
        names = ["f1", "f2", "f3", "f4"]
        enriched = enricher.enrich(_make_filtered_samples(), shap, names)
        assert "explanation" in enriched[0]
        assert len(enriched[0]["explanation"]) > 0

    def test_enrich_count_matches(self) -> None:
        enricher = _make_enricher()
        shap = np.random.randn(3, 5, 4).astype(np.float32)
        names = ["f1", "f2", "f3", "f4"]
        enriched = enricher.enrich(_make_filtered_samples(), shap, names)
        assert len(enriched) == 3

    def test_get_top3_features_count(self) -> None:
        shap = np.random.randn(5, 4).astype(np.float32)
        top3 = ContextEnricher.get_top3_features(shap, ["a", "b", "c", "d"])
        assert len(top3) == 3

    def test_get_top3_features_percentages(self) -> None:
        shap = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], dtype=np.float32)
        top3 = ContextEnricher.get_top3_features(shap, ["a", "b", "c"])
        total_pct = sum(f["contribution_pct"] for f in top3)
        assert abs(total_pct - 100.0) < 0.1

    def test_shap_values_boundary(self) -> None:
        """When fewer SHAP values than samples, enrichment truncates."""
        enricher = _make_enricher()
        shap = np.random.randn(1, 5, 4).astype(np.float32)
        names = ["f1", "f2", "f3", "f4"]
        enriched = enricher.enrich(_make_filtered_samples(n=5), shap, names)
        assert len(enriched) == 1

    def test_get_top3_features_zero_shap(self) -> None:
        """When all SHAP values are zero, percentages should still be valid."""
        shap = np.zeros((5, 3), dtype=np.float32)
        top3 = ContextEnricher.get_top3_features(shap, ["a", "b", "c"])
        assert len(top3) == 3
        # All contributions are zero, total forced to 1.0
        assert all(f["contribution_pct"] == 0.0 for f in top3)

    def test_get_config(self) -> None:
        enricher = _make_enricher()
        cfg = enricher.get_config()
        assert "generator" in cfg
