"""Unit tests for FeatureImportanceRanker (mean |SHAP| ranking)."""

from __future__ import annotations

import numpy as np

from src.phase5_explanation_engine.phase5.feature_importance import (
    FeatureImportanceRanker,
)


def _make_shap_values(n: int = 10, t: int = 5, f: int = 4) -> np.ndarray:
    """Create synthetic SHAP values."""
    np.random.seed(42)
    return np.random.randn(n, t, f).astype(np.float32)


class TestFeatureImportanceRanker:
    def test_rank_output_shape(self) -> None:
        ranker = FeatureImportanceRanker(top_k=3)
        shap = _make_shap_values(n=10, t=5, f=4)
        names = ["f1", "f2", "f3", "f4"]
        df, top = ranker.rank(shap, names)
        assert len(df) == 4
        assert "feature" in df.columns
        assert "mean_abs_shap" in df.columns
        assert "rank" in df.columns

    def test_rank_ordering(self) -> None:
        ranker = FeatureImportanceRanker(top_k=2)
        # Feature 0 has much larger values
        shap = np.zeros((5, 3, 3), dtype=np.float32)
        shap[:, :, 0] = 10.0
        shap[:, :, 1] = 1.0
        shap[:, :, 2] = 0.1
        names = ["big", "medium", "small"]
        df, top = ranker.rank(shap, names)
        assert top[0][0] == "big"
        assert top[0][1] > top[1][1]

    def test_top_k_length(self) -> None:
        ranker = FeatureImportanceRanker(top_k=2)
        shap = _make_shap_values(n=10, t=5, f=4)
        _, top = ranker.rank(shap, ["a", "b", "c", "d"])
        assert len(top) == 2

    def test_rank_column_values(self) -> None:
        ranker = FeatureImportanceRanker(top_k=4)
        shap = _make_shap_values(n=5, t=3, f=4)
        df, _ = ranker.rank(shap, ["a", "b", "c", "d"])
        assert list(df["rank"]) == [1, 2, 3, 4]

    def test_single_feature(self) -> None:
        ranker = FeatureImportanceRanker(top_k=5)
        shap = np.ones((3, 2, 1), dtype=np.float32)
        df, top = ranker.rank(shap, ["only_feature"])
        assert len(df) == 1
        assert top[0][0] == "only_feature"

    def test_get_config(self) -> None:
        ranker = FeatureImportanceRanker(top_k=7)
        assert ranker.get_config() == {"top_k": 7}
