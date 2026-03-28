"""Unit tests for RiskScorer (5-level classification)."""

from __future__ import annotations

import numpy as np

from src.phase4_risk_engine.phase4.cross_modal import CrossModalFusionDetector
from src.phase4_risk_engine.phase4.risk_level import RiskLevel
from src.phase4_risk_engine.phase4.risk_scorer import RiskScorer


class TestRiskScorer:
    """Test risk level classification logic."""

    def test_normal_risk(self) -> None:
        scorer = RiskScorer()
        level = scorer.classify_single(distance=-0.5, mad=0.025)
        assert level == RiskLevel.NORMAL

    def test_low_risk(self) -> None:
        scorer = RiskScorer(low_upper=0.5)
        # distance = 0.01, low_bound = 0.5 * 0.025 = 0.0125
        level = scorer.classify_single(distance=0.01, mad=0.025)
        assert level == RiskLevel.LOW

    def test_medium_risk(self) -> None:
        scorer = RiskScorer(low_upper=0.5, medium_upper=1.0)
        # distance = 0.02, low_bound=0.0125, medium_bound=0.025
        level = scorer.classify_single(distance=0.02, mad=0.025)
        assert level == RiskLevel.MEDIUM

    def test_high_risk(self) -> None:
        scorer = RiskScorer(low_upper=0.5, medium_upper=1.0, high_upper=2.0)
        # distance = 0.04, medium_bound=0.025, high_bound=0.05
        level = scorer.classify_single(distance=0.04, mad=0.025)
        assert level == RiskLevel.HIGH

    def test_critical_requires_cross_modal(self) -> None:
        cross_modal = CrossModalFusionDetector(
            biometric_columns=["Temp", "SpO2"],
            sigma_threshold=2.0,
        )
        scorer = RiskScorer(high_upper=2.0, cross_modal=cross_modal)
        # distance = 0.06 >= 2.0 * 0.025 = 0.05
        # Both bio and net > 2σ
        feature_values = np.array([3.0, 3.0, 3.0, 3.0])
        feature_names = ["Temp", "SpO2", "net1", "net2"]
        level = scorer.classify_single(
            distance=0.06,
            mad=0.025,
            feature_values=feature_values,
            feature_names=feature_names,
        )
        assert level == RiskLevel.CRITICAL

    def test_high_when_single_modal(self) -> None:
        cross_modal = CrossModalFusionDetector(
            biometric_columns=["Temp", "SpO2"],
            sigma_threshold=2.0,
        )
        scorer = RiskScorer(high_upper=2.0, cross_modal=cross_modal)
        # Only bio anomalous, net normal
        feature_values = np.array([3.0, 3.0, 0.5, 0.5])
        feature_names = ["Temp", "SpO2", "net1", "net2"]
        level = scorer.classify_single(
            distance=0.06,
            mad=0.025,
            feature_values=feature_values,
            feature_names=feature_names,
        )
        assert level == RiskLevel.HIGH

    def test_score_returns_list(self) -> None:
        scorer = RiskScorer()
        scores = np.array([0.1, 0.3, 0.5])
        thresholds = np.array([0.2, 0.2, 0.2])
        features = np.random.randn(3, 4).astype(np.float32)
        names = ["f1", "f2", "f3", "f4"]

        results = scorer.score(scores, thresholds, 0.025, features, names)
        assert len(results) == 3

    def test_result_dict_keys(self) -> None:
        scorer = RiskScorer()
        scores = np.array([0.5])
        thresholds = np.array([0.2])
        features = np.random.randn(1, 4).astype(np.float32)
        names = ["f1", "f2", "f3", "f4"]

        results = scorer.score(scores, thresholds, 0.025, features, names)
        expected_keys = {
            "sample_index",
            "anomaly_score",
            "threshold",
            "distance",
            "risk_level",
            "attention_flag",
        }
        assert set(results[0].keys()) == expected_keys
