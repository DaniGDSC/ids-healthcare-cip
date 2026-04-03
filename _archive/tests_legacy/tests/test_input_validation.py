"""Tests for input validation across the risk pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from src.production.score_calibrator import ScoreCalibrator


class TestCalibratorValidation:
    def test_nan_score_returns_normal(self):
        cal = ScoreCalibrator()
        result = cal.calibrate(float("nan"))
        assert result["risk_level"] == "NORMAL"

    def test_inf_score_returns_normal(self):
        cal = ScoreCalibrator()
        result = cal.calibrate(float("inf"))
        assert result["risk_level"] == "NORMAL"

    def test_empty_benign_scores_no_crash(self):
        cal = ScoreCalibrator()
        cal.fit(np.array([]))
        assert not cal.fitted

    def test_nan_in_calibration_scores_filtered(self):
        cal = ScoreCalibrator()
        scores = np.array([0.9, float("nan"), 0.91, 0.92, float("nan")])
        cal.fit(scores)
        assert cal.fitted
        assert len(cal._calibration_scores) == 3

    def test_empty_two_class_no_crash(self):
        cal = ScoreCalibrator()
        cal.fit_two_class(np.array([]), np.array([]))
        assert not cal.fitted

    def test_percentile_division_by_zero(self):
        """Calibrator with empty scores doesn't crash on calibrate."""
        cal = ScoreCalibrator()
        cal._calibration_scores = np.array([])
        cal._fitted = True
        cal._use_two_class = False
        result = cal.calibrate(0.5)
        assert result["risk_level"] == "NORMAL"


class TestRiskScorerValidation:
    def test_mad_zero_returns_normal(self):
        from src.phase4_risk_engine.phase4.risk_scorer import RiskScorer
        from src.phase4_risk_engine.phase4.cross_modal import CrossModalFusionDetector

        scorer = RiskScorer(
            cross_modal=CrossModalFusionDetector(biometric_columns=[]),
        )
        from src.phase4_risk_engine.phase4.risk_level import RiskLevel
        result = scorer.classify_single(distance=0.5, mad=0.0)
        assert result == RiskLevel.NORMAL

    def test_mad_negative_returns_normal(self):
        from src.phase4_risk_engine.phase4.risk_scorer import RiskScorer
        from src.phase4_risk_engine.phase4.cross_modal import CrossModalFusionDetector
        from src.phase4_risk_engine.phase4.risk_level import RiskLevel

        scorer = RiskScorer(
            cross_modal=CrossModalFusionDetector(biometric_columns=[]),
        )
        result = scorer.classify_single(distance=0.5, mad=-1.0)
        assert result == RiskLevel.NORMAL


class TestCIAModifierValidation:
    def test_empty_cia_scores_no_crash(self):
        """CIA modifier handles edge case where cia_scores might be empty."""
        from src.phase4_risk_engine.phase4.cia_risk_modifier import CIARiskModifier
        from src.phase4_risk_engine.phase4.cia_threat_mapper import CIAThreatMapper
        from src.phase4_risk_engine.phase4.device_registry import DeviceRegistry
        from src.phase4_risk_engine.phase4.risk_level import RiskLevel

        modifier = CIARiskModifier(
            threat_mapper=CIAThreatMapper(),
            device_registry=DeviceRegistry(),
        )
        # "normal" attack category produces zero threat vector
        result = modifier.modify(
            base_risk=RiskLevel.LOW,
            attack_category="normal",
            device_id="generic_iomt_sensor",
        )
        assert result.adjusted_risk_level == RiskLevel.LOW
