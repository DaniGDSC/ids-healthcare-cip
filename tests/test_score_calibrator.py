"""Unit tests for ScoreCalibrator (two-class and benign-only modes)."""

from __future__ import annotations

import numpy as np
import pytest

from src.production.score_calibrator import ScoreCalibrator


@pytest.fixture
def synthetic_two_class():
    """Synthetic benign N(0.90, 0.02) + attack N(0.95, 0.02)."""
    rng = np.random.RandomState(42)
    benign = rng.normal(0.90, 0.02, size=150)
    attack = rng.normal(0.95, 0.02, size=50)
    scores = np.clip(np.concatenate([benign, attack]), 0.01, 0.99)
    labels = np.concatenate([np.zeros(150), np.ones(50)]).astype(int)
    return scores, labels


class TestTwoClassFit:
    def test_fit_basic(self, synthetic_two_class):
        scores, labels = synthetic_two_class
        cal = ScoreCalibrator()
        cal.fit_two_class(scores, labels)
        assert cal.fitted
        assert cal._use_two_class

    def test_youden_j_positive(self, synthetic_two_class):
        scores, labels = synthetic_two_class
        cal = ScoreCalibrator()
        cal.fit_two_class(scores, labels)
        assert cal._youden_j > 0.3, f"Youden's J too low: {cal._youden_j}"

    def test_optimal_threshold_in_range(self, synthetic_two_class):
        scores, labels = synthetic_two_class
        cal = ScoreCalibrator()
        cal.fit_two_class(scores, labels)
        assert 0.88 < cal._optimal_threshold < 0.98

    def test_benign_gets_low_risk(self, synthetic_two_class):
        scores, labels = synthetic_two_class
        cal = ScoreCalibrator()
        cal.fit_two_class(scores, labels)
        result = cal.calibrate(0.88)
        assert result["risk_level"] in ("NORMAL", "LOW")

    def test_attack_gets_high_risk(self, synthetic_two_class):
        scores, labels = synthetic_two_class
        cal = ScoreCalibrator()
        cal.fit_two_class(scores, labels)
        result = cal.calibrate(0.97)
        assert result["risk_level"] in ("MEDIUM", "HIGH", "CRITICAL")


class TestFallback:
    def test_benign_only_mode(self):
        cal = ScoreCalibrator()
        scores = list(np.random.RandomState(42).normal(0.90, 0.02, 100))
        labels = [0] * 100
        cal.fit_from_buffer(scores, labels)
        assert cal.fitted
        assert not cal._use_two_class

    def test_too_few_attack_falls_back(self):
        rng = np.random.RandomState(42)
        scores = list(rng.normal(0.90, 0.02, 50)) + list(rng.normal(0.95, 0.02, 3))
        labels = [0] * 50 + [1] * 3
        cal = ScoreCalibrator()
        cal.fit_from_buffer(scores, labels)
        assert cal.fitted
        assert not cal._use_two_class  # 3 attack < min_attack_samples (5)

    def test_too_few_benign_does_not_fit(self):
        scores = [0.95] * 10
        labels = [1] * 10
        cal = ScoreCalibrator()
        cal.fit_from_buffer(scores, labels)
        assert not cal.fitted

    def test_unfitted_returns_raw(self):
        cal = ScoreCalibrator()
        result = cal.calibrate(0.92)
        assert result["risk_level"] in ("NORMAL", "LOW", "MEDIUM", "HIGH", "CRITICAL")
        assert result["raw_score"] == 0.92


class TestCalibrateOutput:
    def test_returns_correct_keys(self, synthetic_two_class):
        scores, labels = synthetic_two_class
        cal = ScoreCalibrator()
        cal.fit_two_class(scores, labels)
        result = cal.calibrate(0.92)
        assert "percentile" in result
        assert "risk_level" in result
        assert "calibrated_score" in result
        assert "raw_score" in result

    def test_platt_spreads_probabilities(self, synthetic_two_class):
        scores, labels = synthetic_two_class
        cal = ScoreCalibrator()
        cal.fit_two_class(scores, labels)

        low_result = cal.calibrate(0.86)
        high_result = cal.calibrate(0.98)
        spread = high_result["calibrated_score"] - low_result["calibrated_score"]
        assert spread > 0.4, (
            f"Platt scaling should spread probabilities: "
            f"low={low_result['calibrated_score']}, high={high_result['calibrated_score']}"
        )


class TestClinicalSLA:
    def test_detection_metrics_on_synthetic(self):
        """Sanity check: two-class calibrator outperforms random on overlapping data.

        Uses heavy-overlap synthetic data (benign/attack means only 2 SD apart).
        Real-data SLA validation is done end-to-end on streaming simulation.
        """
        rng = np.random.RandomState(42)
        n_benign, n_attack = 750, 250
        benign_scores = rng.normal(0.91, 0.02, n_benign)
        attack_scores = rng.normal(0.95, 0.02, n_attack)
        scores = np.clip(np.concatenate([benign_scores, attack_scores]), 0.01, 0.99)
        labels = np.concatenate([np.zeros(n_benign), np.ones(n_attack)]).astype(int)

        cal = ScoreCalibrator()
        cal.fit_two_class(scores, labels)

        detected_levels = {"MEDIUM", "HIGH", "CRITICAL"}
        tp = fn = fp = tn = 0
        for s, l in zip(scores, labels):
            risk = cal.calibrate(float(s))["risk_level"]
            detected = risk in detected_levels
            if l == 1 and detected:
                tp += 1
            elif l == 1 and not detected:
                fn += 1
            elif l == 0 and detected:
                fp += 1
            else:
                tn += 1

        recall = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)

        assert recall > 0.70, f"Recall {recall:.1%} < 70%"
        assert fpr < 0.25, f"FPR {fpr:.1%} > 25%"
        assert f1 > 0.55, f"F1 {f1:.1%} < 55%"


class TestGetConfig:
    def test_two_class_config(self, synthetic_two_class):
        scores, labels = synthetic_two_class
        cal = ScoreCalibrator()
        cal.fit_two_class(scores, labels)
        config = cal.get_config()
        assert config["mode"] == "two_class"
        assert "optimal_threshold" in config
        assert "youden_j" in config
        assert "platt_coef" in config

    def test_benign_only_config(self):
        cal = ScoreCalibrator()
        benign = np.random.RandomState(42).normal(0.90, 0.02, 100)
        cal.fit(benign)
        config = cal.get_config()
        assert config["mode"] == "benign_only"
        assert "thresholds" in config
