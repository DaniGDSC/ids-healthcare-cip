"""Unit tests for ThresholdFallbackManager (lock/resume)."""

from __future__ import annotations

import numpy as np

from src.phase4_risk_engine.phase4.drift_detector import ConceptDriftDetector
from src.phase4_risk_engine.phase4.fallback_manager import ThresholdFallbackManager


class TestThresholdFallbackManager:
    """Test threshold fallback locking and recovery."""

    def test_no_drift_passthrough(self) -> None:
        detector = ConceptDriftDetector(drift_threshold=0.20)
        baseline_threshold = 0.20
        manager = ThresholdFallbackManager(
            drift_detector=detector,
            baseline_threshold=baseline_threshold,
            recovery_threshold=0.10,
            recovery_windows=3,
        )
        # Thresholds close to baseline → no drift
        thresholds = np.full(500, baseline_threshold)
        adjusted, events = manager.process(thresholds, window_size=100)

        assert len(events) == 0
        assert not manager.is_locked

    def test_drift_locks_to_baseline(self) -> None:
        detector = ConceptDriftDetector(drift_threshold=0.20)
        baseline_threshold = 0.20
        manager = ThresholdFallbackManager(
            drift_detector=detector,
            baseline_threshold=baseline_threshold,
            recovery_threshold=0.10,
            recovery_windows=3,
        )
        # Large deviation at window boundary (sample 100)
        thresholds = np.full(500, baseline_threshold)
        thresholds[100] = 0.50  # 150% drift ratio
        adjusted, events = manager.process(thresholds, window_size=100)

        assert len(events) >= 1
        assert events[0]["action"] == "FALLBACK_LOCKED"
        assert np.isclose(adjusted[100], baseline_threshold)

    def test_recovery_after_n_windows(self) -> None:
        detector = ConceptDriftDetector(drift_threshold=0.20)
        baseline_threshold = 0.20
        manager = ThresholdFallbackManager(
            drift_detector=detector,
            baseline_threshold=baseline_threshold,
            recovery_threshold=0.10,
            recovery_windows=2,
        )
        # Drift at 100, then stable at 200, 300
        thresholds = np.full(500, baseline_threshold)
        thresholds[100] = 0.50  # Triggers drift
        # 200 and 300 are at baseline (ratio=0) → stable
        adjusted, events = manager.process(thresholds, window_size=100)

        # Should have FALLBACK_LOCKED and then RESUMED_DYNAMIC
        actions = [e["action"] for e in events]
        assert "FALLBACK_LOCKED" in actions
        assert "RESUMED_DYNAMIC" in actions

    def test_is_locked_property(self) -> None:
        detector = ConceptDriftDetector(drift_threshold=0.20)
        manager = ThresholdFallbackManager(
            drift_detector=detector,
            baseline_threshold=0.20,
            recovery_threshold=0.10,
            recovery_windows=100,  # Never recovers in 500 samples
        )
        thresholds = np.full(500, 0.20)
        thresholds[100] = 0.50
        manager.process(thresholds, window_size=100)

        # Still locked because recovery_windows=100 was never reached
        assert manager.is_locked
