from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.risk_scorer import score_alert


def test_missing_device_context_defaults_safe() -> None:
    """Failure mode 1: no criticality/patchable metadata should still surface a strong alert."""
    result = score_alert(anomaly_score=0.7, device_context={}, event_context=None)
    assert result.should_surface is True
    assert result.adjusted_score >= 0.7


def test_mve_timeout_does_not_suppress() -> None:
    """Failure mode 2: risk scoring is independent of explanation generation."""
    context = {"criticality": "CRITICAL", "patchable": False}
    result = score_alert(anomaly_score=0.6, device_context=context, event_context=None)
    assert result.should_surface is True
    assert result.risk_multiplier >= 1.5


def test_unknown_attack_type_high_score() -> None:
    """Failure mode 3: a strong signal on a HIGH-criticality device should surface."""
    context = {"criticality": "HIGH", "patchable": True}
    result = score_alert(anomaly_score=0.85, device_context=context, event_context=None)
    assert result.should_surface is True
    assert result.threshold < 0.7


def test_feature_distortion_robustness() -> None:
    """Failure mode 4: extreme anomaly scores should still surface under normal context."""
    context = {"criticality": "MEDIUM", "patchable": True}
    result = score_alert(anomaly_score=0.95, device_context=context, event_context=None)
    assert result.should_surface is True
    assert result.adjusted_score >= 0.8


def test_non_patchable_priority() -> None:
    """Failure mode 5: unpatchable devices should receive elevated priority."""
    context = {"criticality": "HIGH", "patchable": False}
    result = score_alert(anomaly_score=0.55, device_context=context, event_context=None)
    assert result.risk_multiplier > 1.0
    assert result.should_surface is True
