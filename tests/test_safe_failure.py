from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.risk_scorer import DEFAULT_THRESHOLD, score_alert


def test_missing_device_context_defaults_to_low_and_preserves_score() -> None:
    result = score_alert(anomaly_score=0.7, device_context={}, event_context=None)
    assert result.should_surface is True
    assert result.adjusted_score == 0.7
    assert result.threshold == DEFAULT_THRESHOLD


def test_critical_unpatchable_device_always_surfaces_low_signal() -> None:
    context = {"criticality": "CRITICAL", "patchable": False}
    result = score_alert(anomaly_score=0.2, device_context=context, event_context=None)
    assert result.should_surface is True
    assert result.risk_multiplier >= 1.5
    assert result.threshold <= 0.35


def test_unknown_criticality_falls_back_without_suppressing_strong_signal() -> None:
    context = {"criticality": "UNKNOWN", "patchable": True}
    result = score_alert(anomaly_score=0.85, device_context=context, event_context=None)
    assert result.should_surface is True
    assert result.threshold == DEFAULT_THRESHOLD
    assert result.adjusted_score == 0.85


def test_new_device_baseline_distortion_lowers_threshold() -> None:
    context = {"criticality": "MEDIUM", "patchable": True}
    event_context = {"baseline_days": 7}
    result = score_alert(anomaly_score=0.4, device_context=context, event_context=event_context)
    assert result.should_surface is True
    assert result.threshold < DEFAULT_THRESHOLD
    assert result.adjusted_score == 0.4


def test_maintenance_vendor_event_reduces_confidence_but_can_still_surface() -> None:
    context = {"criticality": "CRITICAL", "patchable": False}
    event_context = {"is_maintenance_window": True, "is_known_vendor_ip": True}
    result = score_alert(anomaly_score=1.1, device_context=context, event_context=event_context)
    assert result.should_surface is True
    assert result.risk_multiplier == 0.5
    assert result.suppression_reason is not None
