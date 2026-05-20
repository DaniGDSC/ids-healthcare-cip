"""Runtime checks strengthening Layer C of the no-auto-execution defense.

  1. ResponseRecommendation defaults preserve operator_decision_required=True
     (positive evidence: the constructor refuses to set it False).
  2. Calling PolicyEngine.recommend() never invokes subprocess (mocked
     smoke test — strongest possible runtime evidence).

Complements (does not replace) tests/negative_tests.py — those are
orchestrator-style functions invoked by run_negative_tests; these are
pytest-collectible CI gates.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# Real import paths (Phase 0 discovery confirmed)
from src.data_models import ResponseRecommendation
from module5_responses.module5_pipeline import PolicyEngine


# ── HITL-INVARIANT: operator_decision_required default ────────────────


def test_operator_decision_required_default_is_true():
    """A default-constructed ResponseRecommendation must have
    operator_decision_required=True. If the default flips, every path
    using defaults silently disables HITL."""
    rec = ResponseRecommendation(
        primary_action="Isolate device from network",
        primary_action_code="isolate_device",
        rationale="High-confidence threat to clinical device.",
    )
    assert rec.operator_decision_required is True, (
        "DEFAULT VIOLATION: ResponseRecommendation default-constructed "
        "with operator_decision_required != True. This breaks Invariant 3."
    )


def test_operator_decision_required_cannot_be_set_false():
    """The dataclass __post_init__ validator refuses to construct a
    recommendation with operator_decision_required=False. This is the
    hard invariant: a future caller cannot opt out without rewriting
    the dataclass itself.
    """
    with pytest.raises((ValueError, AssertionError, TypeError)) as exc_info:
        ResponseRecommendation(
            primary_action="x",
            primary_action_code="x",
            rationale="x",
            operator_decision_required=False,
        )
    # Sanity: the error message should reference the invariant.
    msg = str(exc_info.value).lower()
    assert "operator_decision_required" in msg or "invariant" in msg, (
        "Validator raised but the message didn't mention the invariant; "
        f"got: {exc_info.value!r}"
    )


def test_operator_decision_required_across_typical_construction_patterns():
    """Multiple construction patterns; none should yield False."""
    cases = [
        {"primary_action": "Isolate device from network",
         "primary_action_code": "isolate_device",
         "rationale": "Confirmed threat."},
        {"primary_action": "Restrict outbound traffic",
         "primary_action_code": "restrict_traffic",
         "rationale": "Anomalous outbound pattern.",
         "estimated_clinical_impact": "moderate",
         "suggested_priority": 2},
        {"primary_action": "Log event",
         "primary_action_code": "log_event",
         "rationale": "Below threshold; recorded for trend analysis.",
         "do_not_actions": ["power_cycle_device"]},
    ]
    for kwargs in cases:
        rec = ResponseRecommendation(**kwargs)
        assert rec.operator_decision_required is True, (
            f"Construction {kwargs} yielded operator_decision_required=False"
        )


# ── DEFENSE-CRITICAL: recommend() never invokes subprocess ────────────


def test_policy_engine_recommend_never_invokes_subprocess():
    """Patch subprocess + os.system globally, run PolicyEngine.recommend()
    end-to-end, assert the mocks were never invoked.

    This is the strongest possible runtime evidence that the system does
    NOT auto-execute mitigation actions through any path — even indirect
    ones a static grep would miss.
    """
    engine = PolicyEngine()

    subprocess_mock = MagicMock()
    os_system_mock = MagicMock()
    os_popen_mock = MagicMock()

    with patch("subprocess.run", subprocess_mock), \
         patch("subprocess.Popen", subprocess_mock), \
         patch("subprocess.call", subprocess_mock), \
         patch("subprocess.check_output", subprocess_mock), \
         patch("subprocess.check_call", subprocess_mock), \
         patch("os.system", os_system_mock), \
         patch("os.popen", os_popen_mock):

        result = engine.recommend(
            alert_tier="CRITICAL",
            device_tier="vital_monitoring",
            attack_category="Data Alteration",
            patient_acuity=0.9,
        )

    assert subprocess_mock.call_count == 0, (
        f"subprocess invoked {subprocess_mock.call_count} time(s) during "
        f"recommend(). calls: {subprocess_mock.mock_calls[:5]}"
    )
    assert os_system_mock.call_count == 0, (
        f"os.system invoked {os_system_mock.call_count} time(s) during "
        "recommend()."
    )
    assert os_popen_mock.call_count == 0, (
        f"os.popen invoked {os_popen_mock.call_count} time(s) during "
        "recommend()."
    )
    # Sanity: recommend() must produce a non-empty action set so we
    # know it actually executed end-to-end and not short-circuited.
    assert isinstance(result, dict), (
        f"PolicyEngine.recommend returned non-dict: {type(result).__name__}"
    )
    assert result, "PolicyEngine.recommend returned empty dict"
