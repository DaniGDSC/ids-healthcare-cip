"""ARCHITECTURE.md Step [15] — ResponseRecommendation contract tests.

Locks:

* INVARIANT 3 — ``operator_decision_required`` is **always True**;
  setting it False raises ``ValueError`` at construction.
* Schema completeness — every required field per the doc is present
  with the right type / domain.
* ``estimated_clinical_impact`` ∈ {minimal, moderate, high}.
* ``suggested_priority`` ∈ [1, 5].
* The ``primary_action_code`` of a structured recommendation is
  **machine-readable** (snake_case / known codes), not a sentence.
* No-auto-execution invariant: nothing in M5 imports subprocess /
  os.system / iptables / ssh / sudo / eval / exec (expanded grep per
  the doc).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from src.data_models import ResponseRecommendation
from module5_responses.module5_pipeline import PolicyEngine

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ── INVARIANT 3 — operator_decision_required always True ──────────────


def test_invariant_3_operator_decision_required_always_true():
    rec = ResponseRecommendation(
        primary_action="Isolate device",
        primary_action_code="isolate_device",
        rationale="HIGH KNOWN_ATTACK on ventilator",
    )
    assert rec.operator_decision_required is True


def test_invariant_3_setting_false_is_rejected():
    with pytest.raises(ValueError, match="INVARIANT 3"):
        ResponseRecommendation(
            primary_action="x",
            primary_action_code="x",
            rationale="r",
            operator_decision_required=False,
        )


# ── Domain validation ─────────────────────────────────────────────────


@pytest.mark.parametrize("impact", ["minimal", "moderate", "high"])
def test_estimated_clinical_impact_accepts_valid_values(impact: str):
    rec = ResponseRecommendation(
        primary_action="x", primary_action_code="x", rationale="r",
        estimated_clinical_impact=impact,
    )
    assert rec.estimated_clinical_impact == impact


def test_estimated_clinical_impact_rejects_unknown_values():
    with pytest.raises(ValueError, match="estimated_clinical_impact"):
        ResponseRecommendation(
            primary_action="x", primary_action_code="x", rationale="r",
            estimated_clinical_impact="catastrophic",
        )


@pytest.mark.parametrize("priority", [1, 2, 3, 4, 5])
def test_suggested_priority_accepts_valid_range(priority: int):
    rec = ResponseRecommendation(
        primary_action="x", primary_action_code="x", rationale="r",
        suggested_priority=priority,
    )
    assert rec.suggested_priority == priority


@pytest.mark.parametrize("priority", [0, 6, -1, 99])
def test_suggested_priority_rejects_out_of_range(priority: int):
    with pytest.raises(ValueError, match="suggested_priority"):
        ResponseRecommendation(
            primary_action="x", primary_action_code="x", rationale="r",
            suggested_priority=priority,
        )


# ── PolicyEngine.recommend_structured produces ResponseRecommendation ─


def test_policy_engine_emits_structured_recommendation():
    engine = PolicyEngine()
    rec = engine.recommend_structured(
        alert_tier="CRITICAL",
        device_tier="life_sustaining",
        attack_category="data_alteration",
    )
    assert isinstance(rec, ResponseRecommendation)
    assert rec.operator_decision_required is True
    # CRITICAL on life-sustaining device → priority 1
    assert rec.suggested_priority == 1
    # primary_action_code must be machine-readable (snake_case)
    assert re.fullmatch(r"[a-z0-9_]+", rec.primary_action_code), (
        f"primary_action_code {rec.primary_action_code!r} is not snake_case"
    )


def test_policy_engine_low_tier_emits_lower_priority():
    engine = PolicyEngine()
    rec = engine.recommend_structured(alert_tier="LOW", device_tier="administrative")
    assert rec.suggested_priority == 4   # LOW → priority 4 per doc mapping


def test_critical_life_sustaining_isolation_marked_high_impact():
    engine = PolicyEngine()
    rec = engine.recommend_structured(
        alert_tier="CRITICAL",
        device_tier="life_sustaining",
        attack_category="data_alteration",
    )
    if rec.primary_action_code == "isolate_device":
        assert rec.estimated_clinical_impact == "high"


# ── No auto-execution: expanded grep per doc ──────────────────────────


_FORBIDDEN_PATTERNS = [
    r"\bsubprocess\b",
    r"\bos\.system\b",
    r"\biptables\b",
    r"\bnetcat\b",
    r"\bssh\b",
    r"\bsudo\b",
    r"\beval\(",
    r"\bexec\(",
    r"\bnc\s",
    r"\bcurl\b",
    r"\bwget\b",
]
_M5_DIR = PROJECT_ROOT / "module5_responses"


def test_no_auto_execution_in_module5_source():
    """No M5 source file may import or call any of the forbidden
    process-mutating primitives. Recommendation-only contract per
    INVARIANT 3 / ARCHITECTURE.md Step [15]."""
    py_files = [
        p for p in _M5_DIR.rglob("*.py")
        if "__pycache__" not in p.parts
    ]
    assert py_files, "No M5 .py files found — fixture mis-pointed?"
    offenders: list[tuple[str, str]] = []
    for path in py_files:
        text = path.read_text(encoding="utf-8")
        for pat in _FORBIDDEN_PATTERNS:
            if re.search(pat, text):
                offenders.append((str(path.relative_to(PROJECT_ROOT)), pat))
    assert not offenders, (
        f"M5 source contains forbidden auto-execution primitives: {offenders}"
    )
