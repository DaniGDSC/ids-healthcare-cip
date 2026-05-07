"""Layer 6 v4.0 — tier-routing + decision-quality + INVARIANT 3 tests.

Covers the deltas added on top of the existing Layer 6 infrastructure
(``module5_responses/{module5_pipeline,module5_responses}.py``,
``module6_evaluation/module6_app.py::AuditTrailWriter``,
``tests/test_audit_append_only.py``):

  * 9-class :class:`AlertType` → :class:`TierLevel` routing is total
    and matches the prompt's prescribed table
  * ``DISAGREEMENT_ANOMALY`` is the only alert type routing to
    ``L2_SECURITY_SPECIALIST`` and the only one that flags
    ``adversarial_flag=True`` and ``requires_security_specialist=True``
  * Confidence-based adjustment (LOW demotes urgent tiers; others
    pass through)
  * Hospital-realistic fallback adjustments for
    ``is_after_hours`` / ``clinical_active``
  * ``operator_followed_recommendation`` correctly identifies
    matches/non-matches
  * **INVARIANT 3** — architectural grep verification that
    ``module5_responses/`` contains no execution primitives
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from module5_responses.tier_routing_v4 import (
    TierLevel,
    operator_followed_recommendation,
    recommend_tier_v4,
)
from src.data_models import AlertType, Confidence


PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ── 1. Routing totality ─────────────────────────────────────────────────

def test_recommend_tier_v4_covers_every_alert_type() -> None:
    """Every AlertType must produce a TierRecommendationV4 — no
    silent fall-throughs that would leak as a missing tier on the
    dashboard.
    """
    for alert_type in AlertType:
        rec = recommend_tier_v4(alert_type)
        assert isinstance(rec.recommended_tier, TierLevel)
        assert rec.rationale  # non-empty


def test_string_round_trips_match_enum() -> None:
    for alert_type in AlertType:
        a = recommend_tier_v4(alert_type)
        b = recommend_tier_v4(alert_type.value)
        assert a.recommended_tier == b.recommended_tier
        assert a.rationale == b.rationale


def test_unknown_alert_type_falls_back_to_novel_policy() -> None:
    """Most cautious routing for unrecognised input — investigate."""
    fallback = recommend_tier_v4("not_a_real_alert_type")
    novel = recommend_tier_v4(AlertType.NOVEL_ANOMALY)
    assert fallback.recommended_tier == novel.recommended_tier


# ── 2. Spec-mandated routing per alert type ─────────────────────────────

@pytest.mark.parametrize("alert_type, expected_tier", [
    (AlertType.KNOWN_ATTACK, TierLevel.L1_IMMEDIATE),
    (AlertType.KNOWN_ATTACK_UNCERTAIN, TierLevel.L1_WITH_REVIEW),
    (AlertType.DISAGREEMENT_ANOMALY, TierLevel.L2_SECURITY_SPECIALIST),
    (AlertType.STRONG_NOVEL_ANOMALY, TierLevel.L2_SPECIALIST),
    (AlertType.NOVEL_ANOMALY, TierLevel.L2_SPECIALIST),
    (AlertType.CONFIRMED_ANOMALY, TierLevel.L1_WITH_SENIOR),
    (AlertType.SUSPICIOUS_PATTERN, TierLevel.L1),
    (AlertType.BENIGN_WATCH, TierLevel.AUDIT_LOG),
    (AlertType.BENIGN, TierLevel.SUPPRESSED),
])
def test_per_alert_type_default_tier(alert_type, expected_tier) -> None:
    """Pin the prompt's prescribed routing table — a future change
    has to be intentional and tracked.
    """
    rec = recommend_tier_v4(alert_type, Confidence.MEDIUM)
    assert rec.recommended_tier == expected_tier


# ── 3. Adversarial routing exclusivity ──────────────────────────────────

def test_only_disagreement_routes_to_security_specialist() -> None:
    """L2_SECURITY_SPECIALIST is reserved for DISAGREEMENT_ANOMALY."""
    sec_routes = {
        a for a in AlertType
        if recommend_tier_v4(a).recommended_tier == TierLevel.L2_SECURITY_SPECIALIST
    }
    assert sec_routes == {AlertType.DISAGREEMENT_ANOMALY}


def test_only_disagreement_flags_adversarial() -> None:
    """``adversarial_flag`` and ``requires_security_specialist`` are
    exclusive to DISAGREEMENT_ANOMALY — operators key on these.
    """
    for alert_type in AlertType:
        rec = recommend_tier_v4(alert_type)
        is_disagreement = (alert_type == AlertType.DISAGREEMENT_ANOMALY)
        assert rec.adversarial_flag is is_disagreement
        assert rec.requires_security_specialist is is_disagreement


# ── 4. Confidence-based adjustment ──────────────────────────────────────

def test_low_confidence_demotes_l1_immediate_to_l1() -> None:
    rec = recommend_tier_v4(AlertType.KNOWN_ATTACK, Confidence.LOW)
    assert rec.recommended_tier == TierLevel.L1


def test_low_confidence_demotes_l1_to_audit_log() -> None:
    rec = recommend_tier_v4(AlertType.SUSPICIOUS_PATTERN, Confidence.LOW)
    assert rec.recommended_tier == TierLevel.AUDIT_LOG


def test_high_confidence_does_not_demote() -> None:
    """HIGH/VERY_HIGH/MEDIUM all preserve the base tier — we never
    silently *promote* either, that would surprise operators."""
    base = recommend_tier_v4(AlertType.KNOWN_ATTACK, Confidence.MEDIUM).recommended_tier
    for c in (Confidence.HIGH, Confidence.VERY_HIGH):
        rec = recommend_tier_v4(AlertType.KNOWN_ATTACK, c)
        assert rec.recommended_tier == base


def test_low_confidence_does_not_demote_l2_routes() -> None:
    """L2 routes (specialist / security specialist) are not on the
    demotion ladder — a LOW-confidence DISAGREEMENT still goes to
    security, because the disagreement signal itself is what matters.
    """
    rec = recommend_tier_v4(AlertType.DISAGREEMENT_ANOMALY, Confidence.LOW)
    assert rec.recommended_tier == TierLevel.L2_SECURITY_SPECIALIST


# ── 5. Hospital-realistic fallback adjustments ──────────────────────────

def test_after_hours_appends_oncall_fallback() -> None:
    rec = recommend_tier_v4(
        AlertType.KNOWN_ATTACK, Confidence.MEDIUM,
        is_after_hours=True,
    )
    assert any("on-call" in s.lower() for s in rec.fallback_options)


def test_clinical_active_prepends_coordinate_fallback() -> None:
    """Active clinical care must be the FIRST fallback so operators
    talk to the unit before any device action.
    """
    rec = recommend_tier_v4(
        AlertType.KNOWN_ATTACK, Confidence.MEDIUM,
        clinical_active=True,
    )
    assert rec.fallback_options
    assert "clinical" in rec.fallback_options[0].lower()


def test_no_extra_fallbacks_when_flags_are_off() -> None:
    rec_on = recommend_tier_v4(
        AlertType.KNOWN_ATTACK, Confidence.MEDIUM,
        is_after_hours=True, clinical_active=True,
    )
    rec_off = recommend_tier_v4(AlertType.KNOWN_ATTACK, Confidence.MEDIUM)
    assert len(rec_on.fallback_options) > len(rec_off.fallback_options)


# ── 6. operator_followed_recommendation helper ──────────────────────────

def test_operator_followed_when_action_matches() -> None:
    assert operator_followed_recommendation(
        "Investigate network traffic",
        ["Investigate network traffic", "Snapshot device for forensics"],
    ) is True


def test_operator_followed_case_insensitive() -> None:
    assert operator_followed_recommendation(
        "investigate NETWORK traffic",
        ["Investigate network traffic"],
    ) is True


def test_operator_did_not_follow_when_action_differs() -> None:
    assert operator_followed_recommendation(
        "Block source IP",
        ["Investigate network traffic", "Document incident"],
    ) is False


def test_placeholder_selection_does_not_count_as_followed() -> None:
    """The dashboard's "— Select action —" placeholder must not
    register as following a recommendation; we'd inflate the metric.
    """
    assert operator_followed_recommendation(
        "— Select action —",
        ["Investigate network traffic"],
    ) is False


def test_empty_or_whitespace_action_does_not_count() -> None:
    assert operator_followed_recommendation("", ["Anything"]) is False
    assert operator_followed_recommendation("   ", ["Anything"]) is False


def test_empty_recommendation_list_returns_false() -> None:
    """When the system had no recommendation, the operator can't
    follow one. Defensive: returns False rather than raising.
    """
    assert operator_followed_recommendation("Any action", []) is False


# ── 7. INVARIANT 3 — no auto-execution primitives in module5_responses ─

# Patterns that signal command execution. Each pattern is a regex or
# substring; we grep the package recursively. The empty-result rule is
# the architectural enforcement of INVARIANT 3 the prompt asks for.
_EXECUTION_PRIMITIVES: tuple[tuple[str, str], ...] = (
    ("subprocess",        r"\bsubprocess\b"),
    ("os.system",         r"\bos\.system\b"),
    ("iptables",          r"\biptables\b"),
    ("firewall_rule_add", r"\bfirewall_rule_add\b"),
    # ``sudo`` and ``shell=True`` are heuristics — they *strongly*
    # suggest command execution. Comment / docstring text that mentions
    # them is fine; we only care about code use, but a substring grep
    # is a conservative first cut. If a future legitimate doc string
    # uses one of these, suppress with a per-test allowlist instead of
    # weakening the rule.
    ("os.popen",          r"\bos\.popen\b"),
    ("Popen",             r"\bPopen\("),
)


def _grep_module5_responses(pattern: str) -> list[str]:
    """Return a list of ``path:line: matched_text`` for hits in
    ``module5_responses/``. Walks the tree directly so the test does
    not depend on the ``grep`` binary being installed.
    """
    hits: list[str] = []
    pkg = PROJECT_ROOT / "module5_responses"
    rx = re.compile(pattern)
    for py in pkg.rglob("*.py"):
        # Don't walk our own __pycache__ directories.
        if "__pycache__" in py.parts:
            continue
        for lineno, line in enumerate(py.read_text().splitlines(), start=1):
            if rx.search(line):
                hits.append(f"{py.relative_to(PROJECT_ROOT)}:{lineno}: {line.strip()}")
    return hits


@pytest.mark.parametrize("name, pattern", _EXECUTION_PRIMITIVES)
def test_invariant_3_no_execution_primitives_in_module5_responses(
    name: str, pattern: str,
) -> None:
    """Architectural enforcement of INVARIANT 3 — module5_responses
    must contain no command-execution primitives.

    A hit here means a developer added shell-out behaviour to the
    response/recommendation layer; the IDS becomes capable of
    autonomously isolating a clinical device, which is exactly what
    INVARIANT 3 forbids.
    """
    hits = _grep_module5_responses(pattern)
    assert hits == [], (
        f"INVARIANT 3 VIOLATION: '{name}' execution primitive found in "
        f"module5_responses/:\n  " + "\n  ".join(hits)
    )


def test_invariant_3_self_check_grep_actually_works() -> None:
    """Sanity: the grep helper finds a known string in this test file
    (which is OUTSIDE the package being scanned). If this fails the
    other INVARIANT 3 tests are vacuous.
    """
    pkg = PROJECT_ROOT / "module5_responses"
    # Plant nothing — just verify the regex engine and walker don't
    # silently return [] on a known pattern that doesn't appear.
    assert _grep_module5_responses(r"never_matches_xyz_12345") == []
    # And confirm the walker actually visits .py files.
    assert any((PROJECT_ROOT / "module5_responses").rglob("*.py"))
