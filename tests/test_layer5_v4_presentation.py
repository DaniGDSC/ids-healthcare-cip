"""Layer 5 v4.0 — presentation metadata tests.

Pins the visual contract the dashboard depends on:

  * Every v4 :class:`AlertType` has a badge entry — totality.
  * The colours match the prompt's prescribed palette.
  * ``DISAGREEMENT_ANOMALY`` is the only purple badge and the only
    "ADVERSARIAL" label — the operator must not confuse it with any
    other alert class.
  * Confidence levels and Mode A/B indicators map deterministically.
  * ``anomalous_dims_markdown`` handles empty / overflow / out-of-range
    inputs without crashing the renderer.
"""
from __future__ import annotations

import pytest

from module6_evaluation.presentation_v4 import (
    BADGE_FOR_ALERT_TYPE,
    CONFIDENCE_INDICATOR,
    MODE_A_LLM,
    MODE_B_RULE_BASED,
    MODE_INDICATOR,
    anomalous_dims_markdown,
    badge_for_alert_type,
    confidence_display,
    mode_display,
)
from src.data_models import AlertType, Confidence


# ── Badge totality + colour spec ───────────────────────────────────────

def test_badge_table_covers_every_alert_type() -> None:
    """No AlertType may be missing a badge — the dashboard would crash."""
    assert set(BADGE_FOR_ALERT_TYPE) == set(AlertType), (
        f"missing badges for: "
        f"{sorted(t.value for t in set(AlertType) - set(BADGE_FOR_ALERT_TYPE))}"
    )


def test_badge_palette_matches_prompt_specification() -> None:
    """Pin the prompt's prescribed hex codes so a future palette
    change has to be intentional and tracked.
    """
    expected = {
        AlertType.KNOWN_ATTACK: "#DC2626",
        AlertType.KNOWN_ATTACK_UNCERTAIN: "#DC2626",
        AlertType.DISAGREEMENT_ANOMALY: "#9333EA",
        AlertType.STRONG_NOVEL_ANOMALY: "#EA580C",
        AlertType.NOVEL_ANOMALY: "#F97316",
        AlertType.CONFIRMED_ANOMALY: "#EAB308",
        AlertType.SUSPICIOUS_PATTERN: "#FACC15",
        AlertType.BENIGN_WATCH: "#94A3B8",
        AlertType.BENIGN: "#94A3B8",
    }
    for alert_type, expected_color in expected.items():
        assert BADGE_FOR_ALERT_TYPE[alert_type]["color"] == expected_color, (
            f"{alert_type.value}: expected colour {expected_color}, got "
            f"{BADGE_FOR_ALERT_TYPE[alert_type]['color']}"
        )


def test_only_disagreement_anomaly_is_purple() -> None:
    """Purple is the adversarial signal — must not collide with any
    other alert class colour.
    """
    purple_badges = {
        alert_type for alert_type, style in BADGE_FOR_ALERT_TYPE.items()
        if style["color"] == "#9333EA"
    }
    assert purple_badges == {AlertType.DISAGREEMENT_ANOMALY}


def test_only_disagreement_anomaly_uses_adversarial_label() -> None:
    """Only the adversarial-detection alert type may carry the
    ADVERSARIAL label — operators key on this string.
    """
    adversarial_badges = {
        alert_type for alert_type, style in BADGE_FOR_ALERT_TYPE.items()
        if "ADVERSARIAL" in style["label"].upper()
    }
    assert adversarial_badges == {AlertType.DISAGREEMENT_ANOMALY}


def test_urgency_levels_consistent_with_alert_severity() -> None:
    """KNOWN_ATTACK family + DISAGREEMENT must be HIGH urgency, NOVEL
    family must be MEDIUM, BENIGN family must be INFO. Anything else
    lives in LOW.
    """
    high = {style["urgency"] for alert_type, style in BADGE_FOR_ALERT_TYPE.items()
            if alert_type in (AlertType.KNOWN_ATTACK,
                              AlertType.KNOWN_ATTACK_UNCERTAIN,
                              AlertType.DISAGREEMENT_ANOMALY)}
    info = {style["urgency"] for alert_type, style in BADGE_FOR_ALERT_TYPE.items()
            if alert_type in (AlertType.BENIGN, AlertType.BENIGN_WATCH)}
    assert high == {"HIGH"}
    assert info == {"INFO"}


def test_badge_lookup_is_total_for_strings() -> None:
    """The dashboard receives string AlertType values from JSON; the
    helper must accept them transparently.
    """
    for alert_type in AlertType:
        assert badge_for_alert_type(alert_type.value) == BADGE_FOR_ALERT_TYPE[alert_type]


def test_badge_lookup_falls_back_to_benign_for_unknown_string() -> None:
    """Stale data must not crash the renderer — fall back to BENIGN
    so the operator still sees a recognisable cue."""
    assert badge_for_alert_type("not_a_real_type") == BADGE_FOR_ALERT_TYPE[AlertType.BENIGN]


# ── Confidence indicator ───────────────────────────────────────────────

def test_confidence_indicator_covers_every_level() -> None:
    assert set(CONFIDENCE_INDICATOR) == set(Confidence)


def test_confidence_symbol_count_matches_level() -> None:
    """The number of dots in the symbol must increase with confidence —
    the indicator is supposed to be readable at a glance."""
    assert len(CONFIDENCE_INDICATOR[Confidence.LOW]["symbol"]) == 1
    assert len(CONFIDENCE_INDICATOR[Confidence.MEDIUM]["symbol"]) == 2
    assert len(CONFIDENCE_INDICATOR[Confidence.HIGH]["symbol"]) == 3
    assert len(CONFIDENCE_INDICATOR[Confidence.VERY_HIGH]["symbol"]) == 4


def test_confidence_display_string_round_trips() -> None:
    for level in Confidence:
        assert confidence_display(level.value) == CONFIDENCE_INDICATOR[level]


def test_confidence_display_unknown_falls_back_to_low() -> None:
    """Conservative default — operators see a low-confidence cue
    rather than a blank cell when the field is corrupt."""
    assert confidence_display("not_a_real_level") == CONFIDENCE_INDICATOR[Confidence.LOW]


# ── Mode A/B indicator ─────────────────────────────────────────────────

def test_mode_indicator_canonical_modes() -> None:
    assert MODE_INDICATOR[MODE_A_LLM]["color"] == "green"
    assert MODE_INDICATOR[MODE_B_RULE_BASED]["color"] == "orange"
    assert "AI" in MODE_INDICATOR[MODE_A_LLM]["badge"].upper()
    assert "RULE" in MODE_INDICATOR[MODE_B_RULE_BASED]["badge"].upper()


def test_mode_display_unknown_string_falls_back_to_mode_b() -> None:
    """Conservative default — if we can't tell which mode produced
    the MVE, show the rule-based fallback indicator so the operator
    treats the content with appropriate scepticism."""
    assert mode_display("unknown_mode") == MODE_INDICATOR[MODE_B_RULE_BASED]


# ── Anomalous-dims markdown ────────────────────────────────────────────

FEATURES = [f"feat_{i:02d}" for i in range(28)]


def test_empty_dims_returns_empty_string() -> None:
    """No dims → empty string, so the caller can omit the expander."""
    assert anomalous_dims_markdown([], FEATURES) == ""


def test_single_dim_renders_with_count_header() -> None:
    out = anomalous_dims_markdown([3], FEATURES)
    # Bold count ``**1**`` + singular noun.
    assert "**1** anomalous dimension" in out
    assert "anomalous dimensions" not in out  # not pluralised
    assert "feat_03" in out
    assert "(dim 3)" in out


def test_multiple_dims_render_pluralised_and_listed() -> None:
    out = anomalous_dims_markdown([0, 5, 12], FEATURES)
    assert "**3** anomalous dimensions" in out  # plural
    assert "feat_00" in out
    assert "feat_05" in out
    assert "feat_12" in out


def test_overflow_collapses_to_count_summary() -> None:
    out = anomalous_dims_markdown(list(range(10)), FEATURES, max_features=5)
    assert "**10** anomalous dimensions" in out
    for i in range(5):
        assert f"feat_{i:02d}" in out
    # Items past max_features must NOT be named individually.
    assert "feat_05" not in out
    assert "feat_09" not in out
    assert "**5** more" in out


def test_out_of_range_indices_are_dropped_silently() -> None:
    """Stale indices from a model trained on a different feature
    schema must not crash the renderer."""
    out = anomalous_dims_markdown([0, 99, -1, 3], FEATURES)
    # Two valid dims (0, 3) — the count line must reflect that.
    assert "**2** anomalous dimensions" in out
    assert "feat_00" in out
    assert "feat_03" in out
    # The bogus indices must not appear anywhere.
    assert "99" not in out


def test_all_invalid_returns_empty_string() -> None:
    out = anomalous_dims_markdown([99, 100, -1], FEATURES)
    assert out == ""
