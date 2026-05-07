"""Layer 4 v4.0 — adapter tests.

Covers the thin shim that lets the v4 9-class :class:`AlertType` and
4-level :class:`Confidence` enums flow through the existing
``src.mve_generator``:

  * legacy template routing (``alert_type_v4_to_legacy``)
  * adversarial flag (``is_adversarial`` /
    ``adversarial_clause``)
  * confidence clause rendering (``confidence_clause``)
  * per-role MITRE format (IT: full id, biomed: threat type, nurse:
    plain language)
  * DAE per-dim Layer 1 enrichment (``anomalous_dims_clause``)
"""
from __future__ import annotations

import pytest

from module4_explanations.triage_v4_adapter import (
    ADVERSARIAL_LAYER_1_HINT,
    adversarial_clause,
    alert_type_v4_to_legacy,
    anomalous_dims_clause,
    confidence_clause,
    format_mitre_for_alert_type,
    format_mitre_for_role,
    is_adversarial,
)
from src.data_models import AlertType, Confidence, OperatorRole


# ── Template routing ─────────────────────────────────────────────────────

def test_every_v4_alert_type_routes_to_a_legacy_template() -> None:
    """The adapter must be total over the v4 typology — no silent
    fall-through to a default that would leak the raw v4 string into
    the legacy generator's switch statement.
    """
    for alert_type in AlertType:
        legacy = alert_type_v4_to_legacy(alert_type)
        assert legacy in {"T1", "T2", "T3", "T4", "T5"}


def test_string_round_trips_match_enum() -> None:
    for alert_type in AlertType:
        assert alert_type_v4_to_legacy(alert_type) == alert_type_v4_to_legacy(alert_type.value)


def test_unknown_string_falls_back_to_T1() -> None:
    assert alert_type_v4_to_legacy("not a real alert type") == "T1"


# ── Adversarial flag ─────────────────────────────────────────────────────

def test_only_disagreement_anomaly_is_adversarial() -> None:
    for alert_type in AlertType:
        expected = (alert_type == AlertType.DISAGREEMENT_ANOMALY)
        assert is_adversarial(alert_type) is expected, (
            f"{alert_type.value}: is_adversarial={is_adversarial(alert_type)}"
        )


def test_adversarial_clause_only_for_disagreement() -> None:
    assert adversarial_clause(AlertType.DISAGREEMENT_ANOMALY) == ADVERSARIAL_LAYER_1_HINT
    assert "adversarial" in ADVERSARIAL_LAYER_1_HINT.lower()
    for alert_type in AlertType:
        if alert_type == AlertType.DISAGREEMENT_ANOMALY:
            continue
        assert adversarial_clause(alert_type) == ""


def test_is_adversarial_accepts_strings_and_unknowns() -> None:
    assert is_adversarial("DISAGREEMENT_ANOMALY") is True
    assert is_adversarial("KNOWN_ATTACK") is False
    assert is_adversarial("not a real type") is False


# ── Confidence clause ────────────────────────────────────────────────────

def test_confidence_clause_for_each_level() -> None:
    for conf in Confidence:
        text = confidence_clause(conf)
        assert text.startswith("Confidence:")
        assert conf.value.replace("_", " ") in text


def test_confidence_clause_unknown_string_does_not_raise() -> None:
    text = confidence_clause("not_a_real_confidence")
    assert "UNKNOWN" in text


# ── Per-role MITRE format ────────────────────────────────────────────────

def test_it_generalist_sees_full_id_with_name() -> None:
    out = format_mitre_for_role(
        "T1071", "Application Layer Protocol",
        OperatorRole.IT_GENERALIST,
    )
    assert "T1071" in out
    assert "Application Layer Protocol" in out


def test_biomed_engineer_sees_threat_type_prose() -> None:
    out = format_mitre_for_role(
        "T1071", "Application Layer Protocol",
        OperatorRole.BIOMED_ENGINEER,
    )
    # Biomed-facing prose should describe the threat without the raw ID.
    assert "T1071" not in out
    assert out  # non-empty


def test_nurse_manager_sees_plain_language() -> None:
    out = format_mitre_for_role(
        "T1071", "Application Layer Protocol",
        OperatorRole.NURSE_MANAGER,
    )
    # Nurse-facing prose must not contain the ID and must read as a
    # sentence (no jargon code).
    assert "T1071" not in out
    assert "Application Layer Protocol" not in out
    assert any(out.lower().startswith(prefix) for prefix in (
        "equipment", "patient", "an authorized", "an authorised",
    ))


def test_per_role_outputs_are_distinct() -> None:
    """For a known technique, each role's rendering should differ —
    that's the whole point of per-role MITRE visibility.
    """
    technique = ("T1565", "Data Manipulation")
    it = format_mitre_for_role(*technique, OperatorRole.IT_GENERALIST)
    biomed = format_mitre_for_role(*technique, OperatorRole.BIOMED_ENGINEER)
    nurse = format_mitre_for_role(*technique, OperatorRole.NURSE_MANAGER)
    assert len({it, biomed, nurse}) == 3, (
        f"role outputs collided: it={it!r}, biomed={biomed!r}, nurse={nurse!r}"
    )


def test_unknown_technique_does_not_raise() -> None:
    """Format helper is total — a never-mapped technique id falls back
    to the technique name (or the id itself) without crashing.
    """
    out = format_mitre_for_role(
        "T9999", "Future Technique", OperatorRole.NURSE_MANAGER,
    )
    assert out  # non-empty
    assert "T9999" not in out  # nurse view never carries the ID


def test_format_mitre_for_alert_type_picks_right_technique() -> None:
    """End-to-end: a v4 AlertType resolves to a legacy template, which
    resolves to a technique id, which renders correctly per role."""
    out = format_mitre_for_alert_type(
        AlertType.KNOWN_ATTACK, OperatorRole.IT_GENERALIST,
    )
    # KNOWN_ATTACK → T1 → T1071 in the legacy mapping.
    assert "T1071" in out


# ── Anomalous dims clause ───────────────────────────────────────────────

FEATURE_NAMES = [f"feat_{i:02d}" for i in range(28)]


def test_empty_anomalous_dims_returns_empty_string() -> None:
    assert anomalous_dims_clause([], FEATURE_NAMES) == ""


def test_single_dim_renders_named() -> None:
    out = anomalous_dims_clause([3], FEATURE_NAMES)
    assert "feat_03" in out
    assert out.startswith("DAE flagged feature")


def test_multiple_dims_rendered_with_oxford_join() -> None:
    out = anomalous_dims_clause([1, 5, 12], FEATURE_NAMES)
    assert "feat_01" in out
    assert "feat_05" in out
    assert "feat_12" in out


def test_more_than_max_features_overflow_summary() -> None:
    out = anomalous_dims_clause([0, 1, 2, 3, 4], FEATURE_NAMES, max_features=3)
    assert "feat_00" in out and "feat_01" in out and "feat_02" in out
    # Overflow ones must NOT appear by name.
    assert "feat_03" not in out
    assert "feat_04" not in out
    # But the count must be advertised so operators know more exist.
    assert "2 more" in out


def test_out_of_range_indices_are_dropped() -> None:
    """Defensive: a stale dim index from a model trained on a different
    feature schema must not crash the renderer.
    """
    out = anomalous_dims_clause([0, 99, -1], FEATURE_NAMES)
    assert "feat_00" in out
    assert "99" not in out


def test_anomalous_dims_clause_word_count_under_layer1_budget() -> None:
    """Layer 1's word budget is 60. With max_features=3 the clause
    should never push us anywhere near the limit even on overflow.
    """
    out = anomalous_dims_clause(list(range(28)), FEATURE_NAMES, max_features=3)
    assert len(out.split()) < 25
