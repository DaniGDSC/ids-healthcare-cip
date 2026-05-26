"""Module 6 triage helpers — filters + tier counts + feed dataframe."""
from __future__ import annotations

from module6_evaluation.triage_helpers import (
    apply_dashboard_filters,
    build_feed_dataframe,
    compute_tier_counts,
    floor_elevated,
    primary_action,
)


# ── floor_elevated ─────────────────────────────────────────────────────


def test_floor_elevated_critical_clinical():
    alert = {"risk_level": "CRITICAL",
             "risk_components": {"D_crit": 0.7, "D_clinical_tier": 0.5}}
    assert floor_elevated(alert) is True


def test_floor_elevated_critical_low_clinical():
    alert = {"risk_level": "CRITICAL",
             "risk_components": {"D_crit": 0.2, "D_clinical_tier": 0.1}}
    assert floor_elevated(alert) is False


def test_floor_elevated_high_threshold():
    alert = {"risk_level": "HIGH",
             "risk_components": {"D_crit": 0.8, "D_clinical_tier": 0.2}}
    assert floor_elevated(alert) is True


def test_floor_elevated_medium_returns_false():
    alert = {"risk_level": "MEDIUM",
             "risk_components": {"D_crit": 1.0, "D_clinical_tier": 1.0}}
    assert floor_elevated(alert) is False


def test_floor_elevated_low_returns_false():
    alert = {"risk_level": "LOW",
             "risk_components": {"D_crit": 0.0, "D_clinical_tier": 0.0}}
    assert floor_elevated(alert) is False


def test_floor_elevated_missing_components():
    alert = {"risk_level": "CRITICAL"}
    assert floor_elevated(alert) is False


# ── apply_dashboard_filters ────────────────────────────────────────────


def _records():
    return [
        {"sample_index": 0, "alert_id": "A0", "risk_level": "CRITICAL",
         "attack_category": "Spoofing"},
        {"sample_index": 1, "alert_id": "A1", "risk_level": "HIGH",
         "attack_category": "Data Alteration"},
        {"sample_index": 2, "alert_id": "A2", "risk_level": "MEDIUM",
         "attack_category": "Spoofing"},
        {"sample_index": 3, "alert_id": "A3", "risk_level": "LOW",
         "attack_category": "normal"},
    ]


def test_filter_no_filters_returns_all():
    out = apply_dashboard_filters(_records())
    assert len(out) == 4


def test_filter_severity_floor_high():
    out = apply_dashboard_filters(_records(), severity_floor="HIGH")
    assert len(out) == 2
    assert {r["risk_level"] for r in out} == {"CRITICAL", "HIGH"}


def test_filter_severity_floor_critical():
    out = apply_dashboard_filters(_records(), severity_floor="CRITICAL")
    assert len(out) == 1
    assert out[0]["risk_level"] == "CRITICAL"


def test_filter_attack_category():
    out = apply_dashboard_filters(_records(), attack_category="Spoofing")
    assert len(out) == 2
    assert all(r["attack_category"] == "Spoofing" for r in out)


def test_filter_search_text_alert_id():
    out = apply_dashboard_filters(_records(), search_text="A2")
    assert len(out) == 1
    assert out[0]["alert_id"] == "A2"


def test_filter_combined():
    out = apply_dashboard_filters(_records(),
                                   severity_floor="MEDIUM",
                                   attack_category="Spoofing")
    assert len(out) == 2
    assert {r["alert_id"] for r in out} == {"A0", "A2"}


def test_filter_empty_input():
    assert apply_dashboard_filters([]) == []


# ── compute_tier_counts ────────────────────────────────────────────────


def test_compute_tier_counts():
    counts = compute_tier_counts(tuple(_records()))
    assert counts["CRITICAL"] == 1
    assert counts["HIGH"] == 1
    assert counts["MEDIUM"] == 1
    assert counts["LOW"] == 1


def test_compute_tier_counts_empty():
    assert compute_tier_counts(()) == {}


# ── build_feed_dataframe ───────────────────────────────────────────────


def test_build_feed_dataframe_columns():
    rows = tuple({
        "sample_index": 0, "alert_id": "A0", "risk_level": "HIGH",
        "risk_score": 0.85, "attack_category": "Spoofing",
        "response": {"actions": ["log_event", "isolate_device"]},
    } for _ in range(2))
    df = build_feed_dataframe(rows)
    assert list(df.columns) == [
        "sample_index", "alert_id", "risk_level", "risk_score",
        "attack_category", "actions",
    ]
    assert len(df) == 2
    assert df.iloc[0]["actions"] == "log_event|isolate_device"


def test_build_feed_dataframe_empty():
    df = build_feed_dataframe(())
    assert len(df) == 0


# ── primary_action ─────────────────────────────────────────────────────


def test_primary_action_picks_highest_cost():
    assert primary_action(["log_event", "isolate_device"]) == "isolate_device"


def test_primary_action_empty():
    assert primary_action([]) == "log_event"


def test_primary_action_unknown_action():
    # Unknown actions get _ACTION_DISPLAY_MISS rank — should not crash.
    out = primary_action(["nonexistent_action", "log_event"])
    assert out in ("nonexistent_action", "log_event")
