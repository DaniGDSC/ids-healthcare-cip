"""Tests for ``tools.coverage_audit`` (Sprint 2.2).

Covers the cross-surface scanner's regex correctness, path walker
edge cases, and the per-split audit invariants.
"""
from __future__ import annotations

import pytest


# ── _read_path walker ─────────────────────────────────────────────


def test_read_path_walks_simple_dict():
    from tools.coverage_audit import _read_path
    records = [
        {"a": {"b": {"c": "hello"}}},
        {"a": {"b": {"c": "world"}}},
    ]
    out = _read_path(records, "a.b.c")
    assert out == ["hello", "world"]


def test_read_path_walks_list_branch():
    from tools.coverage_audit import _read_path
    records = [
        {"items": [{"text": "one"}, {"text": "two"}]},
        {"items": [{"text": "three"}]},
    ]
    out = _read_path(records, "items[].text")
    assert out == ["one", "two", "three"]


def test_read_path_handles_missing_keys():
    from tools.coverage_audit import _read_path
    records = [{"present": "ok"}, {"other": "field"}]
    out = _read_path(records, "present")
    assert out == ["ok"]


def test_read_path_ignores_non_string_leaves():
    from tools.coverage_audit import _read_path
    records = [{"a": 42}, {"a": None}, {"a": "text"}]
    out = _read_path(records, "a")
    assert out == ["text"]


# ── SIGNALS regex correctness ────────────────────────────────────


def test_alert_id_pattern_matches_realistic_format():
    from tools.coverage_audit import SIGNALS
    pat = SIGNALS["alert_id"]["pattern"]
    assert pat.search("[ALERT-0042 · patient_monitor] Isolate device")
    assert pat.search("Alert ID ALERT-9999")
    # Doesn't false-positive on bare numbers
    assert not pat.search("Bed 42 isolated")


def test_extension_sla_pattern_matches_both_formats():
    from tools.coverage_audit import SIGNALS
    pat = SIGNALS["extension_sla"]["pattern"]
    # Compact format (current corpus)
    assert pat.search("(1) Security lead [4401/5m]")
    # Original verbose format
    assert pat.search("Security lead (ext 4401, SLA 5 min)")
    # Doesn't match arbitrary brackets
    assert not pat.search("(1) Security lead")


def test_observation_phrase_pattern():
    from tools.coverage_audit import SIGNALS
    pat = SIGNALS["observation_phrase"]["pattern"]
    assert pat.search("Patient SYS observed +2.5 mmHg vs baseline")
    assert pat.search("observed -1.3 °C")
    assert not pat.search("abnormal blood pressure")


def test_mitre_gloss_pattern_requires_em_dash():
    from tools.coverage_audit import SIGNALS
    pat = SIGNALS["mitre_id_with_gloss"]["pattern"]
    # With gloss
    assert pat.search("Consistent with MITRE T1565 (Data Manipulation — an attacker silently changes data)")
    # Without gloss (no em-dash after the name)
    assert not pat.search("Consistent with MITRE T1565 (Data Manipulation).")


def test_counterfactual_clause_pattern():
    from tools.coverage_audit import SIGNALS
    pat = SIGNALS["counterfactual_clause"]["pattern"]
    assert pat.search("This alert would clear if Flgs dropped")
    assert not pat.search("This alert was triggered")


def test_playbook_pattern():
    from tools.coverage_audit import SIGNALS
    pat = SIGNALS["playbook_markdown"]["pattern"]
    assert pat.search("\n\n**Playbook: biometric_anomaly**")
    assert not pat.search("Just regular markdown")


def test_stability_badge_pattern():
    from tools.coverage_audit import SIGNALS
    pat = SIGNALS["stability_badge"]["pattern"]
    assert pat.search("🟢 Explanation: STABLE — top features are robust")
    assert pat.search("Explanation: UNSTABLE")
    assert not pat.search("The explanation is good")


# ── Full audit smoke test ─────────────────────────────────────────


@pytest.mark.parametrize("split", ["test", "demo"])
def test_audit_runs_on_both_splits(split):
    """End-to-end: the audit must produce a report for both splits
    with all expected signal entries."""
    from tools.coverage_audit import SIGNALS, audit
    report = audit(split)
    assert report["_split"] == split
    assert set(report["signals"].keys()) == set(SIGNALS.keys())
    assert report["_meta"]["n_signals"] == len(SIGNALS)


@pytest.mark.parametrize("split", ["test", "demo"])
def test_audit_passes_on_current_artifacts(split):
    """Sprint 3 acceptance: the audit must pass on the current
    regenerated artifacts. If this fires after a phase change, an
    expected signal stopped emitting (or the regex drifted)."""
    from tools.coverage_audit import audit
    report = audit(split)
    failing = [name for name, b in report["signals"].items() if b["below_floor"]]
    assert not failing, (
        f"Coverage audit failing on {split!r}: {failing}. "
        f"Either the writer regressed or the regex needs updating."
    )
