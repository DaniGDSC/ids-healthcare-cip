"""Tests for the operational_health metric (Sprint 2.4).

The metric is the bottom-up sentinel that would have caught the
formula-bug "context-driven 91% LOW spam" before we measured it
manually. Tests pin:

  - operational precision shape (TP / total emitted)
  - LOW-tier attack density sentinel
  - empty-pool edge case
  - surfaced (RQ1) slice matches the convention
"""
from __future__ import annotations

import pytest


def _records(severities_truths: list[tuple[str, str]]) -> list[dict]:
    return [
        {"sample_index": i, "risk_level": sev, "ground_truth": gt}
        for i, (sev, gt) in enumerate(severities_truths)
    ]


# ── operational_precision shape ─────────────────────────────────


def test_operational_precision_perfect_pool():
    from module4_explanations.phase0_metrics import compute_operational_health
    records = _records([
        ("CRITICAL", "attack"),
        ("HIGH",     "attack"),
        ("MEDIUM",   "attack"),
        ("LOW",      "attack"),
    ])
    out = compute_operational_health(records)
    assert out["operational_precision"] == 1.0
    assert out["fp_pool"] == 0


def test_operational_precision_zero_attacks():
    from module4_explanations.phase0_metrics import compute_operational_health
    records = _records([
        ("LOW", "benign"),
        ("LOW", "benign"),
        ("LOW", "benign"),
    ])
    out = compute_operational_health(records)
    assert out["operational_precision"] == 0.0
    assert out["fp_pool"] == 3


def test_operational_precision_pre_fix_scenario():
    """The pre-formula-fix bug pattern: ~85% of records are LOW
    benign noise. operational_precision should reflect the
    contamination — closer to 15%, not the 80%+ "surfaced precision"
    that the RQ1 metric reports."""
    from module4_explanations.phase0_metrics import compute_operational_health
    records = (
        _records([("LOW", "benign")] * 86)  # noise floor
        + _records([("HIGH", "attack")] * 14)  # real attacks
    )
    out = compute_operational_health(records)
    assert 0.10 < out["operational_precision"] < 0.20
    # but surfaced precision (MEDIUM+ only) is perfect
    assert out["surfaced_precision"] == 1.0


# ── LOW-tier sentinel ────────────────────────────────────────────


def test_low_tier_attack_density_high_when_low_pool_clean():
    from module4_explanations.phase0_metrics import compute_operational_health
    records = _records([
        ("LOW", "attack"),
        ("LOW", "attack"),
        ("LOW", "benign"),
    ])
    out = compute_operational_health(records)
    assert out["low_tier_attack_density"] == pytest.approx(2/3, abs=0.01)


def test_low_tier_attack_density_near_zero_on_context_spam():
    """The diagnostic the formula bug would have tripped: a flood of
    benign LOW alerts pulls the density to near zero."""
    from module4_explanations.phase0_metrics import compute_operational_health
    records = _records([("LOW", "benign")] * 100 + [("LOW", "attack")] * 1)
    out = compute_operational_health(records)
    assert out["low_tier_attack_density"] < 0.02


def test_low_tier_attack_density_no_low_records():
    from module4_explanations.phase0_metrics import compute_operational_health
    records = _records([("HIGH", "attack")])
    out = compute_operational_health(records)
    assert out["low_tier_attack_density"] == 0.0
    assert out["low_tier_n"] == 0


# ── Empty pool ───────────────────────────────────────────────────


def test_operational_health_empty_pool():
    from module4_explanations.phase0_metrics import compute_operational_health
    out = compute_operational_health([])
    assert out["n"] == 0


# ── Surfaced slice matches RQ1 convention ───────────────────────


def test_surfaced_slice_excludes_low():
    from module4_explanations.phase0_metrics import compute_operational_health
    records = _records([
        ("CRITICAL", "attack"),
        ("HIGH",     "attack"),
        ("MEDIUM",   "attack"),
        ("LOW",      "benign"),
        ("LOW",      "benign"),
    ])
    out = compute_operational_health(records)
    # All MEDIUM+ are attacks → surfaced precision 1.0
    assert out["surfaced_precision"] == 1.0
    # But operational precision dragged by the benign LOWs
    assert out["operational_precision"] == 0.6
