"""Hard-fail CI test for MVE word-budget compliance
(RQ2_Compliance.md §5.4).

Gates ``results/rq2_word_budget_audit.json`` produced by
``analysis.audit_word_budgets``.  Skips when the audit JSON is absent
so the regression suite stays runnable on a fresh checkout.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

OUT = Path(__file__).resolve().parents[1] / "results/rq2_word_budget_audit.json"


@pytest.fixture(scope="module")
def audit() -> dict:
    if not OUT.exists():
        pytest.skip("Run analysis/audit_word_budgets.py first")
    return json.loads(OUT.read_text(encoding="utf-8"))


def test_audit_pass(audit: dict) -> None:
    h = audit["headline"]
    assert h["audit_pass"], (
        f"Word-budget audit failed: "
        f"{h['n_records_with_violations']} / {h['n_records']} records "
        "exceeded budget.  See results/rq2_word_budget_audit.json::"
        "violations for diagnostics."
    )


def test_no_total_budget_overflow(audit: dict) -> None:
    """Even with per-layer slack the total must never exceed."""
    over = audit["total_stats"]["n_over"]
    assert over == 0, (
        f"{over} records exceed TOTAL word budget "
        f"({audit['_meta']['config']['total_budget']}).  Investigate "
        "truncation logic in src/mve_generator.py."
    )


def test_no_per_layer_overflow(audit: dict) -> None:
    """Per-layer counts must respect the MVEOutput contract."""
    offenders = {
        layer: stats["n_over"]
        for layer, stats in audit["per_layer_stats"].items()
        if stats["n_over"] > 0
    }
    assert not offenders, (
        f"Per-layer budget violations: {offenders}.  "
        "See per_layer_stats + violations[*].per_layer_counts in "
        "results/rq2_word_budget_audit.json."
    )
