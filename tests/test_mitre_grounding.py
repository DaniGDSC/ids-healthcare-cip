"""Smoke + invariant tests for MVE-MITRE grounding rate
(RQ2_Mitre.md §6.2).

Gates ``results/rq2_mitre_grounding.json`` produced by
``analysis.compute_mitre_grounding``.  Tests skip when the grounding
JSON is absent so the regression suite stays runnable on a fresh
checkout.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

OUT = Path(__file__).resolve().parents[1] / "results/rq2_mitre_grounding.json"


@pytest.fixture(scope="module")
def grounding() -> dict:
    if not OUT.exists():
        pytest.skip("Run analysis/compute_mitre_grounding.py first")
    return json.loads(OUT.read_text(encoding="utf-8"))


def test_schema_complete(grounding: dict) -> None:
    for key in (
        "_meta", "headline", "by_attack_category",
        "by_mode", "appendix_all_mve",
    ):
        assert key in grounding, f"missing top-level key: {key}"


def test_grounding_target(grounding: dict) -> None:
    h = grounding["headline"]
    assert h["pass"], (
        f"MITRE grounding rate {h.get('grounded_pct'):.4f} "
        f"below target {h.get('target')}.  See "
        f"results/rq2_mitre_grounding.json::failure_examples for diagnostics."
    )


def test_mode_b_near_perfect(grounding: dict) -> None:
    """Mode B is rule-based; if the template injects MITRE technique
    names or IDs by construction, grounding should be ≥99%.

    Spec assumes injection; reality depends on the current
    ``src.mve_generator`` template strings.  When this fails the action
    is on the generator, not the test.
    """
    by_mode = grounding.get("by_mode", {})
    b = by_mode.get("B_rule")
    if b is None or (b.get("n_evaluated") or 0) == 0:
        pytest.skip("No Mode B records in grounding evaluation")
    pct = b.get("grounded_pct")
    assert pct is not None and pct >= 0.99, (
        f"Mode B grounding {pct} below 0.99 — rule-based MVE template "
        "does not inject MITRE technique IDs / names into "
        "layer_1_why_anomalous.  Fix: extend src.mve_generator "
        "templates to reference the mapped technique for each "
        "attack_category."
    )


def test_per_category_no_zero_columns(grounding: dict) -> None:
    """No mapped category with ≥10 evaluated alerts should have 0%
    grounding — that would point to a mapping bug or a Mode B template
    that wholly ignores MITRE terms."""
    offenders: list[str] = []
    for cat, stats in grounding.get("by_attack_category", {}).items():
        if (stats.get("n_evaluated") or 0) >= 10 and (stats.get("grounded_pct") or 0) == 0:
            offenders.append(cat)
    assert not offenders, (
        f"Categories with 0% grounding (n≥10): {offenders} — investigate "
        "mapping or MVE template."
    )
