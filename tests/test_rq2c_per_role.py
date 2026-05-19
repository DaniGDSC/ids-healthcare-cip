"""Schema + sanity tests for the RQ2.c per-role analysis output."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "analysis" / "outputs" / "rq2c_per_role.json"

ROLES = ("biomed_engineer", "IT_generalist", "nurse_manager")
METRICS = ("accuracy", "confidence")


@pytest.fixture(scope="module")
def result() -> dict:
    if not OUT.exists():
        pytest.skip("Run analysis/compute_rq2c_per_role.py first")
    return json.loads(OUT.read_text())


def test_schema_complete(result: dict) -> None:
    for key in ("_meta", "methodology_notes", "limitations",
                "overall", "per_role", "cell_diagnostics"):
        assert key in result, f"Missing top-level key: {key}"


def test_methodology_notes_disclose_no_correction(result: dict) -> None:
    text = " ".join(result["methodology_notes"]).lower()
    assert ("no multiple-comparisons correction" in text
            or "no multiple comparisons correction" in text), (
        "Methodology must explicitly disclose absence of multiple-comparisons "
        "correction (defense-critical transparency)."
    )


def test_methodology_notes_disclose_llm_persona(result: dict) -> None:
    text = " ".join(result["methodology_notes"]).lower()
    assert "llm-persona" in text or "llm persona" in text, (
        "Methodology must disclose LLM-persona data source (defense-critical)."
    )


def test_limitations_disclose_multiple_comparisons(result: dict) -> None:
    text = " ".join(result["limitations"]).lower()
    assert "multiple comparisons" in text, (
        "Limitations must call out the multiple-comparisons issue."
    )


def test_limitations_disclose_llm_persona(result: dict) -> None:
    text = " ".join(result["limitations"]).lower()
    assert ("llm persona" in text or "llm-persona" in text
            or "personas, not humans" in text), (
        "Limitations must call out the LLM-persona vs human gap."
    )


def test_overall_has_two_metrics(result: dict) -> None:
    for m in METRICS:
        assert m in result["overall"], f"Missing metric in overall: {m}"


def test_all_three_roles_present(result: dict) -> None:
    for role in ROLES:
        assert role in result["per_role"], f"Missing role: {role}"


def test_p_values_in_valid_range(result: dict) -> None:
    def check(cell: dict) -> None:
        p = cell.get("p_value")
        if p is not None:
            assert 0 <= p <= 1, f"Invalid p-value: {p}"

    for m in METRICS:
        check(result["overall"][m])
        for role in ROLES:
            entry = result["per_role"].get(role, {})
            if isinstance(entry, dict) and m in entry:
                check(entry[m])


def test_cliffs_delta_in_valid_range(result: dict) -> None:
    def check(cell: dict) -> None:
        d = cell.get("cliffs_delta")
        if d is not None:
            assert -1 <= d <= 1, f"Invalid Cliff's delta: {d}"

    for m in METRICS:
        check(result["overall"][m])
        for role in ROLES:
            entry = result["per_role"].get(role, {})
            if isinstance(entry, dict) and m in entry:
                check(entry[m])
