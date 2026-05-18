"""Smoke + invariant tests for the MITRE config audit (RQ2_Mitre.md §6.1).

Gates ``results/rq2_mitre_audit.json`` produced by
``analysis.audit_mitre_config``.  Tests skip when the audit JSON is
absent so the regression suite stays runnable on a fresh checkout.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

OUT = Path(__file__).resolve().parents[1] / "results/rq2_mitre_audit.json"


@pytest.fixture(scope="module")
def audit() -> dict:
    if not OUT.exists():
        pytest.skip("Run analysis/audit_mitre_config.py first")
    return json.loads(OUT.read_text(encoding="utf-8"))


def test_audit_passed(audit: dict) -> None:
    assert audit["headline"]["audit_pass"], (
        f"MITRE audit failed with {audit['headline']['n_fail']} FAIL "
        f"findings.  See results/rq2_mitre_audit.json for details."
    )


def test_no_orphan_categories(audit: dict) -> None:
    orphans = audit["headline"]["orphan_categories"]
    assert not orphans, (
        f"Attack categories without MITRE mapping: {orphans}"
    )


def test_framework_version_pinned(audit: dict) -> None:
    fv = audit["headline"]["mitre_framework_version"]
    assert fv, "mitre_framework_version is required (RQ2_expected_outputs.md §5.2)"


def test_last_validated_present(audit: dict) -> None:
    """Top-level OR per-mapping last_validated must be set."""
    h = audit["headline"]
    has_top = bool(h.get("last_validated_top_level"))
    has_per_entry = int(h.get("last_validated_per_entry_count", 0) or 0) > 0
    assert has_top or has_per_entry, (
        "last_validated must be set top-level or per-mapping"
    )


def test_all_techniques_have_valid_tid(audit: dict) -> None:
    """A6 must be clean: no T-IDs outside the MITRE pattern."""
    findings = audit.get("findings", [])
    a6_warns = [f for f in findings if f.get("check_id") == "A6"]
    assert not a6_warns, (
        f"T-IDs failing the MITRE pattern: "
        f"{[f['details'] for f in a6_warns]}"
    )
