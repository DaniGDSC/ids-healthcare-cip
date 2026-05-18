"""Tests for the RQ2 compliance manifest (RQ2_Compliance.md §7.3).

Gates ``results/rq2_compliance_audit.json`` produced by
``analysis.make_rq2_compliance_table``.  Skips when the audit JSON is
absent so the regression suite stays runnable on a fresh checkout.

Required vs pending evidence:
  * ``evidence_files``   — must exist; this test FAILS if missing.
  * ``evidence_pending`` — informational; no assertion.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

OUT = Path(__file__).resolve().parents[1] / "results/rq2_compliance_audit.json"


@pytest.fixture(scope="module")
def audit() -> dict:
    if not OUT.exists():
        pytest.skip("Run analysis/make_rq2_compliance_table.py first")
    return json.loads(OUT.read_text(encoding="utf-8"))


def test_all_required_evidence_files_exist(audit: dict) -> None:
    missing = [
        (e["id"], e["evidence_required_missing"])
        for e in audit["evidence_audit"]
        if e["evidence_required_missing"]
    ]
    assert not missing, (
        f"Compliance manifest references missing REQUIRED evidence: "
        f"{missing}.  Either add the file or move it to evidence_pending."
    )


def test_every_requirement_has_required_evidence(audit: dict) -> None:
    empty = [
        e["id"] for e in audit["evidence_audit"]
        if e["evidence_required_total"] == 0
    ]
    assert not empty, (
        f"Requirements with zero REQUIRED evidence files: {empty}.  "
        "Each requirement must point at least one auditable artifact."
    )


def test_manifest_last_validated_present(audit: dict) -> None:
    assert audit["_meta"]["last_validated"], (
        "rq2_compliance_manifest.yaml needs last_validated set "
        "(reviewed annually + on schema change)."
    )


def test_manifest_schema_version_present(audit: dict) -> None:
    assert audit["_meta"]["manifest_schema_version"], (
        "Manifest must declare schema_version for future migrations."
    )
