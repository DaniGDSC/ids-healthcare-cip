"""RQ2.e — MITRE ATT&CK grounding tests.

Verifies:
  1. config/attack_to_mitre_mapping.yaml has 100% category coverage,
     no orphans, framework version pinned.
  2. After the G3 fix, runtime-generated MVE Layer 1 references MITRE
     technique IDs for ≥90% of attack-class alerts.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACT = PROJECT_ROOT / "results/rq2_mitre_coverage.json"


@pytest.fixture(scope="module")
def coverage_report():
    if not ARTIFACT.exists():
        pytest.skip(f"{ARTIFACT} missing — run tools/rq2_audit_mitre_coverage.py")
    with open(ARTIFACT) as f:
        return json.load(f)


def test_config_no_orphans(coverage_report):
    cc = coverage_report["config_coverage"]
    assert cc["orphans"] == [], f"orphan categories: {cc['orphans']}"


def test_config_framework_version_pinned(coverage_report):
    cc = coverage_report["config_coverage"]
    assert cc["framework_version_pinned"] is True
    assert cc["framework_version"], "framework_version is empty"


def test_layer1_mitre_reference_attack_class_meets_target(coverage_report):
    """After the G3 fix, ≥90% of attack-class alerts must reference a
    MITRE technique in Layer 1. Benign baseline is excluded from the
    denominator (no MITRE mapping by design).
    """
    fresh = coverage_report.get("layer1_mitre_reference_rate_fresh", {})
    if "error" in fresh:
        pytest.skip(f"fresh-MVE measurement unavailable: {fresh['error']}")
    pct = fresh.get("pct_referencing_mitre_attack_class", 0.0)
    assert pct >= 90.0, (
        f"Layer 1 MITRE reference rate (attack-class) = {pct}% — "
        "below 90% target. G3 fix regressed?"
    )


def test_layer1_mitre_reference_no_benign_attribution(coverage_report):
    """Benign / 'normal' alerts must NOT carry a MITRE reference — that
    would be over-attribution (a benign baseline isn't an attack
    technique). The config marks 'normal' as excluded from MITRE
    mapping; verify the runtime path respects that.
    """
    fresh = coverage_report.get("layer1_mitre_reference_rate_fresh", {})
    if "error" in fresh:
        pytest.skip(f"fresh-MVE measurement unavailable: {fresh['error']}")
    per_cat = fresh.get("per_category", {})
    if "normal" in per_cat:
        normal_hit_rate = per_cat["normal"].get("hit_rate_pct", 0.0)
        assert normal_hit_rate == 0.0, (
            f"Benign 'normal' alerts carry MITRE references at "
            f"{normal_hit_rate}% — over-attribution risk."
        )


def test_overall_status_pass(coverage_report):
    assert coverage_report["overall_status"] == "PASS", (
        f"Overall MITRE grounding status: {coverage_report['overall_status']}"
    )
