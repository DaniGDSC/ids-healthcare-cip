"""RQ1 R3 — Weight sensitivity uses canonical weights, not approximations.

Verifies that `tools/rq1_compute_metrics.py::compute_weight_sensitivity`:
  1. Imports canonical baseline from
     `module3_risk_scoring.module3_risk_scores.WEIGHTS`.
  2. The canonical baseline appears as exactly one row in the grid.
  3. The surfacing threshold = canonical MEDIUM cutoff (RISK_THRESHOLDS[-1]).
  4. FNR_critical invariance holds across the grid (defendability claim).
  5. Module 3 weights have not silently drifted.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

ARTIFACT = PROJECT_ROOT / "results/rq1_weight_sensitivity.json"


@pytest.fixture(scope="module")
def sensitivity():
    if not ARTIFACT.exists():
        pytest.skip(f"{ARTIFACT} missing — run tools/rq1_compute_metrics.py")
    with open(ARTIFACT) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def canonical_weights():
    """Import the canonical weights directly from Module 3."""
    from module3_risk_scoring.module3_risk_scores import WEIGHTS
    return WEIGHTS


def test_module3_canonical_weights_unchanged(canonical_weights):
    """Canonical w1..w4 must match the values RQ1 was computed against.

    Any drift here means the weight sensitivity report is stale and
    needs re-running. Pinning these in the test catches silent
    pipeline-side changes.
    """
    expected = {"w1": 0.40, "w2": 0.25, "w3": 0.15, "w4": 0.20}
    assert canonical_weights == expected, (
        f"Module 3 WEIGHTS drifted from RQ1's reference: {canonical_weights} vs {expected}"
    )


def test_baseline_canonical_metadata_present(sensitivity):
    """The artifact must declare which baseline it used + where it came from."""
    meta = sensitivity["_meta"]
    assert "baseline_canonical" in meta, "missing baseline_canonical key"
    assert "baseline_source" in meta, "baseline_source must cite Module 3"
    assert "module3" in meta["baseline_source"].lower()


def test_baseline_matches_canonical(sensitivity, canonical_weights):
    """The artifact's declared baseline must equal Module 3's WEIGHTS
    (mapped from w1..w4 to alpha/beta/gamma/delta).
    """
    bl = sensitivity["_meta"]["baseline_canonical"]
    expected = {
        "alpha": canonical_weights["w1"],
        "beta":  canonical_weights["w2"],
        "gamma": canonical_weights["w3"],
        "delta": canonical_weights["w4"],
    }
    assert bl == expected, f"baseline_canonical drift: {bl} vs {expected}"


def test_canonical_row_present_in_grid(sensitivity):
    """The canonical baseline must be exactly one row of the grid so
    reviewers can read sensitivity around the actual operating point.
    """
    canon_rows = [g for g in sensitivity["grid"] if g.get("is_canonical")]
    assert len(canon_rows) == 1, (
        f"expected 1 canonical row, got {len(canon_rows)}"
    )
    # The dedicated convenience field should match.
    assert sensitivity["canonical_baseline_row"] == canon_rows[0]


def test_surfacing_threshold_canonical(sensitivity):
    """Surfacing threshold must equal the canonical MEDIUM cutoff (0.40)
    — the boundary above which alerts are shown to operators.

    Previously this looked up ``RISK_THRESHOLDS[-1]`` which was MEDIUM
    while there were 3 thresholds. After the formula-fix upgrade added
    a 4th threshold for NORMAL at the bottom, ``RISK_THRESHOLDS[-1]``
    is the LOW cutoff (0.30). We now look up MEDIUM by name to keep
    the assertion semantic-stable across future threshold additions.
    """
    from module3_risk_scoring.module3_risk_scores import RISK_THRESHOLDS
    medium_threshold = next(t for t, name in RISK_THRESHOLDS if name == "MEDIUM")
    artifact_threshold = sensitivity["_meta"]["surfacing_threshold_on_R"]
    assert abs(artifact_threshold - medium_threshold) < 1e-9, (
        f"surfacing threshold {artifact_threshold} != canonical MEDIUM "
        f"cutoff {medium_threshold}"
    )


def test_fnr_critical_invariance(sensitivity):
    """The headline RQ1 finding — FNR_critical is robust across weight
    choices — must hold. Test asserts ALL grid rows have FNR_critical=0.
    A regression here breaks the safety-property defendability claim.
    """
    fnr_values = {g["FNR_critical"] for g in sensitivity["grid"]}
    assert fnr_values == {0.0}, (
        f"FNR_critical no longer invariant across grid: {fnr_values}"
    )


def test_canonical_row_meets_targets(sensitivity):
    """At the canonical operating point: FNR_critical = 0 + AUC ≥ 0.97."""
    row = sensitivity["canonical_baseline_row"]
    assert row["FNR_critical"] == 0.0
    assert row["AUC"] >= 0.97, f"canonical AUC dropped: {row['AUC']}"


def test_grid_size_sane(sensitivity):
    """Sanity: grid should be non-empty and not pathologically small."""
    grid = sensitivity["grid"]
    assert len(grid) >= 20, f"grid too small ({len(grid)}); perturbation logic regressed?"
