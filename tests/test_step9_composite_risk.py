"""ARCHITECTURE.md Step [9] — Composite risk scoring contract tests.

Locks the doc-promised invariants for the composite-risk formula:

* I1 ``R = w_C·C_detect + w_dcrit·D_crit + w_sdata·S_data + w_dclin·D_clinical_tier``
* I2 Weights MUST sum to 1.0; loaded from
     ``configs/composite_risk_weights.yaml``.
* I3 Tier mapping: CRITICAL ≥ 0.80 / HIGH ≥ 0.60 / MEDIUM ≥ 0.40 /
     LOW < 0.40. Boundaries from the YAML; descending.
* I4 R clipped to [0, 1].
* I5 Per-alert audit: every ScoredAlert / dashboard alert carries
     the four R components (c_detect, d_crit, s_data,
     d_clinical_tier) so "why was this CRITICAL?" is answerable
     from the persisted artifacts.
* I6 Sensitivity: tier assignment stable for ≥ 80% of alerts under
     ±20% perturbation of any single weight (axiom-level claim;
     paper Section 11 reports the exact percentage).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from module3_risk_scoring.module3_risk_scores import (
    WEIGHTS,
    RISK_THRESHOLDS,
    assign_risk_levels,
    compute_composite_risk,
    load_composite_weights,
    load_tier_boundaries,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ── I1: formula correctness on known fixtures ─────────────────────────


def test_formula_matches_doc_for_known_inputs():
    c_detect = np.array([0.8])
    d_crit = np.array([0.5])
    s_data = np.array([0.4])
    d_clinical = np.array([0.6])
    R = compute_composite_risk(c_detect, d_crit, s_data, d_clinical)
    expected = (
        WEIGHTS["w1"] * 0.8
        + WEIGHTS["w2"] * 0.5
        + WEIGHTS["w3"] * 0.4
        + WEIGHTS["w4"] * 0.6
    )
    assert abs(float(R[0]) - expected) < 1e-9


# ── I2: weights sum to 1.0 ───────────────────────────────────────────


def test_weights_sum_to_one():
    w = load_composite_weights()
    assert abs(sum(w.values()) - 1.0) < 1e-6


def test_weights_yaml_is_canonical_source():
    yaml_path = PROJECT_ROOT / "configs" / "composite_risk_weights.yaml"
    assert yaml_path.exists()
    # The module-level WEIGHTS dict must equal what the YAML loader
    # returns at import time.
    assert WEIGHTS == load_composite_weights()


# ── I3: tier boundaries descending ───────────────────────────────────


def test_tier_boundaries_descending():
    bounds = load_tier_boundaries()
    mins = [m for m, _ in bounds]
    assert mins == sorted(mins, reverse=True)


def test_tier_boundaries_match_doc_defaults():
    bounds = dict((tier, m) for m, tier in RISK_THRESHOLDS)
    assert bounds["CRITICAL"] == 0.80
    assert bounds["HIGH"] == 0.60
    assert bounds["MEDIUM"] == 0.40


@pytest.mark.parametrize("R,expected", [
    (0.95, "CRITICAL"),
    (0.80, "CRITICAL"),  # boundary inclusive
    (0.79, "HIGH"),
    (0.60, "HIGH"),
    (0.59, "MEDIUM"),
    (0.40, "MEDIUM"),
    (0.39, "LOW"),
    (0.0,  "LOW"),
])
def test_tier_boundary_edges(R: float, expected: str):
    levels = assign_risk_levels(np.array([R]))
    assert str(levels[0]) == expected


# ── I4: R clipped to [0, 1] ──────────────────────────────────────────


def test_compute_composite_risk_clips_to_unit_interval():
    n = 1000
    rng = np.random.default_rng(0)
    R = compute_composite_risk(
        rng.uniform(-1, 2, n),
        rng.uniform(-1, 2, n),
        rng.uniform(-1, 2, n),
        rng.uniform(-1, 2, n),
    )
    assert R.min() >= 0.0
    assert R.max() <= 1.0


# ── I5: persisted audit fields ───────────────────────────────────────


def test_demo_scores_npz_persists_all_R_components():
    """Doc invariant: ``demo_scores.npz`` must carry c_detect, d_crit,
    s_data, d_clinical_tier, R, risk_levels — so a reviewer can
    answer "why was this alert CRITICAL?" from the persisted artifact
    without re-running M3."""
    npz_path = PROJECT_ROOT / "results" / "reports" / "demo_scores.npz"
    if not npz_path.exists():
        pytest.skip(f"{npz_path} missing — run module3_demo_scores first")
    data = np.load(npz_path, allow_pickle=True)
    for k in ("c_detect", "d_crit", "s_data", "d_clinical_tier",
              "R", "risk_levels"):
        assert k in data.files, f"demo_scores.npz missing {k!r}"


# ── I6: sensitivity under ±20% weight perturbation ───────────────────


def test_tier_assignment_stable_under_20pct_weight_perturbation():
    """≥ 80% of alerts retain their tier under a single-weight ±20%
    perturbation (axiom-level claim per the doc's Section 11)."""
    rng = np.random.default_rng(7)
    n = 500
    c_detect = rng.uniform(0, 1, n)
    d_crit = rng.uniform(0, 1, n)
    s_data = rng.uniform(0, 1, n)
    d_clinical = rng.uniform(0, 1, n)

    R_base = compute_composite_risk(c_detect, d_crit, s_data, d_clinical)
    levels_base = assign_risk_levels(R_base)

    for wkey in ("w1", "w2", "w3", "w4"):
        for delta in (-0.20, +0.20):
            w_pert = dict(WEIGHTS)
            w_pert[wkey] = w_pert[wkey] * (1.0 + delta)
            # Re-normalize to preserve sum=1.0
            total = sum(w_pert.values())
            w_pert = {k: v / total for k, v in w_pert.items()}

            R_pert = compute_composite_risk(
                c_detect, d_crit, s_data, d_clinical, weights=w_pert,
            )
            levels_pert = assign_risk_levels(R_pert)
            agreement = (levels_base == levels_pert).mean()
            assert agreement >= 0.80, (
                f"Tier-assignment agreement under {wkey} {delta:+.0%} "
                f"perturbation = {agreement:.3f} < 0.80 threshold"
            )


# ── RQ1_pipeline.md §4.1 — npz schema v1.1 contract ────────────────────

def test_risk_scores_npz_schema_v1_1():
    """The persisted risk_scores.npz must expose the seven RQ1-pipeline
    extension arrays plus a sidecar meta.json declaring schema_version 1.1.

    Per RQ1_pipeline.md §4.2.  Run Module 3 first to regenerate the npz
    if this test fails due to a stale (v1.0) artifact.
    """
    import json

    repo_root = Path(__file__).resolve().parents[1]
    npz_path = repo_root / "results/reports/risk_scores.npz"
    meta_path = repo_root / "results/reports/risk_scores.meta.json"
    if not npz_path.exists():
        pytest.skip(
            "risk_scores.npz not present — "
            "run `python -m module3_risk_scoring.module3_risk_scores` first."
        )
    # NPZ present without sidecar → stale v1.0 schema; this is exactly the
    # contract-violation case the test is meant to catch.  Fail loudly so
    # CI surfaces it instead of silently skipping.
    assert meta_path.exists(), (
        "risk_scores.npz exists but risk_scores.meta.json is missing — "
        "the npz is the legacy v1.0 schema; re-run Module 3 to regenerate."
    )

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["schema_version"] == "1.1", (
        f"Expected schema v1.1, got {meta.get('schema_version')!r}. "
        "Re-run Module 3 to regenerate."
    )

    data = np.load(npz_path, allow_pickle=False)
    required = {
        "row_id", "attack_category", "device_class",
        "device_criticality", "patchable", "true_severity",
        "R_counterfactual",
    }
    missing = required - set(data.files)
    assert not missing, f"Missing required arrays: {missing}"

    # All arrays same length
    n = len(data["y_true"])
    for name in required:
        assert len(data[name]) == n, (
            f"{name} length {len(data[name])} != {n}"
        )

    # Invariant 1: R_counterfactual >= R - eps (counterfactual is upper bound).
    assert np.all(data["R_counterfactual"] + 1e-9 >= data["R"]), (
        "R_counterfactual must be >= R for every row"
    )

    # Invariant 2: row_id is the identity range over test parquet rows
    # (Module 3 must not shuffle/filter between parquet → npz).
    assert np.array_equal(data["row_id"], np.arange(n, dtype=data["row_id"].dtype)), (
        "row_id must be identity range over test parquet rows"
    )
