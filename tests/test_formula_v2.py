"""Tests for the v2 risk-composition formula (Sprint 4 / Tầng 3.1).

Pins the architectural invariants the upgrade-plan committed to and
catches regressions against the v1 paper artifacts:

  - v1 emits the legacy linear-sum behaviour byte-for-byte (so the
    RQ1 paper is reproducible from the same code path)
  - v2 implements gate + amplify and never raises an alert from
    zero detection (the bug v2 was designed to fix)
  - v1 and v2 thresholds disambiguate when both are loaded
  - the npz writer records ``formula_version`` so a downstream
    consumer can tell what flavour produced the artifact
"""
from __future__ import annotations

import numpy as np
import pytest


# ── v1 byte-exact reproduction (paper artifact stability) ─────────


def test_v1_formula_matches_paper_weighted_sum():
    """v1 must remain bit-for-bit identical to the original linear
    weighted sum so the RQ1 paper numbers stay reproducible."""
    from module3_risk_scoring.composition import compute_composite_risk
    from module3_risk_scoring.config import WEIGHTS

    c = np.array([0.5]); d = np.array([0.6])
    s = np.array([0.7]); t = np.array([0.8])
    R = compute_composite_risk(c, d, s, t, formula_version="v1")
    expected = (
        WEIGHTS["w1"] * 0.5 + WEIGHTS["w2"] * 0.6
        + WEIGHTS["w3"] * 0.7 + WEIGHTS["w4"] * 0.8
    )
    assert R[0] == pytest.approx(expected)


def test_v1_default_when_formula_version_unspecified_is_v1():
    """Existing call sites pass no ``formula_version`` — they must
    keep getting v1 behaviour unchanged."""
    from module3_risk_scoring.composition import compute_composite_risk
    c = np.array([0.5]); d = np.array([0.6])
    s = np.array([0.7]); t = np.array([0.8])
    R_default = compute_composite_risk(c, d, s, t)
    R_v1      = compute_composite_risk(c, d, s, t, formula_version="v1")
    assert np.allclose(R_default, R_v1)


# ── v2 architectural invariants ───────────────────────────────────


def test_v2_returns_zero_when_detection_below_gate():
    """The core fix: v2 must NEVER promote a sample with negligible
    detection signal regardless of how high the context components
    are. This is the bug v1 had with idle vital-monitoring devices."""
    from module3_risk_scoring.composition import compute_composite_risk
    from module3_risk_scoring.config import MIN_DETECTION_GATE
    c = np.array([MIN_DETECTION_GATE - 1e-6])
    d = np.array([1.0]); s = np.array([1.0]); t = np.array([1.0])
    R = compute_composite_risk(c, d, s, t, formula_version="v2")
    assert R[0] == 0.0


def test_v2_amplifies_only_when_detection_present():
    """When detection signal is non-zero, context amplifies the
    score; when context is zero, R equals C_detect."""
    from module3_risk_scoring.composition import compute_composite_risk
    c = np.array([0.5, 0.5])
    no_context = np.array([0.0, 0.0])
    full_context = np.array([1.0, 1.0])
    R_no_ctx   = compute_composite_risk(c, no_context,   no_context,   no_context,   formula_version="v2")
    R_full_ctx = compute_composite_risk(c, full_context, full_context, full_context, formula_version="v2")
    assert R_no_ctx[0]   == pytest.approx(0.5)        # no amplification
    assert R_full_ctx[0] > R_no_ctx[0]                 # amplified
    # Clipped to [0, 1]
    assert R_full_ctx[0] <= 1.0


def test_v2_clipped_to_unit_interval_for_extreme_inputs():
    from module3_risk_scoring.composition import compute_composite_risk
    c = np.array([1.0, 0.5])
    d = np.array([1.0, 1.0])
    s = np.array([1.0, 1.0])
    t = np.array([1.0, 1.0])
    R = compute_composite_risk(c, d, s, t, formula_version="v2")
    assert (R >= 0).all() and (R <= 1).all()


def test_v2_passes_detection_through_when_context_zero():
    """When context components are all zero, v2 reduces to
    ``R = c_detect`` (apart from the gate). Guards against accidental
    re-introduction of a context floor."""
    from module3_risk_scoring.composition import compute_composite_risk
    c = np.array([0.1, 0.3, 0.7])
    z = np.zeros_like(c)
    R = compute_composite_risk(c, z, z, z, formula_version="v2")
    assert np.allclose(R, c)


def test_v2_explicit_amplification_factor():
    """For known inputs the v2 R must equal the closed-form
    ``c_detect × (1 + α·D_crit + β·S_data + γ·D_clinical_tier)``
    formula (then clipped)."""
    from module3_risk_scoring.composition import compute_composite_risk
    from module3_risk_scoring.config import CONTEXT_WEIGHTS_V2
    α, β, γ = (CONTEXT_WEIGHTS_V2[k] for k in ("alpha", "beta", "gamma"))
    c, d, s, t = 0.4, 0.5, 0.6, 0.3
    R = compute_composite_risk(
        np.array([c]), np.array([d]), np.array([s]), np.array([t]),
        formula_version="v2",
    )
    expected = min(1.0, c * (1 + α*d + β*s + γ*t))
    assert R[0] == pytest.approx(expected)


# ── Tier-table dispatch ─────────────────────────────────────────


def test_assign_risk_levels_v2_uses_v2_thresholds():
    """When ``formula_version="v2"`` is passed, the default threshold
    table is RISK_THRESHOLDS_V2, not the v1 table."""
    from module3_risk_scoring.composition import assign_risk_levels
    from module3_risk_scoring.config import RISK_THRESHOLDS_V2
    by_name = {name: t for t, name in RISK_THRESHOLDS_V2}
    # An R value just above v2's MEDIUM cutoff and below v1's MEDIUM
    # cutoff (0.40). Under v2, it's MEDIUM; under v1 it'd be LOW or NORMAL.
    R = np.array([by_name["MEDIUM"] + 0.001])
    out_v2 = assign_risk_levels(R, formula_version="v2")
    out_v1 = assign_risk_levels(R, formula_version="v1")
    assert out_v2[0] == "MEDIUM"
    assert out_v1[0] != "MEDIUM"  # would be lower (LOW or NORMAL)


def test_unsupported_formula_version_raises():
    from module3_risk_scoring.composition import compute_composite_risk
    with pytest.raises(ValueError):
        compute_composite_risk(
            np.array([0.5]), np.array([0.5]),
            np.array([0.5]), np.array([0.5]),
            formula_version="v99",
        )


# ── Threshold table consistency ─────────────────────────────────


def test_v2_thresholds_monotonic_descending():
    from module3_risk_scoring.config import RISK_THRESHOLDS_V2
    thresholds = [t for t, _ in RISK_THRESHOLDS_V2]
    assert thresholds == sorted(thresholds, reverse=True)
    assert all(0.0 < t < 1.0 for t in thresholds)


def test_v2_threshold_labels_match_v1_vocabulary():
    """v2 must preserve the same tier vocabulary so downstream
    consumers (dashboard, response engine) don't need to special-case."""
    from module3_risk_scoring.config import RISK_THRESHOLDS, RISK_THRESHOLDS_V2
    v1_labels = {name for _, name in RISK_THRESHOLDS}
    v2_labels = {name for _, name in RISK_THRESHOLDS_V2}
    assert v1_labels == v2_labels


# ── npz formula_version field ───────────────────────────────────


def test_npz_round_trip_records_formula_version(tmp_path):
    """The npz must carry a ``formula_version`` field so loaders
    know which interpretation applies."""
    from module3_risk_scoring.io import save_outputs
    import numpy as np

    n = 5
    args = [np.zeros(n)] * 7 + [np.array(["LOW"] * n)] + [np.zeros(n, dtype=int), np.zeros(n)]
    save_outputs(
        *args[:10],
        fusion={}, contributions={}, sensitivity={}, worked_examples=[],
        out_npz=tmp_path / "scores.npz",
        formula_version="v2",
    )
    loaded = np.load(tmp_path / "scores.npz", allow_pickle=True)
    assert "formula_version" in loaded.files
    assert str(loaded["formula_version"]) == "v2"


# ── Hypothesis-style property: v2 monotonic in c_detect ─────────


def test_v2_monotonic_in_c_detect():
    """For fixed context, increasing C_detect must never decrease R."""
    from module3_risk_scoring.composition import compute_composite_risk
    c_grid = np.linspace(0.0, 1.0, 50)
    d = np.full_like(c_grid, 0.5)
    s = np.full_like(c_grid, 0.5)
    t = np.full_like(c_grid, 0.5)
    R = compute_composite_risk(c_grid, d, s, t, formula_version="v2")
    diffs = np.diff(R)
    assert (diffs >= -1e-12).all(), "v2 must be monotonic non-decreasing in c_detect"


def test_v2_monotonic_in_context():
    """For fixed C_detect ≥ gate, increasing any context component
    must never decrease R."""
    from module3_risk_scoring.composition import compute_composite_risk
    n = 50
    c = np.full(n, 0.5)
    d_grid = np.linspace(0.0, 1.0, n)
    z = np.zeros(n)
    R = compute_composite_risk(c, d_grid, z, z, formula_version="v2")
    diffs = np.diff(R)
    assert (diffs >= -1e-12).all(), "v2 must be monotonic non-decreasing in D_crit"
