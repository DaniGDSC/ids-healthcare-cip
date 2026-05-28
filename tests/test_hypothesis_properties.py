"""Property-based tests (Sprint 2.3).

Sprint 2.3 introduces Hypothesis-driven property tests for the
Category 3 ("boundary conditions in real data") bug surfaces. Each
test asserts an *invariant* and lets Hypothesis search the input
space for a counterexample, instead of relying on a handful of
example cases. This catches the kind of edge cases we hit during
the upgrade work:

  - degenerate IQR distribution (Flgs benign-cluster at 0) → runaway
    "~4e9 IQR-widths" phrase
  - case-mismatched role names → silent skip in annotate_role
  - boundary R / c_detect values around tier thresholds
  - stability band boundary inclusivity

The properties are deliberately weak ("output is in expected set",
"never exceeds bound") so they don't pin specific numeric output
that legitimate calibration might change.
"""
from __future__ import annotations

import math
import re

import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings, strategies as st


# ── 1. observation_phrase invariants ──────────────────────────────


@settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    median=st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False),
    observed=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
)
def test_observation_phrase_degenerate_distribution_avoids_iqr_widths(
    median, observed,
):
    """The runtime bug we hit on ``Flgs``: when the IQR width is
    essentially zero (binary-like features in normalised space), the
    deviation ``(x - median)/IQR_width`` becomes astronomical and the
    phrase used to print "~4e9 IQR-widths". After the
    degenerate-distribution fallback, the phrase MUST switch to the
    qualitative "benign values cluster tightly" form and NOT include
    "IQR-widths" — that's the bug fix's invariant."""
    from module4_explanations.feature_groups import observation_phrase

    # Construct a degenerate-IQR baseline (width well below the 0.05
    # threshold). Hypothesis explores median + observed combos.
    baselines = {
        "F": {
            "median":   median,
            "iqr_low":  median,  # iqr_low == iqr_high → width = 0
            "iqr_high": median,
            "unit":     "",
            "decimal_places": 2,
            "is_biometric":   False,
            "n_benign":       100,
        },
    }
    phrase = observation_phrase("F", observed, baselines=baselines)

    # The fallback branch must fire — no IQR-widths token, and the
    # qualitative "cluster tightly" phrase must be present.
    assert "IQR-widths" not in phrase, (
        f"Runaway IQR-widths emitted on degenerate distribution. Phrase: {phrase!r}"
    )
    assert "cluster tightly" in phrase, (
        f"Degenerate fallback did not fire. Phrase: {phrase!r}"
    )


@given(observed=st.floats(min_value=-1e3, max_value=1e3, allow_nan=False))
def test_observation_phrase_returns_non_empty_when_baseline_exists(observed):
    """If a baseline is provided, the phrase must be non-empty
    regardless of the observed value — the user always gets a
    rendering of the comparison."""
    from module4_explanations.feature_groups import observation_phrase

    baselines = {
        "F": {"median": 0.0, "iqr_low": -0.5, "iqr_high": 0.5,
              "unit": "", "decimal_places": 2, "is_biometric": False, "n_benign": 100},
    }
    phrase = observation_phrase("F", observed, baselines=baselines)
    assert phrase, "phrase must be non-empty when baseline exists"


# ── 2. annotate_role invariants ──────────────────────────────────


@given(
    role_prefix=st.sampled_from([
        "Security lead", "SECURITY LEAD", "security lead",
        "Charge Nurse", "charge nurse", "Charge nurse on duty",
        "Biomedical Engineering", "BIOMEDICAL ENGINEERING",
        "ICU charge nurse",
    ]),
    suffix=st.text(alphabet=" abcdefABC.,", max_size=20),
)
def test_annotate_role_case_insensitive_substring_match(role_prefix, suffix):
    """The case-sensitivity bug we hit on "Charge nurse" vs
    "Charge Nurse": the annotated output must contain an extension
    (``ext NNNN`` or compact ``[NNNN/...]``) for any case variant of
    a known role, regardless of trailing text."""
    from module5_responses.config import annotate_role

    out = annotate_role(f"{role_prefix} {suffix}".strip())
    assert "ext" in out.lower() or re.search(r"\[\d{3,5}/", out), (
        f"annotate_role did not annotate {role_prefix!r}: got {out!r}"
    )


@given(unknown=st.text(min_size=1, max_size=40).filter(
    lambda s: not any(
        x in s.lower() for x in
        ("security", "soc", "ciso", "nurse", "biomed", "clinical",
         "network admin", "privacy", "incident", "hr", "physician")
    )
))
def test_annotate_role_unknown_passes_through_unchanged(unknown):
    """For phrases that don't match any known role,
    ``annotate_role`` must return the input verbatim — never
    fabricate an extension."""
    from module5_responses.config import annotate_role
    out = annotate_role(unknown)
    assert out == unknown


# ── 3. assign_risk_levels invariants ─────────────────────────────


@given(
    R=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    c_detect=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
)
def test_assign_risk_levels_emits_only_valid_tiers(R, c_detect):
    """No matter what R / c_detect inputs come in (within [0, 1]),
    the output tier must be one of the canonical Literal values."""
    from module3_risk_scoring.composition import assign_risk_levels
    from common.alert_response_schema import AlertRecord
    import typing as _t

    field = AlertRecord.model_fields["risk_level"]
    valid_tiers = set(_t.get_args(field.annotation))

    out = assign_risk_levels(np.array([R]), c_detect=np.array([c_detect]))
    assert out[0] in valid_tiers, (
        f"R={R}, c_detect={c_detect} produced unknown tier {out[0]!r}"
    )


@given(R=st.floats(min_value=0.0, max_value=1.0, allow_nan=False))
def test_low_detection_always_yields_normal(R):
    """The Phase B detection gate property: any sample with
    c_detect < MIN_DETECTION_GATE must be NORMAL regardless of R."""
    from module3_risk_scoring.composition import assign_risk_levels
    from module3_risk_scoring.config import MIN_DETECTION_GATE
    # Pick c_detect strictly below the gate
    out = assign_risk_levels(
        np.array([R]),
        c_detect=np.array([MIN_DETECTION_GATE - 1e-6]),
    )
    assert out[0] == "NORMAL"


# ── 4. stability_band invariants ─────────────────────────────────


@given(score=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False))
def test_stability_band_always_in_valid_set(score):
    """For any score (even out-of-range), the band must be one of
    the canonical three values — never silently break the contract."""
    from module4_explanations.stability import stability_band
    assert stability_band(score) in {"STABLE", "BORDERLINE", "UNSTABLE"}


@given(score=st.floats(min_value=0.0, max_value=1.0, allow_nan=False))
def test_stability_band_monotonic(score):
    """If score A > score B, then band(A) is at least as STABLE as
    band(B). Specifically: STABLE > BORDERLINE > UNSTABLE.

    A monotonic-violation here would mean the threshold logic has a
    bug (off-by-one, wrong comparison operator, etc.)."""
    from module4_explanations.stability import stability_band
    band_rank = {"UNSTABLE": 0, "BORDERLINE": 1, "STABLE": 2}
    eps = 1e-6
    higher = stability_band(min(1.0, score + eps))
    lower  = stability_band(max(0.0, score - eps))
    assert band_rank[higher] >= band_rank[lower]


# ── 5. counterfactual sparsity bound ─────────────────────────────


# Light, deterministic stand-in classifier for hypothesis.
class _LinearStub:
    def __init__(self, w):
        self.w = np.asarray(w, dtype=float)

    def predict_proba(self, X):
        X = np.atleast_2d(X)
        z = X @ self.w
        p1 = 1.0 / (1.0 + np.exp(-z))
        return np.column_stack([1.0 - p1, p1])


_FEAT_NAMES_5 = ["Sport", "DIntPkt", "SrcBytes", "Temp", "SpO2"]
_BASELINES_5 = {
    f: {"median": 0.0, "iqr_low": -0.5, "iqr_high": 0.5,
        "p05": -1.0, "p95": 1.0, "unit": "", "decimal_places": 2,
        "is_biometric": f in ("Temp", "SpO2"), "n_benign": 100}
    for f in _FEAT_NAMES_5
}


@given(
    w0=st.floats(min_value=1.0, max_value=5.0, allow_nan=False),
    x0=st.floats(min_value=0.8, max_value=1.0, allow_nan=False),
    max_sparsity=st.integers(min_value=0, max_value=3),
)
def test_counterfactual_sparsity_respects_max(w0, x0, max_sparsity):
    """For any feasible counterfactual, ``sparsity`` must not
    exceed ``max_sparsity``. This is the contract; it'd be a silent
    bug if the search ever returned more changes than asked."""
    from module4_explanations.counterfactual import compute_counterfactual

    clf = _LinearStub([w0, 0.0, 0.0, 0.0, 0.0])
    x = np.array([x0, 0.0, 0.0, 0.0, 0.0])
    sv = clf.w * x

    r = compute_counterfactual(
        clf, x, sv, _FEAT_NAMES_5, threshold=0.5,
        max_sparsity=max_sparsity, baselines=_BASELINES_5,
    )
    if r.feasible:
        assert r.sparsity <= max_sparsity
        assert len(r.changes) == r.sparsity


@given(
    w=st.lists(
        st.floats(min_value=-3.0, max_value=3.0, allow_nan=False),
        min_size=5, max_size=5,
    ),
    x=st.lists(
        st.floats(min_value=-1.0, max_value=1.0, allow_nan=False),
        min_size=5, max_size=5,
    ),
)
def test_counterfactual_never_proposes_biometric_change(w, x):
    """The clinical-safety invariant: counterfactual changes must
    never include a biometric feature, even when SHAP says biometric
    is what drives the prediction."""
    from module4_explanations.counterfactual import compute_counterfactual

    clf = _LinearStub(w)
    sv = np.asarray(w) * np.asarray(x)
    r = compute_counterfactual(
        clf, np.asarray(x), sv, _FEAT_NAMES_5,
        threshold=0.5, baselines=_BASELINES_5,
    )
    if r.feasible:
        for change in r.changes:
            assert change["feature"] not in ("Temp", "SpO2"), (
                f"Counterfactual proposed biometric change: {change}"
            )
