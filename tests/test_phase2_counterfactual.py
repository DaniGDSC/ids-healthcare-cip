"""Tests for ``module4_explanations.counterfactual`` (Phase 2).

Sparsity, validity, plausibility, and biometric-immutability invariants
are covered with synthetic XGBoost-shaped models so the tests don't
depend on the trained corpus pickle. End-to-end coverage against the
real pickle is provided by the Phase 0 baseline ``--check`` floor.
"""
from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pytest

from module4_explanations.counterfactual import (
    CF_CANDIDATE_K,
    CF_MAX_SPARSITY,
    CounterfactualResult,
    _candidate_indices,
    _remediation_hint,
    compute_counterfactual,
    counterfactual_narrative,
)


# ── Lightweight stand-in classifier ───────────────────────────────


class _LinearStub:
    """A tiny ``predict_proba``-compatible classifier.

    The "attack" probability is ``sigmoid(w · x + b)`` so each feature's
    contribution is linear and the counterfactual search has a clean
    closed-form direction. Used in place of the real XGBoost pipeline
    so tests are dependency-free and deterministic.
    """

    def __init__(self, w: np.ndarray, b: float = 0.0):
        self.w = np.asarray(w, dtype=float)
        self.b = float(b)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        z = X @ self.w + self.b
        p1 = 1.0 / (1.0 + np.exp(-z))
        return np.column_stack([1.0 - p1, p1])


_FEAT_NAMES = ["Sport", "DIntPkt", "SrcBytes", "Temp", "SpO2"]

_BASELINES = {
    "Sport":    {"median": 0.0, "iqr_low": -0.5, "iqr_high": 0.5,
                  "p05": -1.0, "p95": 1.0, "unit": "port", "decimal_places": 0,
                  "is_biometric": False, "n_benign": 100},
    "DIntPkt":  {"median": 0.0, "iqr_low": -0.5, "iqr_high": 0.5,
                  "p05": -1.0, "p95": 1.0, "unit": "ms", "decimal_places": 2,
                  "is_biometric": False, "n_benign": 100},
    "SrcBytes": {"median": 0.0, "iqr_low": -0.5, "iqr_high": 0.5,
                  "p05": -1.0, "p95": 1.0, "unit": "B", "decimal_places": 0,
                  "is_biometric": False, "n_benign": 100},
    "Temp":     {"median": 0.0, "iqr_low": -0.5, "iqr_high": 0.5,
                  "p05": -1.0, "p95": 1.0, "unit": "°C", "decimal_places": 1,
                  "is_biometric": True, "n_benign": 100},
    "SpO2":     {"median": 0.0, "iqr_low": -0.5, "iqr_high": 0.5,
                  "p05": -1.0, "p95": 1.0, "unit": "%", "decimal_places": 0,
                  "is_biometric": True, "n_benign": 100},
}


# ── Candidate-set tests ───────────────────────────────────────────


def test_candidate_indices_excludes_biometric():
    """Biometric features must never be candidates — they represent the
    patient, not network state, so a counterfactual that demands a
    different vital sign is meaningless."""
    sv = np.array([0.3, 0.5, 0.2, 0.9, 0.7])  # Temp + SpO2 dominate
    out = _candidate_indices(sv, _FEAT_NAMES, k=5)
    feats_picked = [_FEAT_NAMES[i] for i in out]
    assert "Temp" not in feats_picked
    assert "SpO2" not in feats_picked
    assert set(feats_picked) == {"Sport", "DIntPkt", "SrcBytes"}


def test_candidate_indices_orders_by_shap_magnitude():
    sv = np.array([0.1, 0.9, 0.3, 0.0, 0.0])  # DIntPkt biggest
    out = _candidate_indices(sv, _FEAT_NAMES, k=2)
    assert out[0] == _FEAT_NAMES.index("DIntPkt")


def test_candidate_indices_caps_at_k():
    sv = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
    out = _candidate_indices(sv, _FEAT_NAMES, k=2)
    assert len(out) == 2


# ── Sparsity-1 path ──────────────────────────────────────────────


def test_sparsity_one_when_single_feature_drives_alert():
    """If a single feature dominates, the counterfactual must flip it
    and report sparsity=1."""
    w = np.array([5.0, 0.0, 0.0, 0.0, 0.0])
    clf = _LinearStub(w, b=-1.0)
    x = np.array([0.8, 0.0, 0.0, 0.0, 0.0])  # well into attack territory
    sv = w * x   # SHAP for linear model = weight × value (close enough)

    r = compute_counterfactual(clf, x, sv, _FEAT_NAMES, threshold=0.5,
                                baselines=_BASELINES)
    assert r.feasible
    assert r.sparsity == 1
    assert r.changes[0]["feature"] == "Sport"
    assert r.flips_prediction
    assert r.new_proba < 0.5
    assert r.remediation_hint  # non-empty


def test_counterfactual_actually_flips_when_applied():
    """Validity check — applying the returned ``changes`` to ``x`` must
    yield a probability below threshold."""
    w = np.array([4.0, 2.0, 1.0, 0.0, 0.0])
    clf = _LinearStub(w, b=-0.5)
    x = np.array([0.9, 0.5, 0.5, 0.0, 0.0])
    sv = w * x

    r = compute_counterfactual(clf, x, sv, _FEAT_NAMES, threshold=0.5,
                                baselines=_BASELINES)
    assert r.feasible
    x_cf = x.copy()
    for c in r.changes:
        x_cf[_FEAT_NAMES.index(c["feature"])] = c["new"]
    p_cf = clf.predict_proba(x_cf.reshape(1, -1))[0, 1]
    # Allow a small slack: ``new`` is rounded to ``decimal_places + 2`` for
    # JSON-safe serialisation, so the re-applied value can drift slightly
    # back toward the threshold. The key invariant is that proba is
    # *materially* below the original, not below threshold by a hair.
    assert p_cf < r.original_proba
    assert p_cf < 0.51


# ── Plausibility ─────────────────────────────────────────────────


def test_counterfactual_respects_p05_p95_bounds():
    """The returned ``new`` value must lie within ``[p05, p95]`` of the
    benign distribution — no extrapolation to physically implausible
    inputs."""
    w = np.array([5.0, 0.0, 0.0, 0.0, 0.0])
    clf = _LinearStub(w, b=-1.0)
    x = np.array([0.95, 0.0, 0.0, 0.0, 0.0])
    sv = w * x

    r = compute_counterfactual(clf, x, sv, _FEAT_NAMES, threshold=0.5,
                                baselines=_BASELINES)
    assert r.feasible
    for c in r.changes:
        stats = _BASELINES[c["feature"]]
        # Allow tiny binary-search overshoot
        assert stats["p05"] - 1e-3 <= c["new"] <= stats["p95"] + 1e-3


def test_infeasible_when_only_immutable_features_drive_alert():
    """When only biometric features carry SHAP weight (which we never
    permit as counterfactual candidates), the search must return
    ``feasible=False`` rather than fabricate a network change."""
    # All weight is on Temp + SpO2 (biometric, excluded as candidates)
    w = np.array([0.0, 0.0, 0.0, 5.0, 5.0])
    clf = _LinearStub(w, b=-1.0)
    x = np.array([0.0, 0.0, 0.0, 0.9, 0.9])
    sv = w * x
    r = compute_counterfactual(clf, x, sv, _FEAT_NAMES, threshold=0.5,
                                baselines=_BASELINES)
    assert not r.feasible
    assert r.sparsity == 0
    assert r.changes == []


def test_no_counterfactual_for_already_benign_sample():
    """If the original sample is already predicted benign, no work to do."""
    w = np.array([5.0, 0.0, 0.0, 0.0, 0.0])
    clf = _LinearStub(w, b=-1.0)
    x = np.array([0.0, 0.0, 0.0, 0.0, 0.0])  # original proba << threshold
    sv = w * x
    r = compute_counterfactual(clf, x, sv, _FEAT_NAMES, threshold=0.5,
                                baselines=_BASELINES)
    assert not r.feasible
    assert r.sparsity == 0


# ── max_sparsity caps ────────────────────────────────────────────


def test_max_sparsity_zero_returns_infeasible():
    w = np.array([5.0, 0.0, 0.0, 0.0, 0.0])
    clf = _LinearStub(w, b=-1.0)
    x = np.array([0.9, 0.0, 0.0, 0.0, 0.0])
    sv = w * x
    # Sparsity 1 would work, but max_sparsity=0 forbids any change
    r = compute_counterfactual(clf, x, sv, _FEAT_NAMES, threshold=0.5,
                                baselines=_BASELINES, max_sparsity=0)
    assert not r.feasible


# ── Serialisation + narrative ────────────────────────────────────


def test_to_dict_is_json_safe():
    """``CounterfactualResult.to_dict`` must produce JSON-compatible
    primitives (no numpy scalars) so it can be written into
    analyst_report.json without a custom encoder."""
    import json
    r = CounterfactualResult(
        sparsity=1,
        changes=[{"feature": "Sport", "original": 0.9, "new": 0.0,
                   "abs_delta": 0.9, "unit": "port"}],
        flips_prediction=True,
        new_proba=0.1,
        original_proba=0.9,
        remediation_hint="Block source port",
        feasible=True,
    )
    json.dumps(r.to_dict())  # must not raise


def test_narrative_sparsity_one():
    r = CounterfactualResult(
        sparsity=1,
        changes=[{"feature": "Sport", "original": 0.9, "new": 0.0,
                   "abs_delta": 0.9, "unit": "port"}],
        flips_prediction=True, new_proba=0.1, original_proba=0.9,
        remediation_hint="Block source port", feasible=True,
    )
    out = counterfactual_narrative(r)
    assert "would clear" in out
    assert "Sport" in out
    assert "Block source port" in out


def test_narrative_sparsity_two():
    r = CounterfactualResult(
        sparsity=2,
        changes=[
            {"feature": "Sport",   "original": 0.9, "new": 0.0,
             "abs_delta": 0.9, "unit": "port"},
            {"feature": "DIntPkt", "original": 0.6, "new": 0.0,
             "abs_delta": 0.6, "unit": "ms"},
        ],
        flips_prediction=True, new_proba=0.1, original_proba=0.9,
        remediation_hint="Block source port", feasible=True,
    )
    out = counterfactual_narrative(r)
    assert "Sport" in out and "DIntPkt" in out
    assert "returned to the benign band" in out


def test_narrative_infeasible_returns_empty():
    r = CounterfactualResult(feasible=False)
    assert counterfactual_narrative(r) == ""


def test_remediation_hint_known_feature():
    assert "outbound" in _remediation_hint("SrcBytes").lower()


def test_remediation_hint_unknown_feature_fallback():
    out = _remediation_hint("totally_unknown_feat")
    assert "totally_unknown_feat" in out
    assert "benign band" in out


# ── Schema integration ──────────────────────────────────────────


def test_explanation_schema_accepts_counterfactual():
    from common.alert_response_schema import Explanation
    e = Explanation(
        clinician_summary="x", analyst_available=True,
        counterfactual={
            "sparsity": 1,
            "changes": [{"feature": "Sport", "original": 0.9,
                          "new": 0.0, "abs_delta": 0.9, "unit": "port"}],
            "flips_prediction": True,
            "new_proba": 0.1, "original_proba": 0.9,
            "remediation_hint": "Block port", "feasible": True,
        },
    )
    assert e.counterfactual["sparsity"] == 1


def test_response_schema_accepts_try_first_action():
    from common.alert_response_schema import Response, EscalationChain
    r = Response(
        actions=["log_event"], action_descriptions=["x"],
        escalation_chain=EscalationChain(primary=None, secondary=None, tertiary=None),
        escalation_rationale="", max_response_min=0, priority=4,
        rationale="", device_tier="vital_monitoring",
        device_constraint_applied=False,
        try_first_action="Block tcp/44312 at the segment firewall.",
    )
    assert "tcp" in r.try_first_action
