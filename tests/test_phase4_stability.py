"""Tests for Phase 4 — stability score, badge mapping, schema, and
the faithfulness CI gate's check functions.

Stability compute is tested against a deterministic stand-in
``shap.TreeExplainer`` so we don't depend on the trained pickle.
"""
from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pytest

from module4_explanations.stability import (
    StabilityResult,
    compute_stability,
    stability_badge,
    stability_band,
    THRESHOLD_BORDERLINE,
    THRESHOLD_STABLE,
)


# ── Banding ────────────────────────────────────────────────────────


def test_stability_band_stable_threshold_inclusive():
    assert stability_band(THRESHOLD_STABLE) == "STABLE"
    assert stability_band(1.0) == "STABLE"


def test_stability_band_borderline_range():
    assert stability_band(THRESHOLD_BORDERLINE) == "BORDERLINE"
    assert stability_band(0.85) == "BORDERLINE"
    assert stability_band(0.7001) == "BORDERLINE"


def test_stability_band_unstable_below_borderline():
    assert stability_band(0.69) == "UNSTABLE"
    assert stability_band(0.0)  == "UNSTABLE"
    assert stability_band(-0.5) == "UNSTABLE"


def test_stability_badge_returns_string_per_band():
    for band in ("STABLE", "BORDERLINE", "UNSTABLE"):
        b = stability_badge(band)
        assert b
        assert band in b


def test_stability_badge_unknown_band_returns_empty():
    assert stability_badge("MADE_UP") == ""


# ── Result ─────────────────────────────────────────────────────────


def test_stability_result_to_dict_is_json_safe():
    import json
    r = StabilityResult(
        score=0.85, band="BORDERLINE",
        n_perturbations=20, sigma=0.01, top_k=5,
        baseline_top_features=["DIntPkt", "Flgs", "Sport"],
        min_overlap=0.6,
    )
    s = json.dumps(r.to_dict())
    assert "BORDERLINE" in s


# ── compute_stability with a stand-in explainer ───────────────────


class _DeterministicExplainer:
    """Returns the same SHAP vector for any input — produces a perfectly
    stable explanation."""

    def __init__(self, n_features: int):
        self.sv = np.zeros(n_features, dtype=float)
        self.sv[0] = 0.9  # always rank feature 0 first
        self.sv[1] = 0.5

    def shap_values(self, X: np.ndarray) -> np.ndarray:
        return np.tile(self.sv, (len(X), 1))


class _ChaoticExplainer:
    """Returns a different random SHAP vector each call — guarantees
    maximum instability."""

    def __init__(self, n_features: int, seed: int = 0):
        self.n = n_features
        self.rng = np.random.default_rng(seed)

    def shap_values(self, X: np.ndarray) -> np.ndarray:
        return self.rng.normal(0.0, 1.0, size=(len(X), self.n))


def test_compute_stability_perfectly_stable_explainer():
    explainer = _DeterministicExplainer(n_features=10)
    feat_names = [f"f{i}" for i in range(10)]
    x = np.zeros(10)
    rng = np.random.default_rng(0)
    r = compute_stability(explainer, x, feat_names,
                          n_perturbations=10, rng=rng)
    assert r.score == 1.0
    assert r.band == "STABLE"
    assert r.min_overlap == 1.0


def test_compute_stability_chaotic_explainer_is_unstable():
    explainer = _ChaoticExplainer(n_features=20, seed=1)
    feat_names = [f"f{i}" for i in range(20)]
    x = np.zeros(20)
    rng = np.random.default_rng(2)
    r = compute_stability(explainer, x, feat_names,
                          n_perturbations=30, top_k=5, rng=rng)
    # With 20 features and 5-element top-K drawn randomly, expected
    # Jaccard ≈ 5/(5+5-2.5) ≈ 0.18 << UNSTABLE threshold.
    assert r.score < 0.5
    assert r.band == "UNSTABLE"


def test_compute_stability_baseline_shap_short_circuits_extra_call():
    """When ``baseline_shap_row`` is provided, the baseline call to
    ``explainer.shap_values`` must be skipped (one fewer call total)."""
    explainer = _DeterministicExplainer(n_features=10)
    counter = {"calls": 0}
    orig_shap = explainer.shap_values

    def _tracked(X):
        counter["calls"] += 1
        return orig_shap(X)
    explainer.shap_values = _tracked

    x = np.zeros(10)
    baseline = orig_shap(np.zeros((1, 10)))[0]
    rng = np.random.default_rng(0)
    compute_stability(explainer, x, [f"f{i}" for i in range(10)],
                       n_perturbations=5, rng=rng,
                       baseline_shap_row=baseline)
    # Only the 5 perturbation calls happen — no baseline call.
    assert counter["calls"] == 5


def test_compute_stability_is_deterministic_given_seed():
    explainer = _ChaoticExplainer(n_features=10, seed=42)
    feat_names = [f"f{i}" for i in range(10)]
    x = np.zeros(10)
    r1 = compute_stability(explainer, x, feat_names,
                            n_perturbations=10, rng=np.random.default_rng(7))
    explainer2 = _ChaoticExplainer(n_features=10, seed=42)
    r2 = compute_stability(explainer2, x, feat_names,
                            n_perturbations=10, rng=np.random.default_rng(7))
    assert r1.score == r2.score
    assert r1.band == r2.band


# ── Explanation schema accepts stability ──────────────────────────


def test_explanation_schema_accepts_stability():
    from common.alert_response_schema import Explanation
    e = Explanation(
        clinician_summary="x",
        analyst_available=True,
        stability={
            "score": 0.85, "band": "BORDERLINE",
            "n_perturbations": 20, "sigma": 0.01, "top_k": 5,
            "baseline_top_features": ["A", "B"],
            "min_overlap": 0.6,
        },
    )
    assert e.stability["band"] == "BORDERLINE"


def test_explanation_schema_legacy_without_stability_still_validates():
    from common.alert_response_schema import Explanation
    e = Explanation(clinician_summary="x", analyst_available=True)
    assert e.stability is None
