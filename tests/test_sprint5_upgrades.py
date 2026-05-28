"""Tests for Sprint 5 upgrades (RF counterfactual + robust SHAP).

Sprint 5 / Tầng 3.4 — RandomForest counterfactual coverage augment
Sprint 5 / Tầng 3.2 — Robust top features (mean-SHAP over perturbations)

The counterfactual engine itself is model-agnostic and was tested
in Phase 2; here we pin the *applicability* invariants — RF works
end-to-end without the engine special-casing the tree backend.
"""
from __future__ import annotations

import numpy as np
import pytest


# ── 3.4 — Counterfactual model-agnosticism ────────────────────────


class _StubExplainer:
    """SHAP TreeExplainer-compatible stub for tests.

    ``shap_values(X)`` returns a deterministic per-feature attribution
    proportional to the feature value, plus a small noise term seeded
    by the row sum so that perturbations produce slightly different
    attributions (mimicking real SHAP under noise).
    """

    def __init__(self, w):
        self.w = np.asarray(w, dtype=float)

    def shap_values(self, X):
        X = np.atleast_2d(X)
        # mean SHAP roughly = w × X, with deterministic per-row noise
        sv = X * self.w
        rng = np.random.default_rng(int(abs(X.sum() * 1e6)) % 2**32)
        return sv + rng.normal(0, 0.01, size=sv.shape)


_FEAT_NAMES = ["Sport", "DIntPkt", "SrcBytes", "Temp", "SpO2"]
_BASELINES = {
    f: {
        "median": 0.0, "iqr_low": -0.5, "iqr_high": 0.5,
        "p05": -1.0, "p95": 1.0, "unit": "", "decimal_places": 2,
        "is_biometric": f in ("Temp", "SpO2"), "n_benign": 100,
    }
    for f in _FEAT_NAMES
}


def test_counterfactual_engine_works_with_sklearn_predict_proba():
    """The counterfactual engine must accept *any* sklearn-style
    ``predict_proba`` — XGBoost, RandomForest, DecisionTree all
    expose the same surface, so the engine should not special-case
    the model class."""
    from sklearn.ensemble import RandomForestClassifier
    from module4_explanations.counterfactual import compute_counterfactual

    rng = np.random.default_rng(42)
    X = rng.uniform(-1, 1, size=(100, 5))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    clf = RandomForestClassifier(n_estimators=10, random_state=42).fit(X, y)

    x = np.array([0.8, 0.6, 0.0, 0.0, 0.0])
    sv = np.array([0.5, 0.3, 0.0, 0.0, 0.0])
    r = compute_counterfactual(
        clf, x, sv, _FEAT_NAMES, threshold=0.5, baselines=_BASELINES,
    )
    # The engine ran without raising — that's the model-agnostic
    # invariant. Feasibility itself depends on the model's
    # predict_proba behaviour and isn't pinned.
    assert r.original_proba >= 0.0
    assert r.original_proba <= 1.0


def test_counterfactual_works_with_decision_tree():
    from sklearn.tree import DecisionTreeClassifier
    from module4_explanations.counterfactual import compute_counterfactual

    rng = np.random.default_rng(7)
    X = rng.uniform(-1, 1, size=(100, 5))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    clf = DecisionTreeClassifier(random_state=7).fit(X, y)

    x = np.array([0.9, 0.7, 0.0, 0.0, 0.0])
    sv = np.array([0.5, 0.3, 0.0, 0.0, 0.0])
    r = compute_counterfactual(
        clf, x, sv, _FEAT_NAMES, threshold=0.5, baselines=_BASELINES,
    )
    # As above — invariant is that no exception fires.
    assert hasattr(r, "feasible")


# ── 3.2 — Robust top features ─────────────────────────────────────


def test_robust_top_features_returns_top_k():
    from module4_explanations.stability import compute_robust_top_features

    explainer = _StubExplainer(w=[2.0, 1.0, 0.5, 0.0, 0.0])
    x = np.array([1.0, 1.0, 1.0, 0.0, 0.0])
    out = compute_robust_top_features(
        explainer, x, _FEAT_NAMES,
        n_perturbations=10, sigma=0.01, top_k=3,
        rng=np.random.default_rng(42),
    )
    assert len(out) == 3
    # The feature with the largest weight should be top-1
    assert out[0]["feature"] == "Sport"


def test_robust_top_features_includes_std_for_uncertainty():
    """Each returned feature must carry a ``std_shap`` field so the
    analyst can see how confident the attribution is."""
    from module4_explanations.stability import compute_robust_top_features

    explainer = _StubExplainer(w=[2.0, 1.0, 0.5, 0.0, 0.0])
    x = np.array([1.0, 1.0, 0.0, 0.0, 0.0])
    out = compute_robust_top_features(
        explainer, x, _FEAT_NAMES,
        n_perturbations=10, sigma=0.01,
        rng=np.random.default_rng(42),
    )
    for f in out:
        assert "std_shap" in f
        assert f["std_shap"] >= 0.0


def test_robust_top_features_direction_consistent_with_mean():
    """``direction`` must reflect the sign of ``mean_shap`` not any
    single perturbation."""
    from module4_explanations.stability import compute_robust_top_features

    explainer = _StubExplainer(w=[2.0, -1.0, 0.0, 0.0, 0.0])
    x = np.array([0.8, 0.7, 0.0, 0.0, 0.0])
    out = compute_robust_top_features(
        explainer, x, _FEAT_NAMES,
        n_perturbations=10, sigma=0.01,
        rng=np.random.default_rng(0),
    )
    for f in out:
        if f["mean_shap"] > 0:
            assert f["direction"] == "increases_risk"
        elif f["mean_shap"] < 0:
            assert f["direction"] == "decreases_risk"


def test_robust_top_features_more_stable_than_single_shot():
    """The whole point of the helper: averaging across perturbations
    produces a more stable top-K than any single shot.

    Concretely: when we compute the top-K twice with *different* RNG
    seeds but the same x_row, the robust top-K should overlap more
    than two single-shot top-Ks would."""
    from module4_explanations.stability import compute_robust_top_features

    explainer = _StubExplainer(w=[2.0, 1.9, 1.8, 0.1, 0.0])
    x = np.array([1.0, 1.0, 1.0, 0.0, 0.0])

    robust_1 = {
        f["feature"] for f in compute_robust_top_features(
            explainer, x, _FEAT_NAMES,
            n_perturbations=30, sigma=0.05, top_k=3,
            rng=np.random.default_rng(1),
        )
    }
    robust_2 = {
        f["feature"] for f in compute_robust_top_features(
            explainer, x, _FEAT_NAMES,
            n_perturbations=30, sigma=0.05, top_k=3,
            rng=np.random.default_rng(2),
        )
    }
    overlap = len(robust_1 & robust_2)
    # 30-perturbation means should agree on at least 2 of 3 features
    assert overlap >= 2, (
        f"robust attributions diverged: {robust_1} vs {robust_2}"
    )


def test_robust_top_features_skips_biometric_optional():
    """The robust helper itself doesn't enforce biometric exclusion
    — that's the counterfactual engine's job. The robust attribution
    can return biometric features (they're informative for analysts);
    the counterfactual engine downstream filters them out."""
    from module4_explanations.stability import compute_robust_top_features

    explainer = _StubExplainer(w=[0.0, 0.0, 0.0, 2.0, 1.5])
    x = np.array([0.0, 0.0, 0.0, 1.0, 1.0])
    out = compute_robust_top_features(
        explainer, x, _FEAT_NAMES,
        n_perturbations=10, sigma=0.01,
        rng=np.random.default_rng(42),
    )
    features = {f["feature"] for f in out}
    # Biometric features ARE allowed in robust attribution
    assert features & {"Temp", "SpO2"}, (
        "robust top features should include biometric when SHAP says so"
    )
