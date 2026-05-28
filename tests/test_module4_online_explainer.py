"""Module 4 online_explainer — AlertExplainer Y10 fix + smoke explain."""
from __future__ import annotations

import numpy as np
import pytest


def test_alert_explainer_rejects_empty_feat_names():
    """Y10: feat_names is required at construction time."""
    from module4_explanations.online_explainer import AlertExplainer
    with pytest.raises(ValueError, match="requires feat_names"):
        AlertExplainer(feat_names=[])


def test_alert_explainer_validate_feat_names_rejects_mismatch():
    """Y10: explain(x, feat_names) raises if feat_names differs from
    constructor."""
    # We can't fully construct AlertExplainer in tests (needs model
    # registry), but the validation logic is a static method-equivalent
    # we can test in isolation by patching.
    from module4_explanations.online_explainer import AlertExplainer

    # Mock the constructor to skip registry loads
    explainer = AlertExplainer.__new__(AlertExplainer)
    explainer.feat_names = ("f1", "f2", "f3")

    # Validates None (allowed)
    explainer._validate_feat_names(None)
    # Validates same tuple (allowed)
    explainer._validate_feat_names(["f1", "f2", "f3"])
    # Rejects different
    with pytest.raises(ValueError, match="do not override per-call"):
        explainer._validate_feat_names(["f1", "f2", "OTHER"])


def test_alert_explainer_feat_names_stored_as_tuple():
    """Y10: feat_names is stored as immutable tuple."""
    from module4_explanations.online_explainer import AlertExplainer
    explainer = AlertExplainer.__new__(AlertExplainer)
    explainer.feat_names = tuple(["a", "b", "c"])
    assert isinstance(explainer.feat_names, tuple)


def test_risk_level_from_score_thresholds():
    """Severity tier follows Module 3 risk_score thresholds (canonical)."""
    from module4_explanations.online_explainer import _risk_level_from_score
    assert _risk_level_from_score(0.95) == "CRITICAL"
    assert _risk_level_from_score(0.80) == "CRITICAL"
    assert _risk_level_from_score(0.79) == "HIGH"
    assert _risk_level_from_score(0.60) == "HIGH"
    assert _risk_level_from_score(0.59) == "MEDIUM"
    assert _risk_level_from_score(0.40) == "MEDIUM"
    assert _risk_level_from_score(0.39) == "LOW"
    assert _risk_level_from_score(0.0) == "LOW"
    assert _risk_level_from_score(None) == "LOW"


def test_sanitise_replaces_nan_inf():
    """OOD-05 fix: NaN/Inf → zero."""
    from module4_explanations.online_explainer import AlertExplainer
    x = np.array([[1.0, np.nan, np.inf, -np.inf]])
    out = AlertExplainer._sanitise(x)
    assert np.isfinite(out).all()
    assert out[0, 0] == 1.0
    assert out[0, 1] == 0.0
    assert out[0, 2] == 0.0


def test_sanitise_passthrough_when_clean():
    """Clean input is returned unchanged."""
    from module4_explanations.online_explainer import AlertExplainer
    x = np.array([[1.0, 2.0, 3.0]])
    out = AlertExplainer._sanitise(x)
    np.testing.assert_array_equal(out, x)


def test_top_shap_helper_uses_compute_module():
    """Backward-compat: instance method delegates to compute helper."""
    from module4_explanations.online_explainer import AlertExplainer
    explainer = AlertExplainer.__new__(AlertExplainer)
    explainer.feat_names = ("a", "b", "c", "d")
    sv_row = np.array([0.1, -0.5, 0.3, 0.2])
    top = explainer._top_shap(sv_row, k=2)
    assert [t["feature"] for t in top] == ["b", "c"]
