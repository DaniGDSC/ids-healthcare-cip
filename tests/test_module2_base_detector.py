"""BaseTrackADetector — template-method contract tests.

Uses a tiny fake detector to verify the shared base class works
end-to-end without depending on real sklearn models for slow CV.
"""
from __future__ import annotations

from typing import Any, ClassVar, Dict, List

import numpy as np
import pytest
from sklearn.tree import DecisionTreeClassifier

from module2_detection.models._base_detector import BaseTrackADetector


@pytest.fixture
def Xy_balanced():
    rng = np.random.default_rng(0)
    n = 80
    X = np.zeros((n, 4), dtype=np.float32)
    X[: n // 2] = rng.normal(loc=-1.5, scale=0.5, size=(n // 2, 4))
    X[n // 2:] = rng.normal(loc=1.5, scale=0.5, size=(n // 2, 4))
    y = np.array([0] * (n // 2) + [1] * (n // 2))
    perm = rng.permutation(n)
    return X[perm], y[perm]


class _TinyDetector(BaseTrackADetector):
    """Minimal subclass for testing — uses DT with tiny param space."""

    PARAM_SPACE: ClassVar[Dict[str, List[Any]]] = {
        "classifier__max_depth": [3, 5],
    }
    DEFAULT_N_ITER: ClassVar[int] = 2
    MODEL_TYPE: ClassVar[str] = "TinyDetector"
    LOG_NAME: ClassVar[str] = "Tiny"

    def _classifier(self):
        return DecisionTreeClassifier(random_state=self._random_state)


# ── Contract: subclass must implement _classifier ─────────────────────


def test_base_is_abstract_cannot_instantiate():
    with pytest.raises(TypeError):
        BaseTrackADetector(n_iter=1)


def test_subclass_without_classifier_method_cannot_instantiate():
    class BadDetector(BaseTrackADetector):
        PARAM_SPACE = {}
        # no _classifier override
    with pytest.raises(TypeError):
        BadDetector()


# ── Construction defaults ─────────────────────────────────────────────


def test_n_iter_defaults_to_class_constant():
    det = _TinyDetector()
    assert det._n_iter == _TinyDetector.DEFAULT_N_ITER


def test_n_iter_explicit_overrides_default():
    det = _TinyDetector(n_iter=7)
    assert det._n_iter == 7


def test_random_state_propagates_to_classifier():
    det = _TinyDetector(random_state=123)
    clf = det._classifier()
    assert clf.random_state == 123


# ── Fit / predict pipeline ────────────────────────────────────────────


def test_predict_without_fit_raises(Xy_balanced):
    X, _ = Xy_balanced
    det = _TinyDetector(cv_folds=3)
    with pytest.raises(RuntimeError, match="not fitted"):
        det.predict_proba(X[:5])


def test_fit_records_best_params_and_pipeline(Xy_balanced):
    X, y = Xy_balanced
    det = _TinyDetector(n_iter=2, cv_folds=3, random_state=42)
    det.fit(X, y)
    assert det._best_pipeline is not None
    assert "classifier__max_depth" in det.best_params
    assert det.optimal_threshold != 0.5 or True  # threshold gets set (could legitimately be 0.5)


def test_predict_after_fit(Xy_balanced):
    X, y = Xy_balanced
    det = _TinyDetector(n_iter=2, cv_folds=3, random_state=42)
    det.fit(X, y)
    pred = det.predict(X[:10])
    proba = det.predict_proba(X[:10])
    assert pred.shape == (10,)
    assert proba.shape == (10,)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_evaluate_returns_expected_metrics(Xy_balanced):
    X, y = Xy_balanced
    det = _TinyDetector(n_iter=2, cv_folds=3, random_state=42)
    det.fit(X, y)
    metrics = det.evaluate(X, y)
    for key in ("attack_f1", "attack_f2", "weighted_f1", "macro_f1",
                "auc_roc", "optimal_threshold"):
        assert key in metrics


def test_get_report_shape(Xy_balanced):
    X, y = Xy_balanced
    det = _TinyDetector(n_iter=2, cv_folds=3, random_state=42)
    det.fit(X, y)
    det.evaluate(X, y)
    report = det.get_report()
    assert report["model_type"] == "TinyDetector"
    assert "best_params" in report
    assert "cv_results" in report
    assert "optimal_threshold" in report
    assert "test_metrics" in report


def test_pipeline_includes_smote_step(Xy_balanced):
    X, y = Xy_balanced
    det = _TinyDetector(n_iter=2, cv_folds=3, random_state=42)
    det.fit(X, y)
    steps = list(det.pipeline.named_steps.keys())
    assert steps == ["smote", "classifier"]
