"""Smoke tests for the 3 concrete Track A detector classes.

Heavy CV logic is covered by ``test_module2_base_detector.py``; this
file just verifies each concrete class instantiates, has the right
defaults, and produces the expected model_type label.
"""
from __future__ import annotations

import pytest

from module2_detection.models.DecisionTree import DecisionTreeDetector
from module2_detection.models.GradientBoosting import GradientBoostingDetector
from module2_detection.models.RandomForest import RandomForestDetector
from module2_detection.models.XGBoost import XGBoostDetector


# ── XGBoost compat alias ────────────────────────────────────────────


def test_xgboost_alias_is_gradient_boosting():
    """XGBoostDetector is retained as a deprecation alias."""
    assert XGBoostDetector is GradientBoostingDetector


def test_xgboost_param_space_still_importable():
    """Legacy ``from XGBoost import PARAM_SPACE`` still works."""
    from module2_detection.models.XGBoost import PARAM_SPACE
    assert "classifier__n_estimators" in PARAM_SPACE


# ── GradientBoostingDetector ────────────────────────────────────────


def test_gradient_boosting_defaults():
    det = GradientBoostingDetector(random_state=42)
    assert det.DEFAULT_N_ITER == 50
    assert det._n_iter == 50
    assert "XGBoost-equivalent" in det.MODEL_TYPE


def test_gradient_boosting_uses_sklearn_gbc():
    from sklearn.ensemble import GradientBoostingClassifier
    det = GradientBoostingDetector(random_state=42)
    clf = det._classifier()
    assert isinstance(clf, GradientBoostingClassifier)
    assert clf.random_state == 42


# ── RandomForestDetector ────────────────────────────────────────────


def test_random_forest_defaults():
    det = RandomForestDetector(random_state=42)
    assert det.DEFAULT_N_ITER == 40
    assert det.MODEL_TYPE == "RandomForestClassifier"


def test_random_forest_uses_n_jobs_minus_one():
    """RF uses n_jobs=-1 by default for parallelism."""
    det = RandomForestDetector(random_state=42)
    clf = det._classifier()
    assert clf.n_jobs == -1
    assert clf.bootstrap is True


# ── DecisionTreeDetector ────────────────────────────────────────────


def test_decision_tree_defaults():
    det = DecisionTreeDetector(random_state=42)
    assert det.DEFAULT_N_ITER == 25
    assert det.MODEL_TYPE == "DecisionTreeClassifier"


def test_decision_tree_uses_sklearn_dt():
    from sklearn.tree import DecisionTreeClassifier
    det = DecisionTreeDetector(random_state=42)
    clf = det._classifier()
    assert isinstance(clf, DecisionTreeClassifier)


# ── Param space distinctness ────────────────────────────────────────


def test_param_spaces_differ_across_detectors():
    """Each detector has its own hyperparameter search space."""
    gb_keys = set(GradientBoostingDetector.PARAM_SPACE.keys())
    rf_keys = set(RandomForestDetector.PARAM_SPACE.keys())
    dt_keys = set(DecisionTreeDetector.PARAM_SPACE.keys())
    # Each has at least one unique param the others don't
    assert "classifier__subsample" in gb_keys
    assert "classifier__class_weight" in rf_keys
    assert "classifier__splitter" in dt_keys


@pytest.mark.parametrize("detector_class,expected_default",
                         [(GradientBoostingDetector, 50),
                          (RandomForestDetector, 40),
                          (DecisionTreeDetector, 25)])
def test_default_n_iter_per_detector(detector_class, expected_default):
    """Default n_iter differs to reflect search-space size + cost."""
    det = detector_class(random_state=42)
    assert det.DEFAULT_N_ITER == expected_default
