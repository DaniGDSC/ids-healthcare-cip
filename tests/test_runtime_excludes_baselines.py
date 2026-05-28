"""Phase 2 guard: runtime path must never load RF/DT classifiers.

After Phase 2 the model registry is split into a *runtime* loader
(``get_track_a_classifiers``, XGBoost only) and a *baseline* loader
(``get_baseline_classifiers``, RandomForest + DecisionTree, used only
by ``tools/rq1_compute_metrics.compute_track_a_ablation``).

These tests enforce that contract at the registry layer so future
edits cannot silently reintroduce RF/DT into the engine / Module 4 /
Module 5 / Module 6 inference paths.
"""

from __future__ import annotations

from common.model_registry import (
    _BASELINE_TRACK_A_PATHS,
    _BASELINE_THRESHOLD_PATHS,
    _RUNTIME_TRACK_A_PATHS,
    _RUNTIME_THRESHOLD_PATHS,
)


def test_runtime_paths_xgboost_only() -> None:
    assert set(_RUNTIME_TRACK_A_PATHS.keys()) == {"xgboost"}
    assert set(_RUNTIME_THRESHOLD_PATHS.keys()) == {"xgboost"}


def test_baseline_paths_rf_dt_only() -> None:
    assert set(_BASELINE_TRACK_A_PATHS.keys()) == {"random_forest", "decision_tree"}
    assert set(_BASELINE_THRESHOLD_PATHS.keys()) == {"random_forest", "decision_tree"}


def test_runtime_and_baseline_disjoint() -> None:
    """A model name MUST appear in exactly one registry, never both."""
    runtime = set(_RUNTIME_TRACK_A_PATHS.keys())
    baseline = set(_BASELINE_TRACK_A_PATHS.keys())
    assert runtime & baseline == set()


def test_engine_constants_match_runtime_registry() -> None:
    """DetectionEngine.PRIMARY_TRACK_A must be a runtime model."""
    from detection_engine.engine import DetectionEngine
    assert DetectionEngine.PRIMARY_TRACK_A in _RUNTIME_TRACK_A_PATHS


def test_dae_augmentation_is_runtime_subset() -> None:
    """TRACK_A_FOR_DAE must reference only runtime models so the DAE input
    augmentation never depends on a baseline that runtime won't load."""
    from common.dae_input import TRACK_A_FOR_DAE
    assert set(TRACK_A_FOR_DAE).issubset(set(_RUNTIME_TRACK_A_PATHS.keys()))
