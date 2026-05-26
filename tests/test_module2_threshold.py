"""`models/_threshold.py` — F-beta threshold optimisation tests."""
from __future__ import annotations

import numpy as np

from module2_detection.models._threshold import find_optimal_threshold


def test_single_class_returns_default_threshold():
    """precision_recall_curve needs 2 classes; fall back to 0.5."""
    y_true = np.zeros(20, dtype=int)
    y_proba = np.random.rand(20)
    assert find_optimal_threshold(y_true, y_proba) == 0.5


def test_all_attack_class_returns_default_threshold():
    y_true = np.ones(20, dtype=int)
    y_proba = np.random.rand(20)
    assert find_optimal_threshold(y_true, y_proba) == 0.5


def test_perfect_separation_picks_separating_threshold():
    """If proba perfectly separates classes, threshold lies in the gap."""
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
    t = find_optimal_threshold(y_true, y_proba, beta=2.0)
    # Optimal F2 threshold for this perfect-separation case is between 0.4 and 0.6
    assert 0.4 < t <= 0.6


def test_all_zero_proba_returns_valid_threshold():
    """Degenerate all-zero proba should still return a valid threshold
    in [0, 1] without crashing, even though every prediction collapses
    to the same class. The precise value depends on how
    precision_recall_curve handles the degenerate case; we just assert
    sanity here.
    """
    y_true = np.array([0, 1, 0, 1])
    y_proba = np.zeros(4)
    t = find_optimal_threshold(y_true, y_proba)
    assert 0.0 <= t <= 1.0


def test_beta_2_weights_recall_over_precision():
    """F2 should pick a lower threshold than F0.5 (more recall, less precision)."""
    rng = np.random.default_rng(0)
    y_true = (rng.uniform(0, 1, 1000) < 0.3).astype(int)
    y_proba = y_true * 0.6 + rng.uniform(0, 0.4, 1000)
    t_f2 = find_optimal_threshold(y_true, y_proba, beta=2.0)
    t_fhalf = find_optimal_threshold(y_true, y_proba, beta=0.5)
    assert t_f2 <= t_fhalf, (
        f"F2 (recall-weighted) should pick threshold ≤ F0.5 "
        f"(precision-weighted): got F2={t_f2} F0.5={t_fhalf}"
    )


def test_no_numpy_warnings_on_degenerate_input():
    """np.where on (p=0, r=0) used to raise an invalid-value warning."""
    import warnings
    y_true = np.array([0, 1, 0, 1])
    y_proba = np.array([0.0, 0.0, 0.0, 0.0])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        # Must not raise — errstate context suppresses the warning
        find_optimal_threshold(y_true, y_proba)
