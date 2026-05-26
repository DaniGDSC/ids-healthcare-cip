"""Module 4 batch_sim — latency stats + run_batch_simulation."""
from __future__ import annotations

import numpy as np

from module4_explanations.batch_sim import compute_latency_stats


def test_compute_latency_stats_empty_input():
    assert compute_latency_stats([]) == {}


def test_compute_latency_stats_returns_percentiles():
    timings = [
        {"total_ms": float(i), "predict_ms": float(i / 10)}
        for i in range(1, 101)
    ]
    stats = compute_latency_stats(timings)
    assert "total_ms" in stats
    assert "predict_ms" in stats
    # p50 of 1..100 = 50.5 (numpy percentile)
    assert 50 <= stats["total_ms"]["p50"] <= 51
    assert stats["total_ms"]["p95"] > stats["total_ms"]["p50"]
    assert stats["total_ms"]["p99"] >= stats["total_ms"]["p95"]


def test_compute_latency_stats_n_samples_correct():
    timings = [{"total_ms": float(i)} for i in range(50)]
    stats = compute_latency_stats(timings)
    assert stats["total_ms"]["n_samples"] == 50


def test_compute_latency_stats_min_max_consistent():
    timings = [{"total_ms": float(i)} for i in range(10, 30)]
    stats = compute_latency_stats(timings)
    assert stats["total_ms"]["min"] == 10.0
    assert stats["total_ms"]["max"] == 29.0


def test_compute_latency_stats_handles_missing_keys():
    """Some timings missing a component → use available ones only."""
    timings = [
        {"total_ms": 10.0, "predict_ms": 5.0},
        {"total_ms": 20.0},
        {"total_ms": 30.0, "predict_ms": 15.0},
    ]
    stats = compute_latency_stats(timings)
    assert stats["total_ms"]["n_samples"] == 3
    assert stats["predict_ms"]["n_samples"] == 2


def test_run_batch_simulation_rejects_feat_names_mismatch():
    """Y10: batch sim requires feat_names matches explainer's constructor."""
    from module4_explanations.online_explainer import AlertExplainer
    import pytest

    # Mock explainer with known feat_names
    explainer = AlertExplainer.__new__(AlertExplainer)
    explainer.feat_names = ("f1", "f2")
    explainer.classifiers = {}
    explainer.explainers = {}
    explainer.thresholds = {}
    explainer.dae = None

    from module4_explanations.batch_sim import run_batch_simulation
    X = np.zeros((1, 2), dtype=np.float32)
    y_pred = np.zeros(1, dtype=int)
    with pytest.raises(ValueError, match="feat_names mismatch"):
        run_batch_simulation(explainer, X, y_pred, ["x", "y", "z"])
