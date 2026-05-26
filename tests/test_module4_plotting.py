"""Module 4 plotting — all functions accept output_dir + produce PNGs."""
from __future__ import annotations

import numpy as np
import pytest

from module4_explanations.plotting import (
    plot_dae_global_weights,
    plot_global_importance_bar,
    plot_latency_cdf,
    plot_latency_component_breakdown,
    plot_latency_distribution,
)


def test_plot_global_importance_bar_writes_png(tmp_path):
    imp = [
        {"rank": 1, "feature": "Pulse_Rate", "mean_abs_shap": 0.5},
        {"rank": 2, "feature": "Dur", "mean_abs_shap": 0.3},
    ]
    out = plot_global_importance_bar("xgboost", imp, output_dir=tmp_path)
    assert out.exists()
    assert out.suffix == ".png"


def test_plot_dae_global_weights_writes_png(tmp_path):
    feat_weights = np.random.rand(10)
    feat_names = [f"f{i}" for i in range(10)]
    out = plot_dae_global_weights(feat_weights, feat_names, output_dir=tmp_path)
    assert out.exists()


def test_plot_latency_distribution_writes_png(tmp_path):
    timings = [{"total_ms": float(i)} for i in range(10, 110)]
    out = plot_latency_distribution(timings, output_dir=tmp_path)
    assert out.exists()


def test_plot_latency_cdf_writes_png(tmp_path):
    timings = [{"total_ms": float(i)} for i in range(10, 110)]
    out = plot_latency_cdf(timings, output_dir=tmp_path)
    assert out.exists()


def test_plot_latency_component_breakdown_writes_png(tmp_path):
    stats = {
        "predict_ms": {"p50": 5.0}, "treeshap_ms": {"p50": 20.0},
        "dae_decompose_ms": {"p50": 2.0}, "nlg_ms": {"p50": 1.0},
        "risk_decompose_ms": {"p50": 0.5},
    }
    out = plot_latency_component_breakdown(stats, output_dir=tmp_path)
    assert out.exists()
