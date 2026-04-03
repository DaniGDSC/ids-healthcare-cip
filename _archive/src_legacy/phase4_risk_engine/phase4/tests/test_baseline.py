"""Unit tests for BaselineComputer (Median + MAD)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.phase4_risk_engine.phase4.base import BaseDetector
from src.phase4_risk_engine.phase4.baseline import BaselineComputer


def _make_attn_df(n_normal: int = 50, n_attack: int = 20) -> pd.DataFrame:
    """Create a fake attention DataFrame with Normal-only training data."""
    np.random.seed(42)
    n_dims = 128
    n_total = n_normal + n_attack

    data = {}
    for i in range(n_dims):
        data[f"attn_{i}"] = np.random.randn(n_total) * 0.01
    data["Label"] = [0] * n_normal + [1] * n_attack
    data["split"] = ["train"] * n_total

    return pd.DataFrame(data)


class TestBaselineComputer:
    """Test baseline computation from Normal-only attention scores."""

    def test_implements_base(self) -> None:
        assert issubclass(BaselineComputer, BaseDetector)

    def test_compute_normal_only(self) -> None:
        attn_df = _make_attn_df()
        computer = BaselineComputer(mad_multiplier=3.0)
        baseline = computer.compute(attn_df)

        assert baseline["n_normal_samples"] == 50
        assert baseline["n_attention_dims"] == 128
        assert baseline["mad_multiplier"] == 3.0
        assert baseline["median"] > 0
        assert baseline["mad"] >= 0
        assert baseline["baseline_threshold"] > baseline["median"]

    def test_immutable_output_keys(self) -> None:
        attn_df = _make_attn_df()
        computer = BaselineComputer(mad_multiplier=3.0)
        baseline = computer.compute(attn_df)

        expected_keys = {
            "median",
            "mad",
            "baseline_threshold",
            "mad_multiplier",
            "n_normal_samples",
            "n_attention_dims",
            "computed_at",
        }
        assert set(baseline.keys()) == expected_keys

    def test_mad_multiplier_effect(self) -> None:
        attn_df = _make_attn_df()
        baseline_low = BaselineComputer(mad_multiplier=2.0).compute(attn_df)
        baseline_high = BaselineComputer(mad_multiplier=5.0).compute(attn_df)

        assert baseline_high["baseline_threshold"] > baseline_low["baseline_threshold"]

    def test_get_config(self) -> None:
        computer = BaselineComputer(mad_multiplier=3.0)
        config = computer.get_config()
        assert config == {"mad_multiplier": 3.0}
