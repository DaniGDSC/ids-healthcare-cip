"""Tests for SearchSpace — grid and random candidate generation."""

from __future__ import annotations

import pytest

from src.phase2_5_fine_tuning.phase2_5.config import SearchSpaceConfig
from src.phase2_5_fine_tuning.phase2_5.search_space import SearchSpace


def _make_small_space() -> SearchSpaceConfig:
    """Create a small search space for fast tests."""
    return SearchSpaceConfig(
        cnn_filters_1=[32, 64],
        cnn_filters_2=[128],
        cnn_kernel_size=[3],
        bilstm_units_1=[128],
        bilstm_units_2=[64],
        dropout_rate=[0.3],
        attention_units=[128],
        timesteps=[20],
        batch_size=[256],
        learning_rate=[0.001],
    )


class TestSearchSpace:
    """Validate SearchSpace grid and random generation."""

    def test_grid_generates_all_combinations(self) -> None:
        space = SearchSpace(_make_small_space(), random_state=42)
        combos = space.grid()
        # 2 * 1 * 1 * 1 * 1 * 1 * 1 * 1 * 1 * 1 = 2
        assert len(combos) == 2

    def test_grid_configs_have_all_keys(self) -> None:
        space = SearchSpace(_make_small_space(), random_state=42)
        combos = space.grid()
        expected_keys = {
            "cnn_filters_1", "cnn_filters_2", "cnn_kernel_size",
            "bilstm_units_1", "bilstm_units_2", "dropout_rate",
            "attention_units", "timesteps", "batch_size", "learning_rate",
        }
        for combo in combos:
            assert set(combo.keys()) == expected_keys

    def test_random_respects_max_trials(self) -> None:
        space = SearchSpace(_make_small_space(), random_state=42)
        sampled = space.random(max_trials=1)
        assert len(sampled) == 1

    def test_random_capped_at_grid_size(self) -> None:
        space = SearchSpace(_make_small_space(), random_state=42)
        sampled = space.random(max_trials=100)
        assert len(sampled) == 2  # Only 2 combos exist

    def test_total_combinations(self) -> None:
        space = SearchSpace(_make_small_space(), random_state=42)
        assert space.total_combinations() == 2

    def test_reproducibility(self) -> None:
        space1 = SearchSpace(_make_small_space(), random_state=42)
        space2 = SearchSpace(_make_small_space(), random_state=42)
        assert space1.random(1) == space2.random(1)

    def test_different_seeds_differ(self) -> None:
        big_space = SearchSpaceConfig(
            cnn_filters_1=[32, 64, 128],
            cnn_filters_2=[64, 128, 256],
            cnn_kernel_size=[3, 5],
            bilstm_units_1=[64, 128],
            bilstm_units_2=[32, 64],
            dropout_rate=[0.1, 0.3, 0.5],
            attention_units=[64, 128],
            timesteps=[10, 20],
            batch_size=[128, 256],
            learning_rate=[0.001, 0.0001],
        )
        s1 = SearchSpace(big_space, random_state=42)
        s2 = SearchSpace(big_space, random_state=99)
        # With many combos, different seeds should give different samples
        r1 = s1.random(5)
        r2 = s2.random(5)
        assert r1 != r2
