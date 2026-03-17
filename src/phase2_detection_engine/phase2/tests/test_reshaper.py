"""Unit tests for DataReshaper (sliding windows, shapes, edge cases)."""

from __future__ import annotations

import numpy as np
import pytest

from src.phase2_detection_engine.phase2.reshaper import DataReshaper


class TestDataReshaper:
    """Test sliding window creation."""

    @pytest.fixture()
    def data(self):
        """Small dataset: 50 samples × 5 features."""
        rng = np.random.RandomState(42)
        X = rng.randn(50, 5).astype(np.float32)
        y = np.array([0] * 40 + [1] * 10, dtype=np.int32)
        return X, y

    def test_output_shape_stride1(self, data) -> None:
        X, y = data
        reshaper = DataReshaper(timesteps=10, stride=1)
        X_w, y_w = reshaper.reshape(X, y)
        # n_windows = (50 - 10) // 1 + 1 = 41
        assert X_w.shape == (41, 10, 5)
        assert y_w.shape == (41,)

    def test_output_shape_stride5(self, data) -> None:
        X, y = data
        reshaper = DataReshaper(timesteps=10, stride=5)
        X_w, y_w = reshaper.reshape(X, y)
        # n_windows = (50 - 10) // 5 + 1 = 9
        assert X_w.shape == (9, 10, 5)
        assert y_w.shape == (9,)

    def test_label_is_last_sample(self, data) -> None:
        X, y = data
        reshaper = DataReshaper(timesteps=10, stride=1)
        X_w, y_w = reshaper.reshape(X, y)
        # Label of window i = label of sample i + timesteps - 1
        for i in range(len(y_w)):
            assert y_w[i] == y[i + 10 - 1]

    def test_window_content(self, data) -> None:
        X, y = data
        reshaper = DataReshaper(timesteps=5, stride=1)
        X_w, _ = reshaper.reshape(X, y)
        # First window should be X[0:5]
        np.testing.assert_array_almost_equal(X_w[0], X[0:5])
        # Second window should be X[1:6]
        np.testing.assert_array_almost_equal(X_w[1], X[1:6])

    def test_dtype_float32(self, data) -> None:
        X, y = data
        reshaper = DataReshaper(timesteps=5)
        X_w, _ = reshaper.reshape(X, y)
        assert X_w.dtype == np.float32

    def test_too_few_samples(self) -> None:
        X = np.random.randn(5, 3).astype(np.float32)
        y = np.zeros(5, dtype=np.int32)
        reshaper = DataReshaper(timesteps=10)
        with pytest.raises(ValueError, match="Cannot create windows"):
            reshaper.reshape(X, y)

    def test_exact_fit(self) -> None:
        """n_samples == timesteps → exactly 1 window."""
        X = np.random.randn(10, 3).astype(np.float32)
        y = np.zeros(10, dtype=np.int32)
        reshaper = DataReshaper(timesteps=10, stride=1)
        X_w, y_w = reshaper.reshape(X, y)
        assert X_w.shape == (1, 10, 3)
        assert y_w.shape == (1,)

    def test_timesteps_validation(self) -> None:
        with pytest.raises(ValueError, match="timesteps"):
            DataReshaper(timesteps=1)

    def test_stride_validation(self) -> None:
        with pytest.raises(ValueError, match="stride"):
            DataReshaper(timesteps=5, stride=0)

    def test_get_config(self) -> None:
        reshaper = DataReshaper(timesteps=20, stride=2)
        cfg = reshaper.get_config()
        assert cfg == {"timesteps": 20, "stride": 2}
