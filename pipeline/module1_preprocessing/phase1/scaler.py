"""Robust scaler transformer — fit on train only.

Wraps ``sklearn.preprocessing.RobustScaler`` directly.  Uses median
and IQR, making it robust to the heavy-tailed distributions
identified in the Phase 0 outlier analysis (§3.2.1).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from .base import BaseTransformer

logger = logging.getLogger(__name__)

_SCALERS = {
    "robust": RobustScaler,
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
}


class RobustScalerTransformer(BaseTransformer):
    """Fit RobustScaler on training data, transform both partitions.

    The scaler is **never fitted on test data** — preventing
    information leakage from test distribution.

    Args:
        method: Scaling method (``"robust"``, ``"standard"``, ``"minmax"``).
    """

    def __init__(self, method: str = "robust") -> None:
        scaler_cls = _SCALERS.get(method)
        if scaler_cls is None:
            raise ValueError(f"Unknown method '{method}'. Use: {list(_SCALERS)}")
        self._scaler = scaler_cls()
        self._method = method
        self._fitted = False

    def fit(self, X_train: np.ndarray) -> RobustScalerTransformer:
        """Fit the scaler on training data only.

        Args:
            X_train: Training feature matrix.

        Returns:
            self
        """
        self._scaler.fit(X_train)
        self._fitted = True
        logger.info("RobustScalerTransformer: fitted on %d×%d", *X_train.shape)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data using the fitted scaler.

        Args:
            X: Feature matrix to scale.

        Returns:
            Scaled feature matrix.

        Raises:
            RuntimeError: If ``fit()`` has not been called.
        """
        if not self._fitted:
            raise RuntimeError("Scaler not fitted. Call fit(X_train) first.")
        return self._scaler.transform(X)

    def fit_transform(self, X_train: np.ndarray) -> np.ndarray:
        """Fit on training data and transform it.

        Args:
            X_train: Training feature matrix.

        Returns:
            Scaled training feature matrix.
        """
        return self.fit(X_train).transform(X_train)

    def scale_both(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fit on train, transform both train and test.

        Args:
            X_train: Training feature matrix.
            X_test: Test feature matrix.

        Returns:
            Tuple of (X_train_scaled, X_test_scaled).
        """
        X_train_s = self.fit_transform(X_train)
        X_test_s = self.transform(X_test)
        logger.info(
            "RobustScalerTransformer: train %d×%d, test %d×%d",
            *X_train_s.shape, *X_test_s.shape,
        )
        return X_train_s, X_test_s

    def save(self, path: Path) -> None:
        """Persist the fitted scaler to disk.

        Args:
            path: Destination pickle file.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._scaler, path)
        logger.info("Scaler saved: %s", path)

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def get_report(self) -> Dict[str, Any]:
        return {"method": self._method, "fitted": self._fitted}
