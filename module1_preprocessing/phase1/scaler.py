"""Robust scaler transformer — fit on train only, persisted as a JSON sidecar.

Wraps ``sklearn.preprocessing.RobustScaler`` directly. Uses median and
IQR, making it robust to the heavy-tailed distributions identified in
the Phase 0 outlier analysis (§3.2.1).

Persistence model
-----------------
The fitted scaler is written to disk as a **JSON sidecar** containing
only the learned parameters (``center_``, ``scale_``, ``n_features_in_``,
plus the scaler ``method`` used). It is NOT written as a joblib pickle.

Why: ``joblib.load`` (and any ``pickle.load`` underneath) executes
arbitrary Python embedded in the byte stream at deserialisation time,
making any trust boundary that crosses the file a remote-code-execution
sink. Phase 1's scaler artefact is consumed only as inspection /
audit material — production inference loads the sklearn Pipelines that
embed the scaler internally — so there is no need to ship a pickle.
The JSON sidecar carries exactly the information needed to reconstruct
the scaler via ``RobustScalerTransformer.from_json`` and refuses to
execute anything during load.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from ._sidecar_io import atomic_write_json, load_sidecar, migrate_legacy_pkl
from .base import BaseTransformer

logger = logging.getLogger(__name__)

_SCALERS = {
    "robust": RobustScaler,
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
}

# Per-scaler-type list of fitted attribute names that fully determine
# the transform. Anything outside this allowlist is NOT serialised, so
# a future sklearn version that adds an executable attribute cannot
# silently smuggle data through the JSON sidecar.
_SCALER_PARAMS: Dict[str, Tuple[str, ...]] = {
    "robust": ("center_", "scale_", "n_features_in_"),
    "standard": ("mean_", "scale_", "var_", "n_features_in_"),
    "minmax": ("min_", "scale_", "data_min_", "data_max_", "data_range_", "n_features_in_"),
}

_SIDECAR_FORMAT = "phase1.scaler.v1"
_SIDECAR_FORMAT_VERSION = 1


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
            *X_train_s.shape,
            *X_test_s.shape,
        )
        return X_train_s, X_test_s

    def save(self, path: Path) -> None:
        """Persist the fitted scaler as a JSON sidecar (NOT a pickle).

        The output file contains only the learned numeric parameters
        (``center_``/``scale_``/``n_features_in_`` for RobustScaler,
        with the matching set for the other scaler types). Loading is
        a pure ``json.loads`` + ``np.asarray`` round-trip; no Python
        code is ever executed during deserialisation.

        If the destination path was historically ``robust_scaler.pkl``,
        the actual file is written next to it as ``robust_scaler.json``
        and any pre-existing ``robust_scaler.pkl`` is removed so a
        downstream consumer cannot silently load a stale pickle.

        Args:
            path: Destination path. ``.pkl`` is rewritten to ``.json``.
        """
        if not self._fitted:
            raise RuntimeError("Scaler not fitted. Call fit(X_train) first.")

        path = migrate_legacy_pkl(Path(path), "scaler")

        attrs = _SCALER_PARAMS.get(self._method)
        if attrs is None:
            raise ValueError(f"Refusing to serialise unknown scaler method '{self._method}'")

        params: Dict[str, Any] = {}
        for attr in attrs:
            if not hasattr(self._scaler, attr):
                raise RuntimeError(
                    f"Scaler {type(self._scaler).__name__} has no fitted "
                    f"attribute '{attr}' — cannot serialise."
                )
            value = getattr(self._scaler, attr)
            if isinstance(value, np.ndarray):
                params[attr] = value.tolist()
            elif isinstance(value, (np.integer, np.floating)):
                params[attr] = value.item()
            else:
                params[attr] = value

        body = {
            "format": _SIDECAR_FORMAT,
            "format_version": _SIDECAR_FORMAT_VERSION,
            "method": self._method,
            "params": params,
        }
        atomic_write_json(path, body)
        logger.info("Scaler sidecar saved: %s (method=%s)", path, self._method)

    @classmethod
    def from_json(cls, path: Path) -> "RobustScalerTransformer":
        """Reconstruct a fitted scaler from a JSON sidecar.

        This is the only supported load path. There is no
        ``RobustScalerTransformer.from_pickle`` and never will be — a
        pickle loader is the same RCE sink the JSON sidecar exists to
        avoid.

        Args:
            path: Path to a sidecar previously written by ``save()``.

        Returns:
            A fitted ``RobustScalerTransformer`` whose internal sklearn
            scaler has the same ``transform`` behaviour as the original.

        Raises:
            FileNotFoundError: if *path* does not exist.
            ValueError: if the file is not a recognised sidecar.
        """
        path = Path(path)
        body = load_sidecar(path, _SIDECAR_FORMAT, "scaler")
        method = body.get("method")
        if method not in _SCALERS:
            raise ValueError(f"Unknown scaler method '{method}' in {path}")

        attrs = _SCALER_PARAMS[method]
        params = body.get("params", {})

        instance = cls(method=method)
        # Materialise a fresh, fitted sklearn scaler by setting the
        # learned attributes directly. Each attribute is checked against
        # the per-method allowlist; everything else is rejected.
        scaler = _SCALERS[method]()
        for attr in attrs:
            if attr not in params:
                raise ValueError(
                    f"{path}: missing required parameter '{attr}' for " f"method '{method}'"
                )
            value = params[attr]
            if isinstance(value, list):
                value = np.asarray(value, dtype=np.float64)
            setattr(scaler, attr, value)
        instance._scaler = scaler
        instance._fitted = True
        logger.info(
            "Scaler sidecar loaded: %s (method=%s, n_features=%s)",
            path,
            method,
            params.get("n_features_in_"),
        )
        return instance

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def get_report(self) -> Dict[str, Any]:
        return {"method": self._method, "fitted": self._fitted}
