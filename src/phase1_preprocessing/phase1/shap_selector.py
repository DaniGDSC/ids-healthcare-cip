"""XGBoost-RFE + SHAP feature selector.

Uses Recursive Feature Elimination with a GradientBoosting surrogate
(XGBoost stand-in) scored by cross-validated F1.  After RFECV selects
the optimal subset, a SHAP cross-check removes any feature whose
mean |SHAP| falls below a relative threshold (% of max).

Applied after the stratified split, on training data only.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)


class SHAPSelector:
    """XGBoost-RFE with SHAP cross-validation.

    1. Run RFECV: iteratively prune the least-important feature
       (by SHAP ranking), recording CV weighted-F1 at each step.
    2. SHAP cross-check: on the RFECV-selected subset, compute SHAP
       and drop any feature below ``shap_threshold`` × max(mean|SHAP|).
    3. Apply the final mask to both train and test.

    Args:
        min_features: Stop RFE when this many features remain.
        n_estimators: Trees in GradientBoosting surrogate.
        cv_folds: Folds for RFECV scoring.
        shap_threshold: Relative threshold for SHAP cross-check
            (fraction of max feature importance).
        random_state: Seed for reproducibility.
    """

    def __init__(
        self,
        min_features: int = 5,
        n_estimators: int = 100,
        cv_folds: int = 5,
        shap_threshold: float = 0.01,
        random_state: int = 42,
    ) -> None:
        self._min_features = min_features
        self._n_estimators = n_estimators
        self._cv_folds = cv_folds
        self._shap_threshold = shap_threshold
        self._random_state = random_state
        self._initial_importances: Dict[str, float] = {}
        self._rfe_elimination_order: List[str] = []
        self._cv_scores: List[Tuple[int, float]] = []
        self._rfe_selected: List[str] = []
        self._shap_crosscheck: Dict[str, float] = {}
        self._selected: List[str] = []
        self._dropped: List[str] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_model(self) -> GradientBoostingClassifier:
        return GradientBoostingClassifier(
            n_estimators=self._n_estimators,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            random_state=self._random_state,
        )

    @staticmethod
    def _sample_weight(y: np.ndarray) -> np.ndarray:
        n_neg = int((y == 0).sum())
        n_pos = int((y == 1).sum())
        spw = n_neg / n_pos if n_pos > 0 else 1.0
        return np.where(y == 1, spw, 1.0)

    def _shap_importances(
        self, X: np.ndarray, y: np.ndarray, sw: np.ndarray,
    ) -> np.ndarray:
        """Train model, return mean |SHAP| vector."""
        import shap

        gbc = self._build_model()
        gbc.fit(X, y, sample_weight=sw)
        explainer = shap.TreeExplainer(gbc)
        shap_values = explainer.shap_values(X)

        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        arr = np.array(shap_values)
        if arr.ndim == 3:
            arr = arr[:, :, 1]

        return np.abs(arr).mean(axis=0)

    def _cv_f1(
        self, X: np.ndarray, y: np.ndarray, sw: np.ndarray,
    ) -> float:
        """Stratified CV weighted-F1 with sample_weight."""
        from sklearn.metrics import f1_score

        skf = StratifiedKFold(
            n_splits=self._cv_folds, shuffle=True,
            random_state=self._random_state,
        )
        scores = []
        for train_idx, val_idx in skf.split(X, y):
            model = self._build_model()
            model.fit(X[train_idx], y[train_idx], sample_weight=sw[train_idx])
            y_pred = model.predict(X[val_idx])
            scores.append(f1_score(y[val_idx], y_pred, average="weighted"))
        return float(np.mean(scores))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        feature_names: List[str],
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Run RFECV + SHAP cross-check on train, apply to both.

        Args:
            X_train: Training feature matrix.
            X_test: Test feature matrix.
            y_train: Training labels (binary).
            feature_names: Ordered column names.

        Returns:
            Tuple of (X_train_filtered, X_test_filtered, selected_names).
        """
        sw = self._sample_weight(y_train)
        current_names = list(feature_names)
        current_idx = list(range(len(feature_names)))

        # ── Initial SHAP importances ──
        full_imp = self._shap_importances(X_train, y_train, sw)
        self._initial_importances = {
            n: float(v) for n, v in zip(feature_names, full_imp)
        }

        # ── RFECV: prune by SHAP ranking, track CV F1 ──
        score = self._cv_f1(X_train, y_train, sw)
        self._cv_scores.append((len(current_names), float(score)))
        best_score = score
        best_idx = list(current_idx)
        best_names = list(current_names)

        logger.info(
            "RFECV start: %d features, CV F1=%.4f",
            len(current_names), score,
        )

        while len(current_names) > self._min_features:
            X_sub = X_train[:, current_idx]
            imp = self._shap_importances(X_sub, y_train, sw)

            worst_local = int(np.argmin(imp))
            worst_name = current_names[worst_local]
            self._rfe_elimination_order.append(worst_name)

            current_names.pop(worst_local)
            current_idx.pop(worst_local)

            X_sub = X_train[:, current_idx]
            score = self._cv_f1(X_sub, y_train, sw)
            self._cv_scores.append((len(current_names), float(score)))

            logger.info(
                "RFECV step: dropped '%s', %d remain, CV F1=%.4f",
                worst_name, len(current_names), score,
            )

            if score >= best_score:
                best_score = score
                best_idx = list(current_idx)
                best_names = list(current_names)

        self._rfe_selected = list(best_names)

        # ── SHAP cross-check on RFECV-selected features ──
        rfe_mask = np.array([i in best_idx for i in range(len(feature_names))])
        X_rfe_train = X_train[:, rfe_mask]
        shap_imp = self._shap_importances(X_rfe_train, y_train, sw)
        self._shap_crosscheck = {
            n: float(v) for n, v in zip(best_names, shap_imp)
        }

        max_shap = shap_imp.max() if len(shap_imp) > 0 else 1.0
        abs_threshold = self._shap_threshold * max_shap
        keep_mask = shap_imp >= abs_threshold

        final_names = [n for n, m in zip(best_names, keep_mask) if m]
        shap_dropped = [n for n, m in zip(best_names, keep_mask) if not m]

        if shap_dropped:
            logger.info(
                "SHAP cross-check: dropped %d features below %.1f%% threshold: %s",
                len(shap_dropped), self._shap_threshold * 100, shap_dropped,
            )

        # ── Build final mask ──
        final_set = set(final_names)
        final_mask = np.array([n in final_set for n in feature_names])

        self._selected = final_names
        self._dropped = [n for n in feature_names if n not in final_set]

        X_train_sel = X_train[:, final_mask]
        X_test_sel = X_test[:, final_mask]

        logger.info(
            "SHAP-RFE done: %d/%d features (RFECV=%d, SHAP cross-check dropped %d), "
            "best CV F1=%.4f",
            len(self._selected), len(feature_names),
            len(self._rfe_selected), len(shap_dropped), best_score,
        )
        return X_train_sel, X_test_sel, self._selected

    def get_report(self) -> Dict[str, Any]:
        return {
            "method": "XGBoost-RFE + SHAP cross-check",
            "surrogate": "GradientBoostingClassifier",
            "initial_importances": self._initial_importances,
            "rfe_elimination_order": self._rfe_elimination_order,
            "cv_scores": self._cv_scores,
            "rfe_selected": self._rfe_selected,
            "shap_crosscheck_importances": self._shap_crosscheck,
            "shap_threshold": self._shap_threshold,
            "selected": self._selected,
            "dropped": self._dropped,
            "n_selected": len(self._selected),
            "n_dropped": len(self._dropped),
        }
