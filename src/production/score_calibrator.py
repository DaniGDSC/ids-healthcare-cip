"""Score calibrator — maps raw sigmoid scores to calibrated risk levels.

Supports two calibration modes:

1. **Two-class (preferred):** Uses both benign AND attack scores from the
   calibration phase. Finds the ROC-optimal threshold via Youden's J,
   then applies Platt scaling (logistic regression) to map the narrow
   sigmoid range (0.88-0.98) to well-calibrated probabilities (0-1).
   Risk levels are derived from calibrated probabilities.

2. **Benign-only (fallback):** When no attack samples are available,
   maps scores to percentiles of the benign distribution. Original
   behavior preserved for backward compatibility.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


class ScoreCalibrator:
    """Calibrate raw sigmoid scores for meaningful risk stratification.

    Auto-selects between two-class (ROC-optimal + Platt) and benign-only
    (percentile) modes based on available calibration data.

    Args:
        risk_thresholds: Percentile boundaries for benign-only mode.
        prob_thresholds: Probability boundaries for two-class mode.
        min_attack_samples: Minimum attack samples to enable two-class mode.
    """

    # Benign-only percentile thresholds (fallback mode)
    DEFAULT_THRESHOLDS: Dict[str, float] = {
        "NORMAL": 75.0,
        "LOW": 85.0,
        "MEDIUM": 93.0,
        "HIGH": 97.0,
    }

    # Two-class probability thresholds (primary mode)
    DEFAULT_PROB_THRESHOLDS: Dict[str, float] = {
        "NORMAL": 0.30,
        "LOW": 0.50,
        "MEDIUM": 0.70,    # detection boundary (MEDIUM+ = "detected")
        "HIGH": 0.90,
    }

    def __init__(
        self,
        risk_thresholds: Optional[Dict[str, float]] = None,
        prob_thresholds: Optional[Dict[str, float]] = None,
        min_attack_samples: Optional[int] = None,
    ) -> None:
        from config.production_loader import cfg

        bt = cfg("calibration.benign_thresholds", {})
        self._thresholds = risk_thresholds or {
            "NORMAL": bt.get("normal", 75.0),
            "LOW": bt.get("low", 85.0),
            "MEDIUM": bt.get("medium", 93.0),
            "HIGH": bt.get("high", 97.0),
        }
        self._prob_thresholds = prob_thresholds or self.DEFAULT_PROB_THRESHOLDS
        self._min_attack = min_attack_samples or int(cfg("calibration.min_attack_samples", 5))
        self._calibration_scores: Optional[np.ndarray] = None
        self._fitted = False
        self._use_two_class = False
        self._lock = threading.Lock()

        # Two-class state
        self._platt_scaler: Any = None
        self._optimal_threshold: float = 0.5
        self._youden_j: float = 0.0

    # ── Fitting ────────────────────────────────────────────────────────

    def fit(self, benign_scores: np.ndarray) -> None:
        """Fit on benign-only distribution (fallback mode).

        Args:
            benign_scores: Array of raw sigmoid scores from benign flows.
        """
        if len(benign_scores) == 0:
            logger.warning("Empty benign scores, cannot fit")
            return
        # Remove non-finite values
        mask = np.isfinite(benign_scores)
        if not mask.all():
            logger.warning("Removed %d non-finite scores from calibration", (~mask).sum())
            benign_scores = benign_scores[mask]
        with self._lock:
            self._calibration_scores = np.sort(benign_scores)
            self._fitted = True
            self._use_two_class = False
        logger.info(
            "ScoreCalibrator fitted (benign-only): %d samples, range [%.4f, %.4f]",
            len(benign_scores), benign_scores.min(), benign_scores.max(),
        )

    def fit_two_class(self, scores: np.ndarray, labels: np.ndarray) -> None:
        """Fit using both benign and attack scores (primary mode).

        Finds ROC-optimal threshold via Youden's J, then fits Platt
        scaling (logistic regression) to map narrow sigmoid range to
        calibrated probabilities.

        Args:
            scores: Raw sigmoid scores (benign + attack).
            labels: Binary labels (0=benign, 1=attack).
        """
        from config.production_loader import cfg
        from sklearn.linear_model import LogisticRegression

        scores = np.asarray(scores, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.int32)

        # Input validation
        if len(scores) == 0:
            logger.warning("Empty scores array, cannot fit two-class")
            return
        finite_mask = np.isfinite(scores)
        if not finite_mask.all():
            logger.warning("Removed %d non-finite scores", (~finite_mask).sum())
            scores, labels = scores[finite_mask], labels[finite_mask]

        # 1. Find optimal threshold via Youden's J (max TPR - FPR)
        sorted_scores = np.unique(scores)
        # Use midpoints between unique scores as candidate thresholds
        if len(sorted_scores) > 1:
            candidates = (sorted_scores[:-1] + sorted_scores[1:]) / 2
        else:
            candidates = sorted_scores

        best_j = -1.0
        best_t = float(np.median(scores))

        for t in candidates:
            pred = (scores >= t).astype(int)
            tp = np.sum((pred == 1) & (labels == 1))
            fn = np.sum((pred == 0) & (labels == 1))
            fp = np.sum((pred == 1) & (labels == 0))
            tn = np.sum((pred == 0) & (labels == 0))

            tpr = tp / max(tp + fn, 1)
            fpr = fp / max(fp + tn, 1)
            j = tpr - fpr

            if j > best_j:
                best_j = j
                best_t = float(t)

        # 2. Fit Platt scaling (logistic regression on 1D scores)
        platt_c = cfg("calibration.platt_regularization", 1e4)
        X = scores.reshape(-1, 1)
        platt_scaler = LogisticRegression(
            C=platt_c, solver="lbfgs", max_iter=1000,
        )
        platt_scaler.fit(X, labels)

        # 3. Benign scores for percentile fallback
        benign_mask = labels == 0
        cal_scores = np.sort(scores[benign_mask])

        # 4. Auto-tune probability thresholds using target-FPR approach.
        # The detection boundary is the LOW threshold — anything above it
        # enters MEDIUM territory and counts as "detected".
        probs_all = platt_scaler.predict_proba(X)[:, 1]
        benign_probs = np.sort(probs_all[benign_mask])

        target_fpr = cfg("calibration.target_fpr", 0.10)
        det_boundary = float(np.percentile(benign_probs, (1 - target_fpr) * 100))

        prob_thresholds = {
            "NORMAL": round(det_boundary * 0.50, 4),
            "LOW": round(det_boundary, 4),               # detection boundary (~10% FPR)
            "MEDIUM": round(det_boundary + (1 - det_boundary) * 0.33, 4),
            "HIGH": round(det_boundary + (1 - det_boundary) * 0.67, 4),
        }

        # Atomic state update under lock
        with self._lock:
            self._optimal_threshold = best_t
            self._youden_j = best_j
            self._platt_scaler = platt_scaler
            self._calibration_scores = cal_scores
            self._prob_thresholds = prob_thresholds
            self._fitted = True
            self._use_two_class = True

        # Log calibration quality
        probs = platt_scaler.predict_proba(X)[:, 1]
        logger.info(
            "ScoreCalibrator fitted (two-class): %d samples (%d benign, %d attack), "
            "Youden's J=%.3f, optimal_threshold=%.6f, "
            "Platt prob range=[%.3f, %.3f]",
            len(scores), benign_mask.sum(), (~benign_mask).sum(),
            best_j, best_t,
            probs.min(), probs.max(),
        )

    def fit_from_buffer(self, scores: list[float], labels: list[int]) -> None:
        """Fit from collected predictions during calibration phase.

        Auto-selects two-class mode if enough attack samples are present
        AND there is meaningful class separation (Youden's J > 0.3).
        Falls back to benign-only percentile mode otherwise.
        """
        scores_arr = np.array(scores, dtype=np.float64)
        labels_arr = np.array(labels, dtype=np.int32)

        n_benign = int((labels_arr == 0).sum())
        n_attack = int((labels_arr == 1).sum())

        if n_attack >= self._min_attack and n_benign >= 20:
            self.fit_two_class(scores_arr, labels_arr)
            # Fall back to benign-only if class separation is too weak
            if self._youden_j < 0.3:
                logger.info(
                    "Youden's J=%.3f < 0.3: weak separation, "
                    "falling back to benign-only mode",
                    self._youden_j,
                )
                benign = scores_arr[labels_arr == 0]
                self.fit(benign)
        elif n_benign >= 20:
            benign = scores_arr[labels_arr == 0]
            self.fit(benign)
        else:
            logger.warning(
                "Too few samples for calibration: %d benign, %d attack",
                n_benign, n_attack,
            )

    # ── Calibration ────────────────────────────────────────────────────

    def calibrate(
        self,
        raw_score: float,
        device_percentile: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Transform a raw score to calibrated risk level.

        Uses Platt scaling (two-class) or percentile mapping (benign-only)
        depending on how the calibrator was fitted.

        Args:
            raw_score: Raw sigmoid output from model.
            device_percentile: Device-specific detection percentile.
                Overrides the default MEDIUM boundary for this device.
                Lower values = more sensitive (e.g., 80 for infusion pump).

        Returns:
            Dict with percentile, risk_level, calibrated_score, raw_score.
        """
        if not np.isfinite(raw_score):
            return {
                "percentile": 0.0,
                "risk_level": "NORMAL",
                "calibrated_score": 0.0,
                "raw_score": float(raw_score),
            }

        with self._lock:
            if not self._fitted:
                return {
                    "percentile": raw_score * 100,
                    "risk_level": self._classify(raw_score * 100, device_percentile),
                    "calibrated_score": raw_score,
                    "raw_score": raw_score,
                }

            if self._use_two_class:
                return self._calibrate_two_class(raw_score)

            return self._calibrate_percentile(raw_score, device_percentile)

    def _calibrate_two_class(self, raw_score: float) -> Dict[str, Any]:
        """Calibrate using Platt scaling + probability thresholds."""
        X = np.array([[raw_score]])
        prob = float(self._platt_scaler.predict_proba(X)[:, 1][0])

        # Also compute percentile for display
        percentile = float(
            np.searchsorted(self._calibration_scores, raw_score)
            / max(len(self._calibration_scores), 1) * 100
        )

        return {
            "percentile": round(percentile, 1),
            "risk_level": self._classify_probability(prob),
            "calibrated_score": round(prob, 4),
            "raw_score": round(raw_score, 6),
        }

    def _calibrate_percentile(
        self, raw_score: float, device_percentile: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Calibrate using benign-distribution percentile (fallback)."""
        percentile = float(
            np.searchsorted(self._calibration_scores, raw_score)
            / max(len(self._calibration_scores), 1) * 100
        )

        return {
            "percentile": round(percentile, 1),
            "risk_level": self._classify(percentile, device_percentile),
            "calibrated_score": round(percentile / 100, 4),
            "raw_score": round(raw_score, 6),
        }

    # ── Classification ─────────────────────────────────────────────────

    def _classify(
        self, percentile: float, device_percentile: Optional[float] = None,
    ) -> str:
        """Map percentile to risk level (benign-only mode).

        If device_percentile is provided, it overrides the MEDIUM boundary
        for device-specific detection sensitivity.
        """
        medium_threshold = device_percentile or self._thresholds["MEDIUM"]
        # Derive LOW boundary proportionally (midpoint between NORMAL and MEDIUM)
        low_threshold = self._thresholds["LOW"] if device_percentile is None else (
            self._thresholds["NORMAL"] + (medium_threshold - self._thresholds["NORMAL"]) * 0.5
        )

        if percentile < self._thresholds["NORMAL"]:
            return "NORMAL"
        if percentile < low_threshold:
            return "LOW"
        if percentile < medium_threshold:
            return "MEDIUM"
        if percentile < self._thresholds["HIGH"]:
            return "HIGH"
        return "CRITICAL"

    def _classify_probability(self, prob: float) -> str:
        """Map calibrated probability to risk level (two-class mode)."""
        if prob < self._prob_thresholds["NORMAL"]:
            return "NORMAL"
        if prob < self._prob_thresholds["LOW"]:
            return "LOW"
        if prob < self._prob_thresholds["MEDIUM"]:
            return "MEDIUM"
        if prob < self._prob_thresholds["HIGH"]:
            return "HIGH"
        return "CRITICAL"

    # ── Properties and config ──────────────────────────────────────────

    @property
    def fitted(self) -> bool:
        return self._fitted

    def get_config(self) -> Dict[str, Any]:
        config: Dict[str, Any] = {
            "fitted": self._fitted,
            "mode": "two_class" if self._use_two_class else "benign_only",
            "n_calibration": len(self._calibration_scores) if self._calibration_scores is not None else 0,
        }
        if self._use_two_class:
            config.update({
                "optimal_threshold": self._optimal_threshold,
                "youden_j": round(self._youden_j, 4),
                "prob_thresholds": self._prob_thresholds,
                "platt_coef": float(self._platt_scaler.coef_[0][0]) if self._platt_scaler else None,
                "platt_intercept": float(self._platt_scaler.intercept_[0]) if self._platt_scaler else None,
            })
        else:
            config["thresholds"] = self._thresholds
        return config
