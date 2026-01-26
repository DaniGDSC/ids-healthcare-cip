"""Optimize anomaly detection threshold."""

import numpy as np
import logging
from typing import Tuple, Dict

logger = logging.getLogger(__name__)


class ThresholdOptimizer:
    """Optimize reconstruction error threshold for anomaly detection."""

    def __init__(self, method: str = 'percentile', percentile: float = 95.0, n_std: float = 3.0):
        self.method = method
        self.percentile = percentile
        self.n_std = n_std

    def compute_threshold(self, errors: np.ndarray) -> float:
        if self.method == 'percentile':
            threshold = np.percentile(errors, self.percentile)
        elif self.method == 'std':
            threshold = np.mean(errors) + self.n_std * np.std(errors)
        elif self.method == 'mad':
            median = np.median(errors)
            mad = np.median(np.abs(errors - median))
            threshold = median + self.n_std * mad
        else:
            raise ValueError(f"Unknown threshold method: {self.method}")

        logger.info(f"Threshold ({self.method}): {threshold:.6f}")
        return threshold

    def grid_search_f1(self, errors: np.ndarray, y_true: np.ndarray,
                        percentiles: np.ndarray) -> Tuple[float, Dict[str, float]]:
        """Search thresholds over percentiles to maximize F1 on validation.

        Returns best_threshold and metrics at that threshold.
        """
        best_f1 = -1.0
        best_thr = None
        best_metrics: Dict[str, float] = {}
        for p in percentiles:
            thr = float(np.percentile(errors, p))
            y_pred = (errors > thr).astype(int)
            tp = int(np.sum((y_true == 1) & (y_pred == 1)))
            fp = int(np.sum((y_true == 0) & (y_pred == 1)))
            tn = int(np.sum((y_true == 0) & (y_pred == 0)))
            fn = int(np.sum((y_true == 1) & (y_pred == 0)))
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            if f1 > best_f1:
                best_f1 = f1
                best_thr = thr
                best_metrics = {
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'fp': fp,
                    'fn': fn,
                    'tp': tp,
                    'tn': tn,
                    'threshold': thr,
                    'percentile': float(p)
                }
        logger.info(f"Grid-search best threshold: {best_thr:.6f} at p={best_metrics.get('percentile')} with F1={best_f1:.4f}")
        return float(best_thr), best_metrics