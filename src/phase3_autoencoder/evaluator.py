"""Autoencoder evaluation utilities."""

import numpy as np
import logging
from typing import Dict, Tuple

logger = logging.getLogger(__name__)


class AutoencoderEvaluator:
    """Evaluate autoencoder performance."""

    def __init__(self):
        pass

    def reconstruction_errors(self, model, X: np.ndarray) -> np.ndarray:
        reconstructions = model.predict(X, verbose=0)
        errors = np.mean(np.square(X - reconstructions), axis=1)
        return errors

    def evaluate(self, model, X: np.ndarray, metrics: Dict[str, bool] = None) -> Dict[str, float]:
        metrics = metrics or {'mse': True, 'mae': True, 'reconstruction_error': True}
        reconstructions = model.predict(X, verbose=0)

        results = {}
        if metrics.get('mse', False):
            results['mse'] = float(np.mean(np.square(X - reconstructions)))
        if metrics.get('mae', False):
            results['mae'] = float(np.mean(np.abs(X - reconstructions)))
        if metrics.get('reconstruction_error', False):
            results['reconstruction_error'] = float(np.mean(np.linalg.norm(X - reconstructions, axis=1)))

        logger.info(f"Autoencoder evaluation: {results}")
        return results

    def binary_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Compute confusion matrix and binary metrics."""
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'fpr': fpr,
            'fnr': fnr,
            'tp': float(tp),
            'fp': float(fp),
            'tn': float(tn),
            'fn': float(fn)
        }