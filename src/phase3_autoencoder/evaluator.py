"""Autoencoder evaluation utilities."""

import numpy as np
import logging
from typing import Dict

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