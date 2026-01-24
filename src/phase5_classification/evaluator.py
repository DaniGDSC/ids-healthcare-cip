"""Classification evaluation utilities."""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
import logging
from typing import Dict

logger = logging.getLogger(__name__)


class ClassificationEvaluator:
    """Evaluate classification performance."""

    def __init__(self, average: str = 'macro'):
        self.average = average

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray = None) -> Dict[str, float]:
        results = {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred, average=self.average, zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, average=self.average, zero_division=0)),
            'f1_score': float(f1_score(y_true, y_pred, average=self.average, zero_division=0)),
        }

        if y_proba is not None and y_proba.ndim == 2 and y_proba.shape[1] > 1:
            try:
                results['roc_auc'] = float(roc_auc_score(y_true, y_proba, multi_class='ovr'))
            except Exception:
                logger.warning("ROC AUC could not be computed for given labels")

        results['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()
        logger.info(f"Classification metrics: {results}")
        return results