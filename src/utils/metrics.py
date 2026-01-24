"""Metrics utilities."""

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
import numpy as np


def compute_classification_metrics(y_true, y_pred, y_proba=None, average='macro'):
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, average=average, zero_division=0)),
        'f1_score': float(f1_score(y_true, y_pred, average=average, zero_division=0)),
    }

    if y_proba is not None and y_proba.ndim == 2 and y_proba.shape[1] > 1:
        try:
            metrics['roc_auc'] = float(roc_auc_score(y_true, y_proba, multi_class='ovr'))
        except Exception:
            pass

    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()
    return metrics