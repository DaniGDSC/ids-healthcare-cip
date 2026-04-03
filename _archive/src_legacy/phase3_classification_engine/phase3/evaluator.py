"""Model evaluator — compute classification metrics on test set only.

Never evaluates on training data (data leakage prevention).
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Compute classification metrics on the test set only.

    Args:
        threshold: Classification threshold for binary (default 0.5).
    """

    def __init__(self, threshold: float = 0.5) -> None:
        self._threshold = threshold

    def evaluate(
        self,
        model: tf.keras.Model,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, Any]:
        """Compute all evaluation metrics.

        Args:
            model: Trained classification model.
            X_test: Windowed test features.
            y_test: Windowed test labels.

        Returns:
            Dict with accuracy, f1_score, precision, recall,
            auc_roc, confusion_matrix, classification_report,
            threshold, and test_samples.
        """
        logger.info("── Evaluation ──")

        y_pred_prob = model.predict(X_test, verbose=0)

        if y_pred_prob.shape[-1] == 1:
            y_pred_prob = y_pred_prob.ravel()
            y_pred = (y_pred_prob > self._threshold).astype(int)
        else:
            y_pred = np.argmax(y_pred_prob, axis=1)

        acc = float(accuracy_score(y_test, y_pred))
        f1 = float(f1_score(y_test, y_pred, average="weighted"))
        prec = float(precision_score(y_test, y_pred, average="weighted"))
        rec = float(recall_score(y_test, y_pred, average="weighted"))
        auc = float(roc_auc_score(y_test, y_pred_prob))
        cm = confusion_matrix(y_test, y_pred)
        cls_report = classification_report(y_test, y_pred, output_dict=True)

        logger.info("  Accuracy:  %.4f", acc)
        logger.info("  F1-score:  %.4f", f1)
        logger.info("  Precision: %.4f", prec)
        logger.info("  Recall:    %.4f", rec)
        logger.info("  AUC-ROC:   %.4f", auc)

        return {
            "accuracy": acc,
            "f1_score": f1,
            "precision": prec,
            "recall": rec,
            "auc_roc": auc,
            "confusion_matrix": cm.tolist(),
            "classification_report": cls_report,
            "threshold": self._threshold,
            "test_samples": len(y_test),
        }
