"""
Phase 5: Classification Evaluation and Metrics

Comprehensive evaluation module for ensemble classifier performance including:
- Confusion matrix and per-class metrics
- ROC/PR curves
- Feature importance analysis
- Confidence score distributions
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix,
    classification_report
)
from typing import Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class ClassificationEvaluator:
    """
    Comprehensive evaluation of ensemble classification performance.
    
    Computes:
    - Overall metrics (accuracy, precision, recall, F1)
    - Per-class metrics with support
    - Confusion matrix
    - ROC-AUC and confidence intervals
    - Precision-recall curves
    - FP/FN analysis
    """
    
    def __init__(self, class_names: list = None):
        """
        Initialize evaluator.
        
        Parameters
        ----------
        class_names : list, optional
            Names for classes (default: ['Benign', 'BruteForce', 'WebAttack'])
        """
        self.class_names = class_names or ['Benign', 'BruteForce', 'WebAttack']
    
    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray = None,
        dataset_name: str = "validation"
    ) -> Dict[str, Any]:
        """
        Comprehensive evaluation on predicted labels.
        
        Parameters
        ----------
        y_true : np.ndarray, shape (n_samples,)
            Ground truth labels
        y_pred : np.ndarray, shape (n_samples,)
            Predicted labels
        y_proba : np.ndarray, shape (n_samples, n_classes), optional
            Prediction probabilities
        dataset_name : str
            Name of dataset (for logging)
        
        Returns
        -------
        metrics : Dict[str, Any]
            Dictionary containing all metrics
        """
        metrics = {}
        
        # Overall metrics
        metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
        metrics['precision_weighted'] = float(
            precision_score(y_true, y_pred, average='weighted', zero_division=0)
        )
        metrics['recall_weighted'] = float(
            recall_score(y_true, y_pred, average='weighted', zero_division=0)
        )
        metrics['f1_weighted'] = float(
            f1_score(y_true, y_pred, average='weighted', zero_division=0)
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Per-class metrics
        metrics['per_class'] = {}
        classes = np.unique(y_true)
        
        for cls in classes:
            class_name = self.class_names[cls] if cls < len(self.class_names) else f"Class_{cls}"
            y_true_binary = (y_true == cls).astype(int)
            y_pred_binary = (y_pred == cls).astype(int)
            
            precision = float(
                precision_score(y_true_binary, y_pred_binary, zero_division=0)
            )
            recall = float(
                recall_score(y_true_binary, y_pred_binary, zero_division=0)
            )
            f1 = float(
                f1_score(y_true_binary, y_pred_binary, zero_division=0)
            )
            support = int(np.sum(y_true == cls))
            
            metrics['per_class'][class_name] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'support': support
            }
        
        # Error analysis
        fp_mask = (y_true != y_pred)
        metrics['false_positive_count'] = int(np.sum((y_pred != y_true) & (y_true == 0)))
        metrics['false_negative_count'] = int(np.sum((y_pred != y_true) & (y_true != 0)))
        metrics['false_positive_rate'] = float(
            metrics['false_positive_count'] / max(np.sum(y_true == 0), 1)
        )
        metrics['false_negative_rate'] = float(
            metrics['false_negative_count'] / max(np.sum(y_true != 0), 1)
        )
        
        # ROC-AUC (if probabilities provided)
        if y_proba is not None:
            try:
                metrics['roc_auc'] = float(
                    roc_auc_score(y_true, y_proba, multi_class='ovr', average='weighted')
                )
            except Exception as e:
                logger.warning(f"ROC AUC computation failed: {e}")
                metrics['roc_auc'] = None
        
        logger.info(f"{dataset_name} evaluation complete: Accuracy={metrics['accuracy']:.4f}")
        
        return metrics
    
    def get_roc_curves(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Compute ROC curves for all classes.
        
        Parameters
        ----------
        y_true : np.ndarray
            Ground truth labels
        y_proba : np.ndarray, shape (n_samples, n_classes)
            Prediction probabilities
        
        Returns
        -------
        roc_data : Dict[str, Tuple]
            Dictionary with class names as keys, (fpr, tpr, thresholds) as values
        """
        roc_data = {}
        classes = np.unique(y_true)
        
        for cls in classes:
            class_name = self.class_names[cls] if cls < len(self.class_names) else f"Class_{cls}"
            y_true_binary = (y_true == cls).astype(int)
            y_score = y_proba[:, cls]
            
            fpr, tpr, thresholds = roc_curve(y_true_binary, y_score)
            roc_data[class_name] = (fpr, tpr, thresholds)
        
        return roc_data
    
    def get_precision_recall_curves(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Compute precision-recall curves for all classes.
        
        Parameters
        ----------
        y_true : np.ndarray
            Ground truth labels
        y_proba : np.ndarray, shape (n_samples, n_classes)
            Prediction probabilities
        
        Returns
        -------
        pr_data : Dict[str, Tuple]
            Dictionary with class names as keys, (precision, recall, thresholds) as values
        """
        pr_data = {}
        classes = np.unique(y_true)
        
        for cls in classes:
            class_name = self.class_names[cls] if cls < len(self.class_names) else f"Class_{cls}"
            y_true_binary = (y_true == cls).astype(int)
            y_score = y_proba[:, cls]
            
            precision, recall, thresholds = precision_recall_curve(y_true_binary, y_score)
            pr_data[class_name] = (precision, recall, thresholds)
        
        return pr_data
    
    def get_confidence_statistics(
        self,
        y_proba: np.ndarray,
        confidence_threshold: float = 0.95
    ) -> Dict[str, Any]:
        """
        Analyze confidence score distributions.
        
        Parameters
        ----------
        y_proba : np.ndarray, shape (n_samples, n_classes)
            Prediction probabilities
        confidence_threshold : float
            Threshold for confidence categorization
        
        Returns
        -------
        stats : Dict[str, Any]
            Confidence statistics including percentiles and categorization
        """
        confidence = np.max(y_proba, axis=1)
        
        stats = {
            'mean': float(np.mean(confidence)),
            'median': float(np.median(confidence)),
            'std': float(np.std(confidence)),
            'min': float(np.min(confidence)),
            'max': float(np.max(confidence)),
            'p25': float(np.percentile(confidence, 25)),
            'p50': float(np.percentile(confidence, 50)),
            'p75': float(np.percentile(confidence, 75)),
            'p95': float(np.percentile(confidence, 95)),
            'p99': float(np.percentile(confidence, 99)),
            'high_confidence_count': int(np.sum(confidence >= confidence_threshold)),
            'high_confidence_pct': float(
                100 * np.sum(confidence >= confidence_threshold) / len(confidence)
            ),
            'low_confidence_count': int(np.sum(confidence < 0.95)),
            'low_confidence_pct': float(
                100 * np.sum(confidence < 0.95) / len(confidence)
            )
        }
        
        return stats
    
    def get_classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> str:
        """
        Generate sklearn classification report.
        
        Parameters
        ----------
        y_true : np.ndarray
            Ground truth labels
        y_pred : np.ndarray
            Predicted labels
        
        Returns
        -------
        report : str
            Formatted classification report
        """
        return classification_report(
            y_true, y_pred,
            target_names=self.class_names,
            zero_division=0
        )