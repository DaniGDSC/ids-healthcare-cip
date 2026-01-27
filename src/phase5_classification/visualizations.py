"""
Phase 5: Visualization Suite

Generates comprehensive plots for ensemble classification:
- Confusion matrices (heatmap)
- ROC curves (multi-class)
- Precision-Recall curves
- Feature importance (top 15)
- Confidence distribution histograms
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list,
    title: str = "Confusion Matrix",
    output_path: Path = None
) -> None:
    """
    Plot confusion matrix as heatmap.
    
    Parameters
    ----------
    cm : np.ndarray, shape (n_classes, n_classes)
        Confusion matrix
    class_names : list
        Class names
    title : str
        Plot title
    output_path : Path, optional
        Path to save figure
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'}
    )
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if output_path:
        output_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path / f"{title.lower().replace(' ', '_')}.png", dpi=150)
        logger.info(f"Saved: {output_path / f'{title.lower().replace(' ', '_')}.png'}")
    
    plt.close()


def plot_roc_curves(
    roc_data: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    output_path: Path = None
) -> None:
    """
    Plot ROC curves for all classes.
    
    Parameters
    ----------
    roc_data : Dict[str, Tuple]
        Dictionary with class names as keys, (fpr, tpr, thresholds) as values
    output_path : Path, optional
        Path to save figure
    """
    from sklearn.metrics import auc
    
    plt.figure(figsize=(10, 8))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for idx, (class_name, (fpr, tpr, _)) in enumerate(roc_data.items()):
        roc_auc = auc(fpr, tpr)
        plt.plot(
            fpr, tpr,
            color=colors[idx % len(colors)],
            lw=2,
            label=f'{class_name} (AUC={roc_auc:.3f})'
        )
    
    # Plot random classifier
    plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Random Classifier')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves (One-vs-Rest)', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if output_path:
        output_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path / 'roc_curves.png', dpi=150)
        logger.info(f"Saved: {output_path / 'roc_curves.png'}")
    
    plt.close()


def plot_precision_recall_curves(
    pr_data: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    output_path: Path = None
) -> None:
    """
    Plot precision-recall curves for all classes.
    
    Parameters
    ----------
    pr_data : Dict[str, Tuple]
        Dictionary with class names as keys, (precision, recall, thresholds) as values
    output_path : Path, optional
        Path to save figure
    """
    from sklearn.metrics import auc
    
    plt.figure(figsize=(10, 8))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for idx, (class_name, (precision, recall, _)) in enumerate(pr_data.items()):
        pr_auc = auc(recall, precision)
        plt.plot(
            recall, precision,
            color=colors[idx % len(colors)],
            lw=2,
            label=f'{class_name} (AP={pr_auc:.3f})'
        )
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curves', fontsize=14, fontweight='bold')
    plt.legend(loc="upper right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if output_path:
        output_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path / 'pr_curves.png', dpi=150)
        logger.info(f"Saved: {output_path / 'pr_curves.png'}")
    
    plt.close()


def plot_feature_importance(
    importances: Dict[str, float],
    top_n: int = 15,
    output_path: Path = None
) -> None:
    """
    Plot top N feature importances.
    
    Parameters
    ----------
    importances : Dict[str, float]
        Dictionary with feature names as keys, importance values as values
    top_n : int
        Number of top features to display
    output_path : Path, optional
        Path to save figure
    """
    # Sort and take top N
    sorted_items = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:top_n]
    names, values = zip(*sorted_items)
    
    plt.figure(figsize=(12, 6))
    bars = plt.barh(names, values, color='#1f77b4', edgecolor='black', linewidth=0.5)
    
    # Color latent features differently
    for idx, (bar, name) in enumerate(zip(bars, names)):
        if 'Latent' in name:
            bar.set_color('#ff7f0e')
    
    plt.xlabel('Importance Score', fontsize=12)
    plt.title(f'Top {top_n} Feature Importance (Decision Tree)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        output_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path / 'feature_importance.png', dpi=150, bbox_inches='tight')
        logger.info(f"Saved: {output_path / 'feature_importance.png'}")
    
    plt.close()


def plot_confidence_distribution(
    y_proba: np.ndarray,
    output_path: Path = None
) -> None:
    """
    Plot confidence score distribution histogram.
    
    Parameters
    ----------
    y_proba : np.ndarray, shape (n_samples, n_classes)
        Prediction probabilities
    output_path : Path, optional
        Path to save figure
    """
    confidence = np.max(y_proba, axis=1)
    
    plt.figure(figsize=(10, 6))
    
    plt.hist(confidence, bins=50, color='#1f77b4', edgecolor='black', alpha=0.7)
    
    # Add percentile lines
    p95 = np.percentile(confidence, 95)
    p99 = np.percentile(confidence, 99)
    p50 = np.median(confidence)
    
    plt.axvline(p50, color='green', linestyle='--', linewidth=2, label=f'Median: {p50:.3f}')
    plt.axvline(p95, color='orange', linestyle='--', linewidth=2, label=f'P95: {p95:.3f}')
    plt.axvline(p99, color='red', linestyle='--', linewidth=2, label=f'P99: {p99:.3f}')
    
    plt.xlabel('Confidence Score', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Prediction Confidence Distribution (Test Set)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    
    if output_path:
        output_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path / 'confidence_distribution.png', dpi=150)
        logger.info(f"Saved: {output_path / 'confidence_distribution.png'}")
    
    plt.close()


def generate_all_visualizations(
    results: Dict,
    feature_names: list,
    output_path: Path
) -> None:
    """
    Generate all visualization plots.
    
    Parameters
    ----------
    results : Dict
        Evaluation results dictionary
    feature_names : list
        Feature names
    output_path : Path
        Output directory for figures
    """
    logger.info("=" * 80)
    logger.info("GENERATING VISUALIZATIONS")
    logger.info("=" * 80)
    
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Confusion matrices
    logger.info("Plotting confusion matrices...")
    plot_confusion_matrix(
        np.array(results['validation']['confusion_matrix']),
        ['Benign', 'BruteForce', 'WebAttack'],
        title="Validation Confusion Matrix",
        output_path=output_path
    )
    
    plot_confusion_matrix(
        np.array(results['test']['confusion_matrix']),
        ['Benign', 'BruteForce', 'WebAttack'],
        title="Test Confusion Matrix",
        output_path=output_path
    )
    
    # ROC curves
    logger.info("Plotting ROC curves...")
    plot_roc_curves(results['roc_curves'], output_path)
    
    # PR curves
    logger.info("Plotting precision-recall curves...")
    plot_precision_recall_curves(results['pr_curves'], output_path)
    
    # Feature importance
    logger.info("Plotting feature importance...")
    plot_feature_importance(results['feature_importance'], top_n=15, output_path=output_path)
    
    # Confidence distribution
    logger.info("Plotting confidence distribution...")
    plot_confidence_distribution(
        results['predictions']['test_proba'],
        output_path
    )
    
    logger.info("All visualizations complete!")
