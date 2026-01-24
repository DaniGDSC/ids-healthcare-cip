"""Cluster analysis utilities."""

import numpy as np
import logging
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

logger = logging.getLogger(__name__)


class ClusterAnalyzer:
    """Analyze clustering results."""

    def __init__(self, compute_silhouette=True, compute_db=True, compute_ch=True):
        self.compute_silhouette = compute_silhouette
        self.compute_db = compute_db
        self.compute_ch = compute_ch

    def evaluate(self, X: np.ndarray, labels: np.ndarray):
        metrics = {}
        unique_labels = np.unique(labels)
        if len(unique_labels) <= 1:
            logger.warning("Cannot compute clustering metrics with <=1 cluster")
            return metrics

        if self.compute_silhouette:
            metrics['silhouette'] = float(silhouette_score(X, labels))
        if self.compute_db:
            metrics['davies_bouldin'] = float(davies_bouldin_score(X, labels))
        if self.compute_ch:
            metrics['calinski_harabasz'] = float(calinski_harabasz_score(X, labels))

        logger.info(f"Clustering metrics: {metrics}")
        return metrics