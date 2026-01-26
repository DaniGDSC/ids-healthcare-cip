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

    def cluster_stats(self, X: np.ndarray, labels: np.ndarray, y_true: np.ndarray):
        """Compute cluster-level stats: size, purity, dominant label, centroids."""
        stats = []
        centroids = []
        unique_labels = np.unique(labels)
        for c in unique_labels:
            mask = labels == c
            cluster_points = X[mask]
            cluster_y = y_true[mask] if y_true is not None else None
            size = int(cluster_points.shape[0])
            if cluster_y is not None and size > 0:
                vals, counts = np.unique(cluster_y, return_counts=True)
                idx = int(np.argmax(counts))
                dominant_label = int(vals[idx])
                purity = float(counts[idx] / size)
            else:
                dominant_label = None
                purity = 0.0
            centroid = cluster_points.mean(axis=0) if size > 0 else np.zeros(X.shape[1])
            centroids.append(centroid)
            stats.append({
                'cluster': int(c),
                'size': size,
                'purity': purity,
                'dominant_label': dominant_label
            })
        return stats, np.vstack(centroids) if len(centroids) > 0 else np.empty((0, X.shape[1]))