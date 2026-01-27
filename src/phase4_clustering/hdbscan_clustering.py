"""HDBSCAN clustering with density-based robust cluster detection."""

import numpy as np
from sklearn.preprocessing import StandardScaler
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    logger.warning("hdbscan not installed; install via: pip install hdbscan")


class HDBSCANClustering:
    """Perform HDBSCAN clustering (hierarchical density-based)."""

    def __init__(self, 
                 min_cluster_size: int = 50,
                 min_samples: int = 10,
                 metric: str = 'euclidean',
                 cluster_selection_epsilon: float = 0.0,
                 algorithm: str = 'best'):
        """
        Initialize HDBSCAN clustering.
        
        Args:
            min_cluster_size: Minimum cluster size (smaller = more clusters).
            min_samples: Minimum samples in a neighborhood (robustness parameter).
            metric: Distance metric (euclidean, manhattan, cosine, etc).
            cluster_selection_epsilon: Epsilon for cluster extraction from hierarchy.
            algorithm: 'best', 'generic', 'prims_kdtree', 'prims_balltree'.
        """
        if not HDBSCAN_AVAILABLE:
            raise ImportError("hdbscan is required; install with: pip install hdbscan")
        
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.metric = metric
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.algorithm = algorithm
        self.hdbscan_model = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric=metric,
            cluster_selection_epsilon=cluster_selection_epsilon,
            algorithm=algorithm
        )
        self.labels_ = None
        self.probabilities_ = None
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Fit HDBSCAN and return cluster labels.
        
        Args:
            X: Feature matrix.
            
        Returns:
            Cluster labels (-1 = noise/outlier).
        """
        self.labels_ = self.hdbscan_model.fit_predict(X)
        self.probabilities_ = self.hdbscan_model.probabilities_
        
        n_clusters = len(set(self.labels_)) - (1 if -1 in self.labels_ else 0)
        n_noise = list(self.labels_).count(-1)
        logger.info(f"HDBSCAN: {n_clusters} clusters, {n_noise} noise points")
        
        return self.labels_
    
    def get_probabilities(self) -> np.ndarray:
        """Get cluster membership probabilities."""
        if self.probabilities_ is None:
            raise ValueError("Model not fitted. Call fit_predict() first.")
        return self.probabilities_
