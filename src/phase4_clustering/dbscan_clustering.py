"""DBSCAN clustering utilities."""

import numpy as np
from sklearn.cluster import DBSCAN
import logging

logger = logging.getLogger(__name__)


class DBSCANClustering:
    """Perform DBSCAN clustering."""

    def __init__(self, eps: float = 0.5, min_samples: int = 5,
                 metric: str = 'euclidean', algorithm: str = 'auto', leaf_size: int = 30):
        self.dbscan = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            metric=metric,
            algorithm=algorithm,
            leaf_size=leaf_size
        )

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        labels = self.dbscan.fit_predict(X)
        logger.info(f"DBSCAN found {len(np.unique(labels))} clusters (including noise)")
        return labels