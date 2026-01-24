"""Information Gain feature selection."""

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)


class InformationGainSelector:
    """Select features using Information Gain (mutual information)."""
    
    def __init__(self, n_features: int = 35, discretization_bins: int = 10):
        """
        Initialize Information Gain selector.
        
        Args:
            n_features: Number of features to select
            discretization_bins: Number of bins for discretization
        """
        self.n_features = n_features
        self.discretization_bins = discretization_bins
        self.scores_ = None
        self.selected_features_ = None
    
    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: List[str] = None):
        """
        Calculate information gain scores.
        
        Args:
            X: Feature matrix
            y: Target labels
            feature_names: Optional feature names
        """
        logger.info(f"Calculating information gain scores for {X.shape[1]} features...")
        
        # Calculate mutual information scores
        self.scores_ = mutual_info_classif(
            X, y,
            discrete_features=False,
            n_neighbors=3,
            random_state=42
        )
        
        # Get top features
        top_indices = np.argsort(self.scores_)[::-1][:self.n_features]
        self.selected_features_ = top_indices
        
        logger.info(f"Selected top {self.n_features} features based on information gain")
        
        if feature_names is not None:
            selected_names = [feature_names[i] for i in top_indices]
            logger.info(f"\nTop 10 features by information gain:")
            for i, (name, score) in enumerate(zip(selected_names[:10], self.scores_[top_indices][:10])):
                logger.info(f"  {i+1}. {name}: {score:.4f}")
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data to selected features.
        
        Args:
            X: Feature matrix
            
        Returns:
            Transformed feature matrix
        """
        if self.selected_features_ is None:
            raise ValueError("Selector not fitted. Call fit() first.")
        
        return X[:, self.selected_features_]
    
    def fit_transform(self, X: np.ndarray, y: np.ndarray, feature_names: List[str] = None) -> np.ndarray:
        """
        Fit and transform in one step.
        
        Args:
            X: Feature matrix
            y: Target labels
            feature_names: Optional feature names
            
        Returns:
            Transformed feature matrix
        """
        self.fit(X, y, feature_names)
        return self.transform(X)
    
    def get_feature_scores(self) -> np.ndarray:
        """Get feature importance scores."""
        return self.scores_
    
    def get_selected_indices(self) -> np.ndarray:
        """Get indices of selected features."""
        return self.selected_features_
