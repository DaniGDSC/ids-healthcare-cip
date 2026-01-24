"""Random Forest feature selection."""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from typing import List
import logging

logger = logging.getLogger(__name__)


class RandomForestSelector:
    """Select features using Random Forest feature importance."""
    
    def __init__(self, 
                 n_features: int = 35,
                 n_estimators: int = 100,
                 max_depth: int = 10,
                 min_samples_split: int = 10,
                 random_state: int = 42):
        """
        Initialize Random Forest selector.
        
        Args:
            n_features: Number of features to select
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of trees
            min_samples_split: Minimum samples required to split
            random_state: Random seed
        """
        self.n_features = n_features
        self.rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state,
            n_jobs=-1
        )
        self.feature_importances_ = None
        self.selected_features_ = None
    
    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: List[str] = None):
        """
        Train Random Forest and calculate feature importances.
        
        Args:
            X: Feature matrix
            y: Target labels
            feature_names: Optional feature names
        """
        logger.info(f"Training Random Forest with {X.shape[1]} features...")
        
        self.rf.fit(X, y)
        self.feature_importances_ = self.rf.feature_importances_
        
        # Get top features
        top_indices = np.argsort(self.feature_importances_)[::-1][:self.n_features]
        self.selected_features_ = top_indices
        
        logger.info(f"Selected top {self.n_features} features based on Random Forest importance")
        
        if feature_names is not None:
            selected_names = [feature_names[i] for i in top_indices]
            logger.info(f"\nTop 10 features by Random Forest importance:")
            for i, (name, score) in enumerate(zip(selected_names[:10], self.feature_importances_[top_indices][:10])):
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
    
    def get_feature_importances(self) -> np.ndarray:
        """Get feature importance scores."""
        return self.feature_importances_
    
    def get_selected_indices(self) -> np.ndarray:
        """Get indices of selected features."""
        return self.selected_features_
