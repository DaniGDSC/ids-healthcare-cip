"""Extremely Randomized Trees (ExtraTrees) feature selection."""

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from typing import List
import logging

logger = logging.getLogger(__name__)


class ExtraTreesSelector:
    """Select features using Extremely Randomized Trees (faster than RF)."""
    
    def __init__(self, 
                 n_features: int = 35,
                 n_estimators: int = 100,
                 max_depth: int = 10,
                 min_samples_split: int = 10,
                 random_state: int = 42,
                 use_sampling: bool = True,
                 sample_size: float = 0.15,
                 exploratory: bool = False):
        """
        Initialize ExtraTrees selector (faster than RandomForest).
        
        Args:
            n_features: Number of features to select
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of trees
            min_samples_split: Minimum samples required to split
            random_state: Random seed
            use_sampling: Train on stratified sample instead of full data
            sample_size: Fraction of data to use for training (0.15 = 15%)
            exploratory: If True, use 20 estimators for fast exploration
        
        Advantages over RandomForest:
        - Faster: splits are random instead of exhaustive search
        - Lower bias: randomness reduces variance at cost of slightly higher bias
        - Good for exploratory phase or large datasets
        """
        self.n_features = n_features
        self.use_sampling = use_sampling
        self.sample_size = sample_size
        self.exploratory = exploratory
        
        actual_n_estimators = 20 if exploratory else n_estimators
        
        self.et = ExtraTreesClassifier(
            n_estimators=actual_n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state,
            n_jobs=-1
        )
        self.feature_importances_ = None
        self.selected_features_ = None
    
    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: List[str] = None):
        """
        Train ExtraTrees and calculate feature importances.
        
        Args:
            X: Feature matrix
            y: Target labels
            feature_names: Optional feature names
        """
        X_train = X
        y_train = y
        
        if self.use_sampling:
            logger.info(f"Using sampling: {int(self.sample_size*100)}% of {X.shape[0]} samples")
            X_train, _, y_train, _ = train_test_split(
                X, y,
                train_size=self.sample_size,
                stratify=y,
                random_state=42
            )
            logger.info(f"  Sample size: {X_train.shape[0]} samples, {X_train.shape[1]} features")
        
        logger.info(f"Training ExtraTrees with {X_train.shape[1]} features...")
        
        self.et.fit(X_train, y_train)
        self.feature_importances_ = self.et.feature_importances_
        
        top_indices = np.argsort(self.feature_importances_)[::-1][:self.n_features]
        self.selected_features_ = top_indices
        
        logger.info(f"Selected top {self.n_features} features based on ExtraTrees importance")
        
        if feature_names is not None:
            selected_names = [feature_names[i] for i in top_indices]
            logger.info(f"\nTop 10 features by ExtraTrees importance:")
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
