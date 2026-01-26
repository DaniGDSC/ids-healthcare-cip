"""Recursive Feature Elimination (RFE) selector."""

import numpy as np
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from typing import List
import logging

logger = logging.getLogger(__name__)


class RFESelector:
    """Select features using Recursive Feature Elimination."""
    
    def __init__(self,
                 n_features: int = 35,
                 estimator: str = 'linear_svc',
                 step: int = 1,
                 cv: int = 5,
                 random_state: int = 42,
                 greedy: bool = False,
                 greedy_batch_pct: float = 0.05):
        """
        Initialize RFE selector.
        
        Args:
            n_features: Number of features to select
            estimator: Base estimator (linear_svc/random_forest/logistic)
            step: Number of features to remove at each iteration (standard RFE)
            cv: Cross-validation folds
            random_state: Random seed
            greedy: Use greedy batch elimination instead of exhaustive RFE
            greedy_batch_pct: Percentage of features to remove per iteration (0.05 = 5%)
        """
        self.n_features = n_features
        self.step = step
        self.cv = cv
        self.greedy = greedy
        self.greedy_batch_pct = greedy_batch_pct
        
        # Select base estimator
        if estimator == 'random_forest':
            self.estimator = RandomForestClassifier(
                n_estimators=50,
                max_depth=10,
                random_state=random_state,
                n_jobs=-1
            )
        elif estimator == 'svm' or estimator == 'linear_svc':
            self.estimator = SVC(kernel='linear', random_state=random_state)
        elif estimator == 'logistic':
            self.estimator = LogisticRegression(max_iter=1000, random_state=random_state)
        else:
            raise ValueError(f"Unknown estimator: {estimator}")
        
        if not greedy:
            self.rfe = RFE(
                estimator=self.estimator,
                n_features_to_select=n_features,
                step=step
            )
        else:
            self.rfe = None
        
        self.ranking_ = None
        self.selected_features_ = None
    
    def _fit_greedy(self, X: np.ndarray, y: np.ndarray, feature_names: List[str] = None):
        """
        Greedy backward elimination: remove bottom batch_pct% features per iteration.
        Much faster than exhaustive RFE (removes ~5% per iteration vs 1 feature).
        
        Args:
            X: Feature matrix
            y: Target labels
            feature_names: Optional feature names
        """
        logger.info(f"Performing GREEDY RFE with {X.shape[1]} features...")
        logger.info(f"  Target: {self.n_features} features")
        logger.info(f"  Batch elimination: {self.greedy_batch_pct*100:.1f}% per iteration")
        
        active_features = np.arange(X.shape[1])
        iteration = 0
        
        while len(active_features) > self.n_features:
            iteration += 1
            n_current = len(active_features)
            
            # Train estimator on current feature set
            X_current = X[:, active_features]
            self.estimator.fit(X_current, y)
            
            # Get feature importances
            if hasattr(self.estimator, 'feature_importances_'):
                importances = self.estimator.feature_importances_
            elif hasattr(self.estimator, 'coef_'):
                importances = np.abs(self.estimator.coef_[0])
            else:
                importances = np.ones(n_current)
            
            # Remove bottom batch_pct% features
            batch_size = max(1, int(n_current * self.greedy_batch_pct))
            bottom_indices = np.argsort(importances)[:batch_size]
            
            # Keep top features
            keep_mask = np.ones(n_current, dtype=bool)
            keep_mask[bottom_indices] = False
            active_features = active_features[keep_mask]
            
            logger.info(f"  Iteration {iteration}: {n_current} → {len(active_features)} features (removed {batch_size})")
        
        self.selected_features_ = active_features
        logger.info(f"Greedy RFE complete. Selected {len(self.selected_features_)} features")
        
        if feature_names is not None:
            selected_names = [feature_names[i] for i in self.selected_features_]
            logger.info(f"\nTop 10 selected features by Greedy RFE:")
            for i, name in enumerate(selected_names[:10]):
                logger.info(f"  {i+1}. {name}")
    
    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: List[str] = None):
        """
        Perform RFE to select features.
        
        Args:
            X: Feature matrix
            y: Target labels
            feature_names: Optional feature names
        """
        if self.greedy:
            self._fit_greedy(X, y, feature_names)
        else:
            logger.info(f"Performing RFE with {X.shape[1]} features...")
            logger.info(f"  Target: {self.n_features} features")
            logger.info(f"  Step size: {self.step}")
            
            self.rfe.fit(X, y)
            
            self.ranking_ = self.rfe.ranking_
            self.selected_features_ = np.where(self.rfe.support_)[0]
            
            logger.info(f"RFE complete. Selected {len(self.selected_features_)} features")
            
            if feature_names is not None:
                selected_names = [feature_names[i] for i in self.selected_features_]
                logger.info(f"\nTop 10 selected features by RFE:")
                for i, name in enumerate(selected_names[:10]):
                    logger.info(f"  {i+1}. {name}")
    
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
    
    def get_feature_ranking(self) -> np.ndarray:
        """Get feature ranking (1 = selected)."""
        return self.ranking_
    
    def get_selected_indices(self) -> np.ndarray:
        """Get indices of selected features."""
        return self.selected_features_
