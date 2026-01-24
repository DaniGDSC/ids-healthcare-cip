"""Data normalization utilities."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import joblib
from pathlib import Path
from typing import Union, Tuple
import logging

logger = logging.getLogger(__name__)


class Normalizer:
    """Normalize numerical features."""
    
    def __init__(self, method: str = 'standard'):
        """
        Initialize Normalizer.
        
        Args:
            method: Normalization method (standard/minmax/robust)
        """
        self.method = method
        self.scaler = None
        
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        elif method == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    def fit(self, X: np.ndarray):
        """
        Fit the scaler on training data.
        
        Args:
            X: Training data
        """
        logger.info(f"Fitting {self.method} scaler...")
        self.scaler.fit(X)
        logger.info("Scaler fitted successfully")
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data using fitted scaler.
        
        Args:
            X: Data to transform
            
        Returns:
            Normalized data
        """
        if self.scaler is None:
            raise ValueError("Scaler not fitted. Call fit() first.")
        
        logger.info(f"Transforming data with {self.method} scaler...")
        X_normalized = self.scaler.transform(X)
        
        return X_normalized
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit scaler and transform data.
        
        Args:
            X: Data to fit and transform
            
        Returns:
            Normalized data
        """
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Reverse normalization.
        
        Args:
            X: Normalized data
            
        Returns:
            Original scale data
        """
        if self.scaler is None:
            raise ValueError("Scaler not fitted")
        
        return self.scaler.inverse_transform(X)
    
    def save(self, filepath: Union[str, Path]):
        """
        Save fitted scaler to disk.
        
        Args:
            filepath: Path to save scaler
        """
        if self.scaler is None:
            raise ValueError("No scaler to save")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self.scaler, filepath)
        logger.info(f"Scaler saved to {filepath}")
    
    def load(self, filepath: Union[str, Path]):
        """
        Load fitted scaler from disk.
        
        Args:
            filepath: Path to load scaler from
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Scaler file not found: {filepath}")
        
        self.scaler = joblib.load(filepath)
        logger.info(f"Scaler loaded from {filepath}")
    
    def get_statistics(self) -> dict:
        """
        Get scaler statistics.
        
        Returns:
            Dictionary with scaler statistics
        """
        if self.scaler is None:
            return {}
        
        stats = {'method': self.method}
        
        if hasattr(self.scaler, 'mean_'):
            stats['mean'] = self.scaler.mean_
        if hasattr(self.scaler, 'scale_'):
            stats['scale'] = self.scaler.scale_
        if hasattr(self.scaler, 'data_min_'):
            stats['data_min'] = self.scaler.data_min_
        if hasattr(self.scaler, 'data_max_'):
            stats['data_max'] = self.scaler.data_max_
        
        return stats
