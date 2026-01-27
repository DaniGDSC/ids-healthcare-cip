"""Data splitting utilities."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class DataSplitter:
    """Split data into train/validation/test sets."""
    
    def __init__(self,
                 train_ratio: float = 0.6,
                 val_ratio: float = 0.2,
                 test_ratio: float = 0.2,
                 random_state: int = 42,
                 stratify: bool = True):
        """
        Initialize DataSplitter.
        
        Args:
            train_ratio: Proportion for training set
            val_ratio: Proportion for validation set
            test_ratio: Proportion for test set
            random_state: Random seed
            stratify: Whether to stratify split by labels
        """
        if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
            raise ValueError("Ratios must sum to 1.0")
        
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_state = random_state
        self.stratify = stratify
    
    def split(self, 
              X: np.ndarray, 
              y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, 
                                      np.ndarray, np.ndarray, np.ndarray]:
        """
        Split features and labels into train/val/test sets.
        
        Args:
            X: Feature matrix
            y: Label vector
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        logger.info(f"Splitting data: train={self.train_ratio}, val={self.val_ratio}, test={self.test_ratio}")
        
        stratify_labels = y if self.stratify else None
        
        # First split: separate test set
        test_size = self.test_ratio
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, 
            test_size=test_size,
            random_state=self.random_state,
            stratify=stratify_labels
        )
        
        # Second split: separate train and validation from remaining data
        val_size_adjusted = self.val_ratio / (self.train_ratio + self.val_ratio)
        stratify_temp = y_temp if self.stratify else None
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            random_state=self.random_state,
            stratify=stratify_temp
        )
        
        logger.info(f"Train set: {X_train.shape[0]} samples")
        logger.info(f"Validation set: {X_val.shape[0]} samples")
        logger.info(f"Test set: {X_test.shape[0]} samples")
        
        if self.stratify:
            self._log_label_distribution(y_train, y_val, y_test)
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def _log_label_distribution(self, y_train, y_val, y_test):
        """Log label distribution for each split."""
        logger.info("\nLabel distribution:")
        
        for name, labels in [('Train', y_train), ('Val', y_val), ('Test', y_test)]:
            unique, counts = np.unique(labels, return_counts=True)
            logger.info(f"\n{name}:")
            for label, count in zip(unique, counts):
                percentage = count / len(labels) * 100
                logger.info(f"  {label}: {count} ({percentage:.2f}%)")
    
    def get_split_indices(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get indices for train/val/test splits.
        
        Args:
            n_samples: Total number of samples
            
        Returns:
            Tuple of (train_indices, val_indices, test_indices)
        """
        indices = np.arange(n_samples)
        np.random.seed(self.random_state)
        np.random.shuffle(indices)
        
        train_end = int(n_samples * self.train_ratio)
        val_end = int(n_samples * (self.train_ratio + self.val_ratio))
        
        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]
        
        return train_indices, val_indices, test_indices
