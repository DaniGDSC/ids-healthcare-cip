"""Data cleaning utilities."""

import pandas as pd
import numpy as np
import logging
from typing import Optional, List, Tuple

logger = logging.getLogger(__name__)


class DataCleaner:
    """Clean and preprocess raw data."""
    
    def __init__(self, 
                 drop_na_threshold: float = 0.5,
                 fill_strategy: str = 'median',
                 replace_inf: bool = True):
        """
        Initialize DataCleaner.
        
        Args:
            drop_na_threshold: Drop columns with missing rate > threshold
            fill_strategy: Strategy for filling missing values (mean/median/mode/zero)
            replace_inf: Whether to replace infinite values
        """
        self.drop_na_threshold = drop_na_threshold
        self.fill_strategy = fill_strategy
        self.replace_inf = replace_inf
    
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all cleaning steps.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        logger.info("Starting data cleaning...")
        
        df = df.copy()
        
        # Remove duplicates
        df = self._remove_duplicates(df)
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Handle infinite values
        if self.replace_inf:
            df = self._handle_infinite_values(df)
        
        # Remove constant features
        df = self._remove_constant_features(df)
        
        logger.info(f"Cleaning complete. Final shape: {df.shape}")
        
        return df
    
    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows."""
        initial_rows = len(df)
        df = df.drop_duplicates()
        removed = initial_rows - len(df)
        
        if removed > 0:
            logger.info(f"Removed {removed} duplicate rows")
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values."""
        # Drop columns with too many missing values
        missing_ratio = df.isnull().sum() / len(df)
        cols_to_drop = missing_ratio[missing_ratio > self.drop_na_threshold].index.tolist()
        
        if cols_to_drop:
            logger.info(f"Dropping {len(cols_to_drop)} columns with >{self.drop_na_threshold*100}% missing")
            df = df.drop(columns=cols_to_drop)
        
        # Fill remaining missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if self.fill_strategy == 'mean':
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        elif self.fill_strategy == 'median':
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        elif self.fill_strategy == 'zero':
            df[numeric_cols] = df[numeric_cols].fillna(0)
        
        # Drop any remaining rows with missing values
        rows_before = len(df)
        df = df.dropna()
        rows_dropped = rows_before - len(df)
        
        if rows_dropped > 0:
            logger.info(f"Dropped {rows_dropped} rows with remaining missing values")
        
        return df
    
    def _handle_infinite_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Replace infinite values."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        inf_count = np.isinf(df[numeric_cols]).sum().sum()
        
        if inf_count > 0:
            logger.info(f"Replacing {inf_count} infinite values with 0")
            df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], 0)
        
        return df
    
    def _remove_constant_features(self, df: pd.DataFrame, threshold: float = 0.01) -> pd.DataFrame:
        """Remove features with near-zero variance."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Calculate variance
        variances = df[numeric_cols].var()
        constant_cols = variances[variances < threshold].index.tolist()
        
        if constant_cols:
            logger.info(f"Removing {len(constant_cols)} near-constant features")
            df = df.drop(columns=constant_cols)
        
        return df
    
    def get_data_summary(self, df: pd.DataFrame) -> dict:
        """
        Get summary statistics of the data.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with summary statistics
        """
        summary = {
            'n_rows': len(df),
            'n_columns': len(df.columns),
            'missing_values': df.isnull().sum().sum(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
            'dtypes': df.dtypes.value_counts().to_dict()
        }
        
        return summary
