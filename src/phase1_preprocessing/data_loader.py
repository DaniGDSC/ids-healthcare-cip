"""Data loader for CIC-IDS-2018 dataset."""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    """Load and combine CIC-IDS-2018 CSV files."""
    
    def __init__(self, data_dir: str):
        """
        Initialize DataLoader.
        
        Args:
            data_dir: Directory containing CIC-IDS-2018 CSV files
        """
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    def load_csv_files(self, pattern: str = "*.csv") -> pd.DataFrame:
        """
        Load all CSV files matching pattern.
        
        Args:
            pattern: Glob pattern for CSV files
            
        Returns:
            Combined DataFrame
        """
        csv_files = list(self.data_dir.glob(pattern))
        
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {self.data_dir}")
        
        logger.info(f"Found {len(csv_files)} CSV files")
        
        dfs = []
        for csv_file in csv_files:
            logger.info(f"Loading {csv_file.name}...")
            try:
                df = pd.read_csv(csv_file, encoding='utf-8', low_memory=False)
                dfs.append(df)
                logger.info(f"  Loaded {len(df)} rows")
            except Exception as e:
                logger.error(f"Error loading {csv_file}: {e}")
                raise
        
        combined_df = pd.concat(dfs, ignore_index=True)
        logger.info(f"Total rows: {len(combined_df)}")
        logger.info(f"Total columns: {len(combined_df.columns)}")
        
        return combined_df
    
    def load_single_file(self, filename: str) -> pd.DataFrame:
        """
        Load a single CSV file.
        
        Args:
            filename: Name of CSV file
            
        Returns:
            DataFrame
        """
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        logger.info(f"Loading {filename}...")
        df = pd.read_csv(filepath, encoding='utf-8', low_memory=False)
        logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
        
        return df
    
    def get_label_distribution(self, df: pd.DataFrame, label_col: str = 'Label') -> pd.Series:
        """
        Get distribution of labels.
        
        Args:
            df: Input DataFrame
            label_col: Name of label column
            
        Returns:
            Series with label counts
        """
        if label_col not in df.columns:
            raise ValueError(f"Label column '{label_col}' not found")
        
        distribution = df[label_col].value_counts()
        logger.info(f"\nLabel distribution:\n{distribution}")
        
        return distribution
