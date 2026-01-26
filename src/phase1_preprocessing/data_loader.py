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
    
    # Allowed file extensions for security
    ALLOWED_EXTENSIONS = {'.csv', '.CSV'}
    MAX_FILENAME_LENGTH = 255
    
    def __init__(self, data_dir: str):
        """
        Initialize DataLoader.
        
        Args:
            data_dir: Directory containing CIC-IDS-2018 CSV files
            
        Raises:
            FileNotFoundError: If data directory does not exist
            ValueError: If data_dir contains path traversal sequences
        """
        self.data_dir = self._validate_and_resolve_path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        if not self.data_dir.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {data_dir}")
    
    @staticmethod
    def _validate_and_resolve_path(path_str: str) -> Path:
        """
        Validate and resolve a path to prevent traversal attacks.
        
        Args:
            path_str: Path string to validate
            
        Returns:
            Resolved Path object
            
        Raises:
            ValueError: If path contains traversal sequences or is invalid
        """
        if not path_str or not isinstance(path_str, str):
            raise ValueError("Path must be a non-empty string")
        
        # Check for path traversal sequences
        if ".." in path_str or "~" in path_str:
            raise ValueError("Path traversal sequences (.., ~) are not allowed")
        
        path = Path(path_str).resolve()
        
        # Verify resolved path is not a symlink to prevent symlink attacks
        if path.is_symlink():
            raise ValueError("Symbolic links are not allowed for security reasons")
        
        return path
    
    @staticmethod
    def _validate_filename(filename: str) -> Path:
        """
        Validate filename for security.
        
        Args:
            filename: Filename to validate
            
        Returns:
            Validated Path object
            
        Raises:
            ValueError: If filename is invalid or contains disallowed characters
        """
        if not filename or not isinstance(filename, str):
            raise ValueError("Filename must be a non-empty string")
        
        if len(filename) > DataLoader.MAX_FILENAME_LENGTH:
            raise ValueError(f"Filename exceeds maximum length of {DataLoader.MAX_FILENAME_LENGTH}")
        
        # Prevent absolute paths and traversal
        if filename.startswith('/') or filename.startswith('\\'):
            raise ValueError("Absolute paths are not allowed")
        
        if ".." in filename or "~" in filename:
            raise ValueError("Path traversal sequences (.., ~) are not allowed")
        
        # Prevent null bytes
        if '\x00' in filename:
            raise ValueError("Null bytes in filename are not allowed")
        
        path = Path(filename)
        
        # Ensure no path components
        if len(path.parts) > 1:
            raise ValueError("Subdirectories in filename are not allowed")
        
        return path
    
    def load_csv_files(self, pattern: str = "*.csv") -> pd.DataFrame:
        """
        Load all CSV files matching pattern.
        
        Args:
            pattern: Glob pattern for CSV files (must be simple, no path traversal)
            
        Returns:
            Combined DataFrame
            
        Raises:
            ValueError: If pattern contains disallowed sequences
            FileNotFoundError: If no CSV files found
        """
        # Validate pattern to prevent glob attacks
        if ".." in pattern or "/" in pattern or "\\" in pattern:
            raise ValueError("Pattern cannot contain path traversal sequences or path separators")
        
        csv_files = list(self.data_dir.glob(pattern))
        
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {self.data_dir}")
        
        logger.info(f"Found {len(csv_files)} CSV files")
        
        dfs = []
        for csv_file in csv_files:
            # Verify file is within data_dir and not a symlink
            try:
                csv_file_resolved = csv_file.resolve()
                data_dir_resolved = self.data_dir.resolve()
                
                # Ensure file is within the data directory
                csv_file_resolved.relative_to(data_dir_resolved)
                
                if csv_file_resolved.is_symlink():
                    logger.warning(f"Skipping symlink file: {csv_file.name}")
                    continue
                    
            except ValueError:
                logger.warning(f"Skipping file outside data directory: {csv_file}")
                continue
            
            logger.info(f"Loading {csv_file.name}...")
            try:
                df = pd.read_csv(csv_file, encoding='utf-8', low_memory=False)
                dfs.append(df)
                logger.info(f"  Loaded {len(df)} rows")
            except pd.errors.ParserError as e:
                logger.error(f"CSV parsing error in {csv_file}: {e}")
                raise ValueError(f"Invalid CSV file format: {csv_file.name}") from e
            except Exception as e:
                logger.error(f"Error loading {csv_file}: {e}")
                raise
        
        combined_df = pd.concat(dfs, ignore_index=True)
        logger.info(f"Total rows: {len(combined_df)}")
        logger.info(f"Total columns: {len(combined_df.columns)}")
        
        return combined_df
    
    def load_single_file(self, filename: str) -> pd.DataFrame:
        """
        Load a single CSV file with path validation.
        
        Args:
            filename: Name of CSV file (without directory path)
            
        Returns:
            DataFrame
            
        Raises:
            ValueError: If filename is invalid or contains traversal sequences
            FileNotFoundError: If file does not exist
        """
        # Validate filename
        validated_filename = self._validate_filename(filename)
        
        filepath = self.data_dir / validated_filename
        
        # Verify resolved path is within data_dir
        try:
            filepath_resolved = filepath.resolve()
            data_dir_resolved = self.data_dir.resolve()
            filepath_resolved.relative_to(data_dir_resolved)
        except ValueError:
            raise ValueError(f"Path traversal detected: {filename}")
        
        if not filepath_resolved.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        if filepath_resolved.is_symlink():
            raise ValueError(f"Symbolic links are not allowed: {filename}")
        
        if not filepath_resolved.is_file():
            raise ValueError(f"Path is not a file: {filename}")
        
        logger.info(f"Loading {filename}...")
        try:
            df = pd.read_csv(filepath, encoding='utf-8', low_memory=False)
            logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
            return df
        except pd.errors.ParserError as e:
            logger.error(f"CSV parsing error: {e}")
            raise ValueError(f"Invalid CSV file format: {filename}") from e
    
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
