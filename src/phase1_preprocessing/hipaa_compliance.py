"""HIPAA compliance utilities."""

import hashlib
import logging
from typing import List, Optional
import pandas as pd
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)


class HIPAACompliance:
    """Ensure HIPAA compliance for healthcare data."""
    
    def __init__(self, enabled: bool = True):
        """
        Initialize HIPAA compliance checker.
        
        Args:
            enabled: Whether HIPAA compliance is enabled
        """
        self.enabled = enabled
        self.access_log = []
    
    def anonymize_ip_addresses(self, df: pd.DataFrame, ip_columns: List[str]) -> pd.DataFrame:
        """
        Anonymize IP addresses using hashing.
        
        Args:
            df: Input DataFrame
            ip_columns: List of columns containing IP addresses
            
        Returns:
            DataFrame with anonymized IPs
        """
        if not self.enabled:
            return df
        
        logger.info("Anonymizing IP addresses for HIPAA compliance...")
        
        df = df.copy()
        
        for col in ip_columns:
            if col in df.columns:
                df[col] = self._hash_series(df[col])
                logger.info(f"  Anonymized column: {col}")
        
        return df
    
    def _hash_value(self, value) -> str:
        """
        Hash a value using SHA-256.
        
        Args:
            value: Value to hash
            
        Returns:
            Hashed value
        """
        if pd.isna(value):
            return value
        
        # Convert to string and hash
        hash_object = hashlib.sha256(str(value).encode())
        return hash_object.hexdigest()[:16]  # Return first 16 chars

    def _hash_series(self, series: pd.Series) -> pd.Series:
        """Vectorized hashing with de-duplication to avoid per-row Python calls."""
        if series.empty:
            return series

        mask = series.notna()
        if not mask.any():
            return series

        values = series.loc[mask].astype(str)
        unique_values = pd.unique(values)
        hashed_lookup = {val: hashlib.sha256(val.encode()).hexdigest()[:16] for val in unique_values}

        result = series.copy()
        result.loc[mask] = values.map(hashed_lookup)
        return result
    
    def log_data_access(self, 
                       user: str, 
                       action: str, 
                       data_description: str,
                       record_count: Optional[int] = None):
        """
        Log data access for audit trail.
        
        Args:
            user: User accessing data
            action: Action performed (read/write/process)
            data_description: Description of data accessed
            record_count: Number of records accessed
        """
        if not self.enabled:
            return
        
        access_entry = {
            'timestamp': datetime.now().isoformat(),
            'user': user,
            'action': action,
            'data_description': data_description,
            'record_count': record_count
        }
        
        self.access_log.append(access_entry)
        
        logger.info(f"HIPAA Audit: {user} - {action} - {data_description}")
    
    def remove_phi_columns(self, df: pd.DataFrame, phi_columns: List[str]) -> pd.DataFrame:
        """
        Remove columns containing Protected Health Information (PHI).
        
        Args:
            df: Input DataFrame
            phi_columns: List of PHI column names
            
        Returns:
            DataFrame with PHI columns removed
        """
        if not self.enabled:
            return df
        
        logger.info("Removing PHI columns for HIPAA compliance...")
        
        cols_to_remove = [col for col in phi_columns if col in df.columns]
        
        if cols_to_remove:
            df = df.drop(columns=cols_to_remove)
            logger.info(f"  Removed {len(cols_to_remove)} PHI columns: {cols_to_remove}")
        
        return df
    
    def validate_data_retention(self, data_age_days: int, max_retention_days: int = 2555) -> bool:
        """
        Validate data retention period (HIPAA allows up to 7 years = 2555 days).
        
        Args:
            data_age_days: Age of data in days
            max_retention_days: Maximum allowed retention period
            
        Returns:
            True if data retention is compliant
        """
        if not self.enabled:
            return True
        
        is_compliant = data_age_days <= max_retention_days
        
        if not is_compliant:
            logger.warning(f"Data retention violation: {data_age_days} days exceeds maximum {max_retention_days} days")
        
        return is_compliant
    
    def get_access_log(self) -> List[dict]:
        """
        Get the access log for audit purposes.
        
        Returns:
            List of access log entries
        """
        return self.access_log
    
    def export_access_log(self, filepath: str):
        """
        Export access log to file.
        
        Args:
            filepath: Path to save log file
        """
        if not self.access_log:
            logger.warning("No access log entries to export")
            return
        
        import json
        
        with open(filepath, 'w') as f:
            json.dump(self.access_log, f, indent=2)
        
        logger.info(f"Access log exported to {filepath}")
