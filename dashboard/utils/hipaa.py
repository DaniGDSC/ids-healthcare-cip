"""HIPAA compliance utilities for PHI sanitization.

All display-bound DataFrames must pass through sanitize_display()
before rendering in any st.dataframe() or st.table() call.
"""

from __future__ import annotations

import hashlib
from typing import List, Set

import pandas as pd

HIPAA_HASH_PREFIX_LENGTH: int = 8

PHI_COLUMNS: Set[str] = {
    "srcaddr", "dstaddr", "sport", "dport", "srcmac", "dstmac",
    "patient_id", "device_serial", "name", "ssn", "mrn", "dob",
    "address", "phone", "email", "src ip", "dst ip", "src port",
    "dst port", "flow id",
}


def _hash_value(value: str) -> str:
    """Hash a PHI value to a truncated SHA-256 prefix.

    Args:
        value: Raw PHI string.

    Returns:
        Truncated hash string with ellipsis.
    """
    h = hashlib.sha256(str(value).encode()).hexdigest()
    return h[:HIPAA_HASH_PREFIX_LENGTH] + "..."


def sanitize_display(df: pd.DataFrame) -> pd.DataFrame:
    """Sanitize a DataFrame by hashing any PHI columns.

    Must be called before ANY st.dataframe() or st.table() call.

    Args:
        df: Raw DataFrame that may contain PHI fields.

    Returns:
        Sanitized DataFrame with PHI columns hashed.
    """
    df = df.copy()
    for col in df.columns:
        if col.lower().strip() in PHI_COLUMNS:
            df[col] = df[col].apply(lambda x: _hash_value(str(x)))
    return df


def detect_phi_columns(df: pd.DataFrame) -> List[str]:
    """Detect columns that match known PHI field names.

    Args:
        df: DataFrame to scan.

    Returns:
        List of column names matching PHI patterns.
    """
    return [c for c in df.columns if c.lower().strip() in PHI_COLUMNS]


def assert_no_phi(df: pd.DataFrame, context: str = "") -> None:
    """Assert that no PHI columns are present in the DataFrame.

    Args:
        df: DataFrame to check.
        context: Description of where this check is being performed.

    Raises:
        ValueError: If PHI columns are found.
    """
    found = detect_phi_columns(df)
    if found:
        raise ValueError(
            f"PHI columns detected in {context}: {found}. "
            "Apply sanitize_display() before rendering."
        )
