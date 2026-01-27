"""Shared utilities for CSV loading and memory-safe processing."""

from pathlib import Path
from typing import Union
import pandas as pd


def validate_and_resolve_path(path_str: str) -> Path:
    if not path_str or not isinstance(path_str, str):
        raise ValueError("Path must be a non-empty string")
    if ".." in path_str or "~" in path_str:
        raise ValueError("Path traversal sequences (.., ~) are not allowed")
    path = Path(path_str).resolve()
    if path.is_symlink():
        raise ValueError("Symbolic links are not allowed for security reasons")
    return path


def validate_filename(filename: str) -> Path:
    if not filename or not isinstance(filename, str):
        raise ValueError("Filename must be a non-empty string")
    if len(filename) > 255:
        raise ValueError("Filename exceeds maximum length of 255")
    if filename.startswith(("/", "\\")):
        raise ValueError("Absolute paths are not allowed")
    if ".." in filename or "~" in filename:
        raise ValueError("Path traversal sequences (.., ~) are not allowed")
    if "\x00" in filename:
        raise ValueError("Null bytes in filename are not allowed")

    path = Path(filename)
    if len(path.parts) > 1:
        raise ValueError("Subdirectories in filename are not allowed")
    return path


def ensure_within_dir(path: Union[str, Path], base_dir: Union[str, Path]) -> Path:
    path_obj = Path(path).resolve()
    base_obj = Path(base_dir).resolve()
    path_obj.relative_to(base_obj)
    if path_obj.is_symlink():
        raise ValueError(f"Symbolic links are not allowed: {path_obj}")
    return path_obj


def downcast_df(df: pd.DataFrame, categorical_threshold: int = 50) -> pd.DataFrame:
    for col in df.columns:
        col_data = df[col]
        if pd.api.types.is_numeric_dtype(col_data):
            if pd.api.types.is_integer_dtype(col_data):
                df[col] = pd.to_numeric(col_data, downcast='integer')
            else:
                df[col] = pd.to_numeric(col_data, downcast='float')
        elif pd.api.types.is_object_dtype(col_data):
            if col_data.nunique(dropna=True) <= categorical_threshold:
                df[col] = col_data.astype('category')
    return df
