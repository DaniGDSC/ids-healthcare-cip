"""Interfaces for pluggable preprocessing components."""

from typing import Protocol, List, Optional, Tuple, Dict
import pandas as pd


class DataLoaderProtocol(Protocol):
    """Abstraction for data loaders to enable swapping implementations."""

    def smart_load(
        self,
        pattern: str = "*.csv",
        memory_threshold_mb: int = 2000,
        chunksize: int = 200_000,
        use_chunking: bool = True,
        io_workers: int = 1,
    ) -> pd.DataFrame:
        ...

    def load_csv_files(
        self,
        pattern: str = "*.csv",
        use_smart_loading: bool = True,
        io_workers: int = 1,
    ) -> pd.DataFrame:
        ...


class PreprocessingStep(Protocol):
    """Strategy interface for dataframe preprocessing steps."""

    name: str
    enabled: bool

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        ...
