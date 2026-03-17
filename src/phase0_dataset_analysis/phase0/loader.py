"""DataLoader — load and validate the raw WUSTL-EHMS-2020 CSV dataset.

Single Responsibility
---------------------
This class does exactly two things: read a CSV from disk, and verify that
the resulting DataFrame contains the columns declared as required in the
configuration.  No statistics, no exports, no transformations.

Dependency Inversion
--------------------
The data path and required-column list are injected via ``Phase0Config``
rather than hard-coded, making the loader fully testable with any config.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .config import Phase0Config

logger = logging.getLogger(__name__)


class DataLoader:
    """Load and validate the raw WUSTL-EHMS-2020 CSV dataset.

    Args:
        config: Validated ``Phase0Config`` instance providing the data path,
                required columns, and display preferences.

    Example::

        config = Phase0Config.from_yaml(Path("phase0/config.yaml"))
        loader = DataLoader(config)
        df = loader.load()
        loader.validate(df)
        loader.overview(df)
    """

    def __init__(self, config: Phase0Config) -> None:
        self._config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> pd.DataFrame:
        """Read the raw CSV from the configured path.

        Returns:
            Raw DataFrame with all original columns retained.

        Raises:
            FileNotFoundError: If the CSV file does not exist.
        """
        path: Path = self._config.data_path
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")

        df = pd.read_csv(path, low_memory=False)
        logger.info(
            "Loaded dataset: %d rows × %d columns from %s",
            len(df), len(df.columns), path,
        )
        return df

    def validate(self, df: pd.DataFrame) -> None:
        """Assert that all required columns are present in *df*.

        Args:
            df: DataFrame returned by :meth:`load`.

        Raises:
            KeyError: If one or more required columns are absent.
                      The error message lists every missing column.
        """
        missing = [c for c in self._config.required_columns if c not in df.columns]
        if missing:
            msg = f"Required columns missing from dataset: {missing}"
            logger.error(msg)
            raise KeyError(msg)
        logger.info(
            "Schema validation passed — all %d required columns present",
            len(self._config.required_columns),
        )

    def overview(self, df: pd.DataFrame) -> None:
        """Log dataset shape, column dtypes, and the first *head_rows* rows.

        Args:
            df: Loaded DataFrame to summarise.
        """
        logger.info("Shape  : %d rows × %d columns", len(df), len(df.columns))
        logger.info("Dtypes :\n%s", df.dtypes.to_string())
        logger.info(
            "Head (%d rows):\n%s",
            self._config.head_rows,
            df.head(self._config.head_rows).to_string(),
        )
