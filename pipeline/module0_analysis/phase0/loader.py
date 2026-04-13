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

import io
import logging
from pathlib import Path

import pandas as pd

from .config import Phase0Config
from .security import (
    ColumnAllowlist,
    IntegrityVerifier,
    PathValidator,
    log_phase0_event,
)

logger = logging.getLogger(__name__)

# Resolved once at import; loader instances re-derive the workspace
# from this anchor so the security controls have a stable root.
_WORKSPACE_ROOT: Path = Path(__file__).resolve().parents[3]


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

    def __init__(
        self,
        config: Phase0Config,
        *,
        workspace_root: Path | None = None,
    ) -> None:
        self._config = config
        self._workspace_root = workspace_root or _WORKSPACE_ROOT
        self._path_validator = PathValidator(self._workspace_root)
        # Integrity baseline lives next to the config so an operator who
        # owns the config also owns the baseline. Bootstrapped via the
        # `bootstrap_integrity` CLI; verify_and_read() refuses to run
        # without an existing baseline.
        self._integrity = IntegrityVerifier(
            metadata_dir=self._workspace_root / "pipeline/module0_analysis/phase0",
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> pd.DataFrame:
        """Read the raw CSV from the configured path with full security wiring.

        Pipeline:
            1. Resolve the configured path inside the workspace
               (rejects out-of-tree escapes).
            2. Optionally enforce read-only on the raw dataset
               (PHASE0_PROD=1).
            3. Verify the SHA-256 against the signed baseline AND read
               the file bytes in a single shot (no TOCTOU).
            4. Parse the bytes via ``io.BytesIO`` so the parser sees the
               exact same buffer that was hashed.

        Returns:
            Raw DataFrame with all original columns retained.

        Raises:
            FileNotFoundError: If the CSV file does not exist.
            PermissionError: If the path escapes the workspace, or the
                file is writable in production mode.
            IntegrityError: If the file hash differs from the baseline,
                the metadata signature is invalid, or no baseline exists.
        """
        validated_path = self._path_validator.validate_input_path(self._config.data_path)
        self._path_validator.check_read_only(validated_path)

        data, digest = self._integrity.verify_and_read(validated_path)

        df = pd.read_csv(io.BytesIO(data), low_memory=False)
        log_phase0_event(
            "DATASET_LOADED",
            {
                "file": validated_path.name,
                "rows": len(df),
                "cols": len(df.columns),
                "sha256_prefix": digest[:16],
            },
        )
        logger.info(
            "Loaded dataset: %d rows × %d columns from %s " "(integrity verified)",
            len(df),
            len(df.columns),
            validated_path,
        )
        return df

    def validate(self, df: pd.DataFrame) -> None:
        """Assert that all required columns are present in *df*.

        Delegates to ``ColumnAllowlist.validate`` so the failure path is
        audited via ``log_phase0_event`` (and therefore appended to the
        Module 5 signed audit chain).

        Args:
            df: DataFrame returned by :meth:`load`.

        Raises:
            ValueError: If one or more required columns are absent.
        """
        ColumnAllowlist.validate(
            self._config.required_columns,
            set(df.columns),
            context="phase0.required_columns",
        )
        logger.info(
            "Schema validation passed — all %d required columns present",
            len(self._config.required_columns),
        )

    def overview(self, df: pd.DataFrame) -> None:
        """Log dataset shape and column dtypes only.

        Raw row contents are NEVER logged: the dataset contains patient
        biometrics (Temp/SpO2/Pulse_Rate/SYS/DIA/Heart_rate/Resp_Rate/ST)
        which are PHI under HIPAA. Only schema-level information escapes
        this method.

        Args:
            df: Loaded DataFrame to summarise.
        """
        logger.info("Shape  : %d rows × %d columns", len(df), len(df.columns))
        logger.info("Dtypes :\n%s", df.dtypes.to_string())
