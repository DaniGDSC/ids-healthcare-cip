"""Configuration dataclass for Phase 0 analysis.

All paths, thresholds, and tuneable parameters are defined in ``config.yaml``
and surfaced here as a validated ``Phase0Config`` dataclass.  Nothing in the
analysis pipeline hard-codes a path or a magic number.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List

import yaml

logger = logging.getLogger(__name__)

_MISSING_WARN_PCT_DEFAULT: float = 5.0


@dataclass
class Phase0Config:
    """Validated configuration for Phase 0 EDA.

    Attributes:
        data_path: Path to the raw WUSTL-EHMS CSV file.
        output_dir: Directory where all analysis artifacts are written.
        label_column: Binary label column name (0 = Normal, 1 = Attack).
        required_columns: Columns that must be present; loader raises on absence.
        correlation_threshold: Minimum |r| to flag a high-correlation pair.
        head_rows: Number of rows shown in the dataset overview.
        missing_value_warn_pct: Percentage threshold that triggers a WARNING log.
        stats_report_file: Filename for the JSON statistics report.
        high_correlations_file: Filename for the high-correlations CSV.
        correlation_matrix_file: Filename for the full correlation matrix Parquet.
    """

    data_path: Path
    output_dir: Path
    label_column: str
    required_columns: List[str]
    leakage_columns: List[str]
    network_feature_count: int
    biometric_feature_count: int
    correlation_threshold: float
    head_rows: int
    missing_value_warn_pct: float
    outlier_iqr_multiplier: float
    top_variance_k: int
    random_state: int
    train_ratio: float
    test_ratio: float
    stats_report_file: str
    high_correlations_file: str
    correlation_matrix_file: str
    quality_report_file: str

    def __post_init__(self) -> None:
        """Validate all fields after construction.

        Raises:
            ValueError: If any field violates its invariant.
        """
        if not 0.0 < self.correlation_threshold < 1.0:
            raise ValueError(
                f"correlation_threshold must be in (0, 1), "
                f"got {self.correlation_threshold}"
            )
        if self.head_rows < 1:
            raise ValueError(
                f"head_rows must be >= 1, got {self.head_rows}"
            )
        if self.missing_value_warn_pct < 0.0:
            raise ValueError(
                f"missing_value_warn_pct must be >= 0, "
                f"got {self.missing_value_warn_pct}"
            )
        if self.outlier_iqr_multiplier <= 0.0:
            raise ValueError(
                f"outlier_iqr_multiplier must be > 0, "
                f"got {self.outlier_iqr_multiplier}"
            )
        if not self.label_column:
            raise ValueError("label_column must not be empty")
        if not self.required_columns:
            raise ValueError("required_columns must contain at least one entry")

    @classmethod
    def from_yaml(cls, path: Path) -> "Phase0Config":
        """Load and validate configuration from a YAML file.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            Fully validated ``Phase0Config`` instance.

        Raises:
            FileNotFoundError: If *path* does not exist.
            KeyError: If a required YAML key is absent.
            ValueError: If a value fails validation in ``__post_init__``.
        """
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        raw: dict = yaml.safe_load(path.read_text())
        dataset = raw["dataset"]
        analysis = raw["analysis"]
        output = raw["output"]

        cfg = cls(
            data_path=Path(dataset["data_path"]),
            output_dir=Path(output["output_dir"]),
            label_column=dataset["label_column"],
            required_columns=list(
                dataset.get("required_columns", [dataset["label_column"]])
            ),
            leakage_columns=list(dataset.get("leakage_columns", [])),
            network_feature_count=int(dataset.get("network_feature_count", 0)),
            biometric_feature_count=int(dataset.get("biometric_feature_count", 0)),
            correlation_threshold=float(analysis["correlation_threshold"]),
            head_rows=int(analysis["head_rows"]),
            missing_value_warn_pct=float(
                analysis.get("missing_value_warn_pct", _MISSING_WARN_PCT_DEFAULT)
            ),
            outlier_iqr_multiplier=float(
                analysis.get("outlier_iqr_multiplier", 1.5)
            ),
            top_variance_k=int(analysis.get("top_variance_k", 5)),
            random_state=int(analysis.get("random_state", 42)),
            train_ratio=float(analysis.get("train_ratio", 0.70)),
            test_ratio=float(analysis.get("test_ratio", 0.30)),
            stats_report_file=output["stats_report_file"],
            high_correlations_file=output["high_correlations_file"],
            correlation_matrix_file=output["correlation_matrix_file"],
            quality_report_file=output.get(
                "quality_report_file", "report_section_quality.md"
            ),
        )
        logger.info("Configuration loaded from %s", path)
        return cfg
