"""Phase 1 configuration — pydantic-validated settings.

Loads from ``config.yaml`` and validates all fields at construction time.
Paths are resolved relative to the project root.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import yaml
from pydantic import BaseModel, field_validator, model_validator


class Phase1Config(BaseModel):
    """Validated configuration for the Phase 1 preprocessing pipeline.

    All paths are stored relative to the project root and resolved
    at load time.
    """

    # Data
    input_dir: Path
    output_dir: Path
    file_pattern: str = "*.csv"
    label_column: str = "Label"

    # HIPAA
    hipaa_enabled: bool = True
    hipaa_columns: List[str]

    # Missing values
    biometric_columns: List[str]
    biometric_strategy: str = "ffill"
    network_strategy: str = "dropna"

    # Redundancy
    correlation_enabled: bool = True
    correlation_threshold: float = 0.95
    phase0_corr_file: Path

    # Variance filtering
    variance_enabled: bool = True
    variance_max_unique: int = 1

    # Split
    train_ratio: float = 0.70
    test_ratio: float = 0.30
    random_state: int = 42
    stratify: bool = True

    # SMOTE
    smote_enabled: bool = True
    smote_strategy: str = "auto"
    smote_k_neighbors: int = 5

    # Scaling
    scaling_method: str = "robust"

    # Phase 0 artifacts
    phase0_stats_file: Path = Path("results/phase0_analysis/stats_report.json")
    phase0_integrity_file: Path = Path("results/phase0_analysis/dataset_integrity.json")

    # Output filenames
    train_parquet: str = "train_phase1.parquet"
    test_parquet: str = "test_phase1.parquet"
    scaler_file: str = "robust_scaler.pkl"
    report_file: str = "preprocessing_report.json"

    model_config = {"arbitrary_types_allowed": True}

    @field_validator("correlation_threshold")
    @classmethod
    def _threshold_in_range(cls, v: float) -> float:
        if not 0.0 < v <= 1.0:
            raise ValueError(f"correlation_threshold must be in (0, 1], got {v}")
        return v

    @field_validator("smote_k_neighbors")
    @classmethod
    def _k_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"smote_k_neighbors must be ≥ 1, got {v}")
        return v

    @model_validator(mode="after")
    def _ratios_sum_to_one(self) -> Phase1Config:
        total = round(self.train_ratio + self.test_ratio, 4)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"train_ratio + test_ratio must equal 1.0, got {total}"
            )
        return self

    @classmethod
    def from_yaml(cls, path: Path) -> Phase1Config:
        """Load and validate configuration from a YAML file.

        Args:
            path: Path to the YAML config file.

        Returns:
            Validated Phase1Config instance.
        """
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))

        data = raw.get("data", {})
        hipaa = raw.get("hipaa", {})
        mv = raw.get("missing_values", {})
        corr = raw.get("correlation_removal", {})
        var = raw.get("variance_filtering", {})
        split = raw.get("splitting", {})
        smote = raw.get("smote", {})
        norm = raw.get("normalization", {})
        output = raw.get("output", {})

        return cls(
            input_dir=Path(data.get("input_dir", "data/raw/WUSTL-EHMS")),
            output_dir=Path(data.get("output_dir", "data/processed")),
            file_pattern=data.get("file_pattern", "*.csv"),
            label_column=data.get("label_column", "Label"),
            hipaa_enabled=hipaa.get("enabled", True),
            hipaa_columns=hipaa.get("remove_columns", []),
            biometric_columns=mv.get("biometric_columns", []),
            biometric_strategy=mv.get("biometric_strategy", "ffill"),
            network_strategy=mv.get("network_strategy", "dropna"),
            correlation_enabled=corr.get("enabled", True),
            correlation_threshold=corr.get("threshold", 0.95),
            phase0_corr_file=Path(corr.get(
                "phase0_corr_file",
                "results/phase0_analysis/high_correlations.csv",
            )),
            variance_enabled=var.get("enabled", True),
            variance_max_unique=var.get("max_unique", 1),
            train_ratio=split.get("train_ratio", 0.70),
            test_ratio=split.get("test_ratio", 0.30),
            random_state=split.get("random_state", 42),
            stratify=split.get("stratify", True),
            smote_enabled=smote.get("enabled", True),
            smote_strategy=smote.get("sampling_strategy", "auto"),
            smote_k_neighbors=smote.get("k_neighbors", 5),
            scaling_method=norm.get("method", "robust"),
            train_parquet=output.get("train_parquet", "train_phase1.parquet"),
            test_parquet=output.get("test_parquet", "test_phase1.parquet"),
            scaler_file=output.get("scaler_file", "robust_scaler.pkl"),
            report_file=output.get("report_file", "preprocessing_report.json"),
        )
