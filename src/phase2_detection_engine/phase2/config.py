"""Phase 2 configuration — pydantic-validated settings.

Loads from ``config/phase2_config.yaml`` and validates all fields
at construction time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, field_validator


class Phase2Config(BaseModel):
    """Validated configuration for the Phase 2 detection pipeline."""

    # Data paths (relative to project root)
    train_parquet: Path
    test_parquet: Path
    metadata_file: Path
    report_file: Path
    label_column: str = "Label"

    # Reshape
    timesteps: int = 20
    stride: int = 1

    # CNN
    cnn_filters_1: int = 64
    cnn_filters_2: int = 128
    cnn_kernel_size: int = 3
    cnn_activation: str = "relu"
    cnn_pool_size: int = 2

    # BiLSTM
    bilstm_units_1: int = 128
    bilstm_units_2: int = 64
    dropout_rate: float = 0.3

    # Attention
    attention_units: int = 128

    # Output
    output_dir: Path = Path("data/phase2")
    model_file: str = "detection_model.weights.h5"
    attention_parquet: str = "attention_output.parquet"
    report_json: str = "detection_report.json"

    # Reproducibility
    random_state: int = 42

    # Inference
    predict_batch_size: int = 256

    model_config = {"arbitrary_types_allowed": True}

    @field_validator("timesteps")
    @classmethod
    def _timesteps_positive(cls, v: int) -> int:
        if v < 2:
            raise ValueError(f"timesteps must be >= 2, got {v}")
        return v

    @field_validator("stride")
    @classmethod
    def _stride_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"stride must be >= 1, got {v}")
        return v

    @field_validator("dropout_rate")
    @classmethod
    def _dropout_in_range(cls, v: float) -> float:
        if not 0.0 <= v < 1.0:
            raise ValueError(f"dropout_rate must be in [0, 1), got {v}")
        return v

    @field_validator(
        "cnn_filters_1", "cnn_filters_2", "bilstm_units_1",
        "bilstm_units_2", "attention_units",
    )
    @classmethod
    def _units_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"units/filters must be >= 1, got {v}")
        return v

    @field_validator("cnn_pool_size")
    @classmethod
    def _pool_size_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"cnn_pool_size must be >= 1, got {v}")
        return v

    @field_validator("model_file")
    @classmethod
    def _model_file_extension(cls, v: str) -> str:
        if not v.endswith(".weights.h5"):
            raise ValueError(
                f"model_file must end with '.weights.h5' "
                f"(Keras 3 requirement), got '{v}'"
            )
        return v

    @classmethod
    def from_yaml(cls, path: Path) -> Phase2Config:
        """Load and validate configuration from a YAML file.

        Args:
            path: Path to the YAML config file.

        Returns:
            Validated Phase2Config instance.
        """
        raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))

        data = raw.get("data", {})
        reshape = raw.get("reshape", {})
        cnn = raw.get("cnn", {})
        bilstm = raw.get("bilstm", {})
        attention = raw.get("attention", {})
        output = raw.get("output", {})

        return cls(
            train_parquet=Path(data.get("train_parquet", "")),
            test_parquet=Path(data.get("test_parquet", "")),
            metadata_file=Path(data.get("metadata_file", "")),
            report_file=Path(data.get("report_file", "")),
            label_column=data.get("label_column", "Label"),
            timesteps=reshape.get("timesteps", 20),
            stride=reshape.get("stride", 1),
            cnn_filters_1=cnn.get("filters_1", 64),
            cnn_filters_2=cnn.get("filters_2", 128),
            cnn_kernel_size=cnn.get("kernel_size", 3),
            cnn_activation=cnn.get("activation", "relu"),
            cnn_pool_size=cnn.get("pool_size", 2),
            bilstm_units_1=bilstm.get("units_1", 128),
            bilstm_units_2=bilstm.get("units_2", 64),
            dropout_rate=bilstm.get("dropout_rate", 0.3),
            attention_units=attention.get("units", 128),
            output_dir=Path(output.get("output_dir", "data/phase2")),
            model_file=output.get("model_file", "detection_model.weights.h5"),
            attention_parquet=output.get("attention_parquet", "attention_output.parquet"),
            report_json=output.get("report_file", "detection_report.json"),
            random_state=raw.get("random_state", 42),
        )
