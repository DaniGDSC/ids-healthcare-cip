"""Pydantic-validated configuration for the Phase 3 classification pipeline.

Loads from ``config/phase3_config.yaml`` via ``Phase3Config.from_yaml()``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import yaml
from pydantic import BaseModel, field_validator


class TrainingPhaseConfig(BaseModel):
    """Configuration for a single progressive-unfreezing phase."""

    name: str
    epochs: int
    learning_rate: float
    frozen: List[str]


class Phase3Config(BaseModel):
    """Validated configuration for the Phase 3 classification pipeline."""

    # Data paths (relative to project root)
    phase2_dir: Path
    phase1_train: Path
    phase1_test: Path
    phase2_metadata: Path
    label_column: str = "Label"

    # Classification head
    dense_units: int = 64
    dense_activation: str = "relu"
    head_dropout_rate: float = 0.3

    # Training
    training_phases: List[TrainingPhaseConfig]
    batch_size: int = 256
    validation_split: float = 0.2

    # Callbacks
    early_stopping_patience: int = 3
    reduce_lr_patience: int = 2
    reduce_lr_factor: float = 0.5

    # Evaluation
    threshold: float = 0.5

    # Output
    output_dir: Path = Path("data/phase3")
    model_file: str = "classification_model.weights.h5"
    metrics_file: str = "metrics_report.json"
    confusion_matrix_file: str = "confusion_matrix.csv"
    history_file: str = "training_history.json"

    # Reproducibility
    random_state: int = 42

    model_config: Dict[str, Any] = {"arbitrary_types_allowed": True}

    # ── Validators ────────────────────────────────────────────────

    @field_validator("dense_units")
    @classmethod
    def _units_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"dense_units must be >= 1, got {v}")
        return v

    @field_validator("head_dropout_rate")
    @classmethod
    def _dropout_range(cls, v: float) -> float:
        if not 0.0 <= v < 1.0:
            raise ValueError(f"head_dropout_rate must be in [0, 1), got {v}")
        return v

    @field_validator("threshold")
    @classmethod
    def _threshold_range(cls, v: float) -> float:
        if not 0.0 < v < 1.0:
            raise ValueError(f"threshold must be in (0, 1), got {v}")
        return v

    @field_validator("training_phases")
    @classmethod
    def _at_least_one_phase(cls, v: List[TrainingPhaseConfig]) -> List[TrainingPhaseConfig]:
        if len(v) < 1:
            raise ValueError("At least one training phase required")
        return v

    @field_validator("model_file")
    @classmethod
    def _weights_extension(cls, v: str) -> str:
        if not v.endswith(".weights.h5"):
            raise ValueError(
                f"model_file must end with '.weights.h5' " f"(Keras 3 requirement), got '{v}'"
            )
        return v

    # ── YAML loader ───────────────────────────────────────────────

    @classmethod
    def from_yaml(cls, path: Path) -> Phase3Config:
        """Load and validate configuration from a YAML file.

        Args:
            path: Path to the YAML config file.

        Returns:
            Validated Phase3Config instance.
        """
        raw: Dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))

        data = raw.get("data", {})
        head = raw.get("classification_head", {})
        training = raw.get("training", {})
        callbacks = raw.get("callbacks", {})
        evaluation = raw.get("evaluation", {})
        output = raw.get("output", {})

        phases = [TrainingPhaseConfig(**p) for p in training.get("phases", [])]

        return cls(
            phase2_dir=Path(data.get("phase2_dir", "data/phase2")),
            phase1_train=Path(data.get("phase1_train", "")),
            phase1_test=Path(data.get("phase1_test", "")),
            phase2_metadata=Path(data.get("phase2_metadata", "")),
            label_column=data.get("label_column", "Label"),
            dense_units=head.get("dense_units", 64),
            dense_activation=head.get("dense_activation", "relu"),
            head_dropout_rate=head.get("dropout_rate", 0.3),
            training_phases=phases,
            batch_size=training.get("batch_size", 256),
            validation_split=training.get("validation_split", 0.2),
            early_stopping_patience=callbacks.get("early_stopping_patience", 3),
            reduce_lr_patience=callbacks.get("reduce_lr_patience", 2),
            reduce_lr_factor=callbacks.get("reduce_lr_factor", 0.5),
            threshold=evaluation.get("threshold", 0.5),
            output_dir=Path(output.get("output_dir", "data/phase3")),
            model_file=output.get("model_file", "classification_model.weights.h5"),
            metrics_file=output.get("metrics_file", "metrics_report.json"),
            confusion_matrix_file=output.get("confusion_matrix_file", "confusion_matrix.csv"),
            history_file=output.get("history_file", "training_history.json"),
            random_state=raw.get("random_state", 42),
        )
