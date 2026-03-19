"""Pydantic-validated configuration for the Phase 4 risk-adaptive pipeline.

Loads from ``config/phase4_config.yaml`` via ``Phase4Config.from_yaml()``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import yaml
from pydantic import BaseModel, field_validator


class KScheduleEntry(BaseModel):
    """Time-of-day sensitivity multiplier entry."""

    start_hour: int
    end_hour: int
    k: float


class Phase4Config(BaseModel):
    """Validated configuration for the Phase 4 risk-adaptive pipeline."""

    # ── Data paths (relative to project root) ──────────────────────
    phase3_dir: Path
    phase3_metadata: Path
    phase2_dir: Path
    phase2_metadata: Path
    phase1_train: Path
    phase1_test: Path
    label_column: str = "Label"

    # ── Baseline ───────────────────────────────────────────────────
    mad_multiplier: float = 3.0

    # ── Dynamic threshold ─────────────────────────────────────────
    window_size: int = 100
    k_schedule: List[KScheduleEntry]

    # ── Concept drift ─────────────────────────────────────────────
    drift_threshold: float = 0.20
    recovery_threshold: float = 0.10
    recovery_windows: int = 3

    # ── Risk levels (MAD-relative boundaries) ─────────────────────
    low_upper: float = 0.5
    medium_upper: float = 1.0
    high_upper: float = 2.0

    # ── Biometric columns ─────────────────────────────────────────
    biometric_columns: List[str]

    # ── Output ────────────────────────────────────────────────────
    output_dir: Path = Path("data/phase4")
    baseline_file: str = "baseline_config.json"
    threshold_file: str = "threshold_config.json"
    risk_report_file: str = "risk_report.json"
    drift_log_file: str = "drift_log.csv"

    # ── Reproducibility ───────────────────────────────────────────
    random_state: int = 42

    model_config: Dict[str, Any] = {"arbitrary_types_allowed": True}

    # ── Validators ────────────────────────────────────────────────

    @field_validator("mad_multiplier")
    @classmethod
    def _mad_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"mad_multiplier must be > 0, got {v}")
        return v

    @field_validator("window_size")
    @classmethod
    def _window_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"window_size must be >= 1, got {v}")
        return v

    @field_validator("drift_threshold")
    @classmethod
    def _drift_range(cls, v: float) -> float:
        if not 0.0 < v < 1.0:
            raise ValueError(f"drift_threshold must be in (0, 1), got {v}")
        return v

    @field_validator("recovery_threshold")
    @classmethod
    def _recovery_range(cls, v: float) -> float:
        if not 0.0 < v < 1.0:
            raise ValueError(f"recovery_threshold must be in (0, 1), got {v}")
        return v

    @field_validator("recovery_windows")
    @classmethod
    def _recovery_windows_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"recovery_windows must be >= 1, got {v}")
        return v

    @field_validator("k_schedule")
    @classmethod
    def _at_least_one_schedule(cls, v: List[KScheduleEntry]) -> List[KScheduleEntry]:
        if len(v) < 1:
            raise ValueError("At least one k_schedule entry required")
        return v

    @field_validator("biometric_columns")
    @classmethod
    def _at_least_one_biometric(cls, v: List[str]) -> List[str]:
        if len(v) < 1:
            raise ValueError("At least one biometric column required")
        return v

    # ── YAML loader ───────────────────────────────────────────────

    @classmethod
    def from_yaml(cls, path: Path) -> Phase4Config:
        """Load and validate configuration from a YAML file.

        Args:
            path: Path to the YAML config file.

        Returns:
            Validated Phase4Config instance.
        """
        raw: Dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))

        data = raw.get("data", {})
        baseline = raw.get("baseline", {})
        dyn = raw.get("dynamic_threshold", {})
        drift = raw.get("concept_drift", {})
        risk = raw.get("risk_levels", {})
        output = raw.get("output", {})

        k_entries = [KScheduleEntry(**e) for e in dyn.get("k_schedule", [])]

        return cls(
            phase3_dir=Path(data.get("phase3_dir", "data/phase3")),
            phase3_metadata=Path(data.get("phase3_metadata", "")),
            phase2_dir=Path(data.get("phase2_dir", "data/phase2")),
            phase2_metadata=Path(data.get("phase2_metadata", "")),
            phase1_train=Path(data.get("phase1_train", "")),
            phase1_test=Path(data.get("phase1_test", "")),
            label_column=data.get("label_column", "Label"),
            mad_multiplier=baseline.get("mad_multiplier", 3.0),
            window_size=dyn.get("window_size", 100),
            k_schedule=k_entries,
            drift_threshold=drift.get("drift_threshold", 0.20),
            recovery_threshold=drift.get("recovery_threshold", 0.10),
            recovery_windows=drift.get("recovery_windows", 3),
            low_upper=risk.get("low_upper", 0.5),
            medium_upper=risk.get("medium_upper", 1.0),
            high_upper=risk.get("high_upper", 2.0),
            biometric_columns=raw.get("biometric_columns", []),
            output_dir=Path(output.get("output_dir", "data/phase4")),
            baseline_file=output.get("baseline_file", "baseline_config.json"),
            threshold_file=output.get("threshold_file", "threshold_config.json"),
            risk_report_file=output.get("risk_report_file", "risk_report.json"),
            drift_log_file=output.get("drift_log_file", "drift_log.csv"),
            random_state=raw.get("random_state", 42),
        )
