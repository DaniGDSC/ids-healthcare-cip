"""Pydantic-validated configuration for the Phase 5 explanation pipeline.

Loads from ``config/phase5_config.yaml`` via ``Phase5Config.from_yaml()``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import yaml
from pydantic import BaseModel, field_validator


class ExplanationTemplates(BaseModel):
    """Risk-level explanation template strings."""

    CRITICAL: str
    HIGH: str
    MEDIUM: str
    LOW: str


class Phase5Config(BaseModel):
    """Validated configuration for Phase 5 explanation pipeline."""

    # ── Data paths ────────────────────────────────────────────
    phase4_dir: Path
    phase4_metadata: Path
    phase3_dir: Path
    phase3_metadata: Path
    phase2_dir: Path
    phase2_metadata: Path
    phase1_train: Path
    phase1_test: Path
    label_column: str = "Label"

    # ── SHAP parameters ──────────────────────────────────────
    background_samples: int = 100
    max_explain_samples: int = 200
    top_features: int = 10
    max_waterfall_charts: int = 5
    max_timeline_charts: int = 3

    # ── Output ────────────────────────────────────────────────
    output_dir: Path = Path("data/phase5")
    shap_values_file: str = "shap_values.parquet"
    explanation_report_file: str = "explanation_report.json"
    metadata_file: str = "explanation_metadata.json"
    charts_dir: str = "charts"

    # ── Explanation templates ─────────────────────────────────
    explanation_templates: ExplanationTemplates

    # ── Biometric columns ─────────────────────────────────────
    biometric_columns: List[str]

    # ── Reproducibility ───────────────────────────────────────
    random_state: int = 42

    model_config = {"arbitrary_types_allowed": True}

    # ── Validators ────────────────────────────────────────────

    @field_validator("background_samples")
    @classmethod
    def _bg_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"background_samples must be >= 1, got {v}")
        return v

    @field_validator("max_explain_samples")
    @classmethod
    def _max_explain_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"max_explain_samples must be >= 1, got {v}")
        return v

    @field_validator("top_features")
    @classmethod
    def _top_features_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"top_features must be >= 1, got {v}")
        return v

    @field_validator("max_waterfall_charts")
    @classmethod
    def _max_waterfall_non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"max_waterfall_charts must be >= 0, got {v}")
        return v

    @field_validator("max_timeline_charts")
    @classmethod
    def _max_timeline_non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"max_timeline_charts must be >= 0, got {v}")
        return v

    @field_validator("biometric_columns")
    @classmethod
    def _at_least_one_biometric(cls, v: List[str]) -> List[str]:
        if len(v) < 1:
            raise ValueError("At least one biometric column required")
        return v

    # ── YAML loader ───────────────────────────────────────────

    @classmethod
    def from_yaml(cls, path: Path) -> Phase5Config:
        """Load and validate configuration from a YAML file."""
        raw: Dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))

        data = raw.get("data", {})
        shap = raw.get("shap", {})
        output = raw.get("output", {})
        templates_raw = raw.get("explanation_templates", {})

        templates = ExplanationTemplates(**templates_raw)

        return cls(
            phase4_dir=Path(data.get("phase4_dir", "data/phase4")),
            phase4_metadata=Path(data.get("phase4_metadata", "")),
            phase3_dir=Path(data.get("phase3_dir", "data/phase3")),
            phase3_metadata=Path(data.get("phase3_metadata", "")),
            phase2_dir=Path(data.get("phase2_dir", "data/phase2")),
            phase2_metadata=Path(data.get("phase2_metadata", "")),
            phase1_train=Path(data.get("phase1_train", "")),
            phase1_test=Path(data.get("phase1_test", "")),
            label_column=data.get("label_column", "Label"),
            background_samples=shap.get("background_samples", 100),
            max_explain_samples=shap.get("max_explain_samples", 200),
            top_features=shap.get("top_features", 10),
            max_waterfall_charts=shap.get("max_waterfall_charts", 5),
            max_timeline_charts=shap.get("max_timeline_charts", 3),
            output_dir=Path(output.get("output_dir", "data/phase5")),
            shap_values_file=output.get("shap_values_file", "shap_values.parquet"),
            explanation_report_file=output.get(
                "explanation_report_file", "explanation_report.json"
            ),
            metadata_file=output.get("metadata_file", "explanation_metadata.json"),
            charts_dir=output.get("charts_dir", "charts"),
            explanation_templates=templates,
            biometric_columns=raw.get("biometric_columns", []),
            random_state=raw.get("random_state", 42),
        )
