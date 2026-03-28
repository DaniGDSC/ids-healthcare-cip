"""Pydantic-validated configuration for the Phase 2.5 tuning pipeline.

Loads from ``config/phase2_5_config.yaml`` via ``Phase2_5Config.from_yaml()``.

Simplified to 5 core tunable parameters (head_lr, finetune_lr, cw_attack,
head_epochs, ft_epochs) identified by Optuna importance analysis as the
only parameters with >5% contribution.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, field_validator


class SearchSpaceConfig(BaseModel):
    """Hyperparameter search space — 5 core parameters."""

    head_lr_low: float = 5e-4
    head_lr_high: float = 5e-3
    finetune_lr_low: float = 1e-6
    finetune_lr_high: float = 1e-4
    cw_attack_low: float = 1.0
    cw_attack_high: float = 10.0
    head_epochs: List[int] = [3, 5, 7, 9]
    ft_epochs: List[int] = [1, 2, 3, 4, 5]


class AblationVariantConfig(BaseModel):
    """Configuration for a single ablation variant."""

    name: str
    description: str
    remove: Optional[str] = None
    replace: Optional[str] = None
    override: Optional[Dict[str, Any]] = None


class QuickTrainConfig(BaseModel):
    """Training settings for search evaluation."""

    epochs: int = 5
    validation_split: float = 0.2
    early_stopping_patience: int = 3
    dense_units: int = 64
    dense_activation: str = "relu"
    head_dropout_rate: float = 0.3


class MultiSeedConfig(BaseModel):
    """Configuration for multi-seed validation of top-K configs."""

    enabled: bool = False
    top_k: int = 3
    seeds: List[int] = [42, 123, 456, 789, 1024]
    full_epochs: int = 10


class Phase2_5Config(BaseModel):
    """Validated configuration for the Phase 2.5 tuning pipeline."""

    # Data paths (relative to project root)
    phase1_train: Path
    phase1_test: Path
    phase2_config: Path
    label_column: str = "Label"

    # Search settings
    max_trials: int = 30
    search_metric: str = "attack_f2"
    search_direction: str = "maximize"
    search_space: SearchSpaceConfig = SearchSpaceConfig()

    # Quick-train settings
    quick_train: QuickTrainConfig = QuickTrainConfig()

    # Multi-seed validation
    multi_seed: MultiSeedConfig = MultiSeedConfig()

    # Ablation variants
    ablation_baseline: str = "full"
    ablation_variants: List[AblationVariantConfig] = []

    # Output
    output_dir: Path = Path("data/phase2_5")
    tuning_results_file: str = "tuning_results.json"
    ablation_results_file: str = "ablation_results.json"
    best_config_file: str = "best_config.json"
    importance_file: str = "param_importance.json"
    multi_seed_file: str = "multi_seed_validation.json"
    report_file: str = "tuning_report.json"

    # Reproducibility
    random_state: int = 42

    model_config: Dict[str, Any] = {"arbitrary_types_allowed": True}

    # ── Validators ────────────────────────────────────────────────

    @field_validator("search_direction")
    @classmethod
    def _valid_direction(cls, v: str) -> str:
        if v not in ("maximize", "minimize"):
            raise ValueError(f"search_direction must be 'maximize' or 'minimize', got '{v}'")
        return v

    @field_validator("search_metric")
    @classmethod
    def _valid_metric(cls, v: str) -> str:
        allowed = {
            "f1_score", "auc_roc", "accuracy", "precision", "recall",
            "attack_f1", "attack_f2", "attack_recall", "attack_precision", "macro_f1",
        }
        if v not in allowed:
            raise ValueError(f"search_metric must be one of {allowed}, got '{v}'")
        return v

    @field_validator("max_trials")
    @classmethod
    def _positive_trials(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"max_trials must be >= 1, got {v}")
        return v

    # ── YAML loader ───────────────────────────────────────────────

    @classmethod
    def from_yaml(cls, path: Path) -> Phase2_5Config:
        """Load and validate configuration from a YAML file."""
        raw: Dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))

        data = raw.get("data", {})
        search = raw.get("search", {})
        space_raw = search.get("space", {})
        quick = raw.get("quick_train", {})
        ablation = raw.get("ablation", {})
        output = raw.get("output", {})
        multi_seed_raw = raw.get("multi_seed", {})

        space = SearchSpaceConfig(**space_raw) if space_raw else SearchSpaceConfig()

        variants = [
            AblationVariantConfig(**v) for v in ablation.get("variants", [])
        ]

        head_cfg = quick.get("classification_head", {})
        qt = QuickTrainConfig(
            epochs=quick.get("epochs", 5),
            validation_split=quick.get("validation_split", 0.2),
            early_stopping_patience=quick.get("early_stopping_patience", 3),
            dense_units=head_cfg.get("dense_units", 64),
            dense_activation=head_cfg.get("dense_activation", "relu"),
            head_dropout_rate=head_cfg.get("dropout_rate", 0.3),
        )

        ms = MultiSeedConfig(
            enabled=multi_seed_raw.get("enabled", False),
            top_k=multi_seed_raw.get("top_k", 3),
            seeds=multi_seed_raw.get("seeds", [42, 123, 456, 789, 1024]),
            full_epochs=multi_seed_raw.get("full_epochs", 10),
        )

        return cls(
            phase1_train=Path(data.get("phase1_train", "")),
            phase1_test=Path(data.get("phase1_test", "")),
            phase2_config=Path(data.get("phase2_config", "")),
            label_column=data.get("label_column", "Label"),
            max_trials=search.get("max_trials", 30),
            search_metric=search.get("metric", "attack_f2"),
            search_direction=search.get("direction", "maximize"),
            search_space=space,
            quick_train=qt,
            multi_seed=ms,
            ablation_baseline=ablation.get("baseline", "full"),
            ablation_variants=variants,
            output_dir=Path(output.get("output_dir", "data/phase2_5")),
            tuning_results_file=output.get("tuning_results_file", "tuning_results.json"),
            ablation_results_file=output.get("ablation_results_file", "ablation_results.json"),
            best_config_file=output.get("best_config_file", "best_config.json"),
            importance_file=output.get("importance_file", "param_importance.json"),
            multi_seed_file=output.get("multi_seed_file", "multi_seed_validation.json"),
            report_file=output.get("report_file", "tuning_report.json"),
            random_state=raw.get("random_state", 42),
        )
