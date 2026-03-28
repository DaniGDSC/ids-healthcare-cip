"""Pydantic-validated configuration for the Phase 2.5 tuning pipeline.

Loads from ``config/phase2_5_config.yaml`` via ``Phase2_5Config.from_yaml()``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from pydantic import BaseModel, field_validator


# ── Continuous parameter range (log-scale support) ────────────────

class ContinuousRange(BaseModel):
    """A continuous parameter range with optional log-scale sampling."""

    low: float
    high: float
    log: bool = False


class SearchSpaceConfig(BaseModel):
    """Hyperparameter search space definition.

    Categorical parameters use ``List[int]`` or ``List[float]``.
    Continuous parameters use ``ContinuousRange`` for log-uniform sampling.
    """

    cnn_kernel_size: List[int] = [3, 5, 7]
    dropout_rate: Union[List[float], ContinuousRange] = ContinuousRange(low=0.001, high=0.1, log=False)
    timesteps: List[int] = [10, 20, 30]
    batch_size: List[int] = [256, 512]
    learning_rate: Union[List[float], ContinuousRange] = ContinuousRange(low=0.0001, high=0.001, log=True)

    # Focal loss tuning
    focal_alpha: Union[List[float], ContinuousRange] = ContinuousRange(low=0.15, high=0.75, log=False)
    focal_gamma: Union[List[float], ContinuousRange] = ContinuousRange(low=0.5, high=3.0, log=False)

    # Phase 3 unfreezing schedule (joint optimisation)
    phase_a_lr: Union[List[float], ContinuousRange] = ContinuousRange(low=0.0005, high=0.005, log=True)
    phase_b_lr: Union[List[float], ContinuousRange] = ContinuousRange(low=0.00005, high=0.001, log=True)
    phase_c_lr: Union[List[float], ContinuousRange] = ContinuousRange(low=0.000005, high=0.0001, log=True)
    unfreezing_epochs: List[int] = [3, 5, 8]


class AblationVariantConfig(BaseModel):
    """Configuration for a single ablation variant."""

    name: str
    description: str
    remove: Optional[str] = None
    replace: Optional[str] = None
    override: Optional[Dict[str, Any]] = None


class QuickTrainConfig(BaseModel):
    """Reduced training settings for fast evaluation during search."""

    epochs: int = 3
    validation_split: float = 0.2
    early_stopping_patience: int = 2
    dense_units: int = 64
    dense_activation: str = "relu"
    head_dropout_rate: float = 0.3


class MultiSeedConfig(BaseModel):
    """Configuration for multi-seed validation of top-K configs."""

    enabled: bool = True
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
    search_strategy: str = "bayesian"
    max_trials: int = 30
    search_metric: str = "attack_f1"
    search_direction: str = "maximize"
    search_space: SearchSpaceConfig = SearchSpaceConfig()

    # Hyperband pruning
    pruning_enabled: bool = True
    min_resource: int = 1
    reduction_factor: int = 3

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

    @field_validator("search_strategy")
    @classmethod
    def _valid_strategy(cls, v: str) -> str:
        if v not in ("grid", "random", "bayesian"):
            raise ValueError(f"search_strategy must be 'grid', 'random', or 'bayesian', got '{v}'")
        return v

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
            "attack_f1", "attack_recall", "attack_precision", "macro_f1",
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
        pruning = search.get("pruning", {})
        multi_seed_raw = raw.get("multi_seed", {})

        space = _parse_search_space(space_raw)

        variants = [
            AblationVariantConfig(**v) for v in ablation.get("variants", [])
        ]

        head_cfg = quick.get("classification_head", {})
        qt = QuickTrainConfig(
            epochs=quick.get("epochs", 3),
            validation_split=quick.get("validation_split", 0.2),
            early_stopping_patience=quick.get("early_stopping_patience", 2),
            dense_units=head_cfg.get("dense_units", 64),
            dense_activation=head_cfg.get("dense_activation", "relu"),
            head_dropout_rate=head_cfg.get("dropout_rate", 0.3),
        )

        ms = MultiSeedConfig(
            enabled=multi_seed_raw.get("enabled", True),
            top_k=multi_seed_raw.get("top_k", 3),
            seeds=multi_seed_raw.get("seeds", [42, 123, 456, 789, 1024]),
            full_epochs=multi_seed_raw.get("full_epochs", 10),
        )

        return cls(
            phase1_train=Path(data.get("phase1_train", "")),
            phase1_test=Path(data.get("phase1_test", "")),
            phase2_config=Path(data.get("phase2_config", "")),
            label_column=data.get("label_column", "Label"),
            search_strategy=search.get("strategy", "bayesian"),
            max_trials=search.get("max_trials", 30),
            search_metric=search.get("metric", "f1_score"),
            search_direction=search.get("direction", "maximize"),
            search_space=space,
            pruning_enabled=pruning.get("enabled", True),
            min_resource=pruning.get("min_resource", 1),
            reduction_factor=pruning.get("reduction_factor", 3),
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


def _parse_search_space(raw: Dict[str, Any]) -> SearchSpaceConfig:
    """Parse search space, handling both list and continuous range formats."""
    parsed: Dict[str, Any] = {}
    for key, value in raw.items():
        if isinstance(value, dict) and "low" in value and "high" in value:
            parsed[key] = ContinuousRange(**value)
        else:
            parsed[key] = value
    return SearchSpaceConfig(**parsed) if parsed else SearchSpaceConfig()
