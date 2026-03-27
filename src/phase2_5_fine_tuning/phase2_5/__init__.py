"""Phase 2.5 fine-tuning & ablation package — SOLID-architected pipeline.

Public API
----------
Phase2_5Config             — pydantic-validated configuration
SearchSpaceConfig          — hyperparameter search space definition
ContinuousRange            — log-scale continuous parameter range
MultiSeedConfig            — multi-seed validation configuration
HyperparameterTuner       — systematic hyperparameter search (grid/random/bayesian)
AblationRunner             — component ablation study
QuickEvaluator             — fast train+eval for tuning iterations
MultiSeedValidator         — retrain top-K with multiple seeds
TuningExporter             — export results, best config, metadata
TuningPipeline             — orchestrates all steps
render_tuning_report       — generates tuning report Markdown
compute_importance         — parameter importance analysis
"""

from .ablation import AblationRunner
from .config import ContinuousRange, MultiSeedConfig, Phase2_5Config, SearchSpaceConfig
from .evaluator import QuickEvaluator
from .exporter import TuningExporter
from .importance import compute_importance
from .multi_seed import MultiSeedValidator
from .pipeline import TuningPipeline
from .report import render_tuning_report
from .search_space import SearchSpace
from .tuner import HyperparameterTuner

__all__ = [
    "Phase2_5Config",
    "SearchSpaceConfig",
    "ContinuousRange",
    "MultiSeedConfig",
    "SearchSpace",
    "HyperparameterTuner",
    "AblationRunner",
    "QuickEvaluator",
    "MultiSeedValidator",
    "TuningExporter",
    "TuningPipeline",
    "render_tuning_report",
    "compute_importance",
]
