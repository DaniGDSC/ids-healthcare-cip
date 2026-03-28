"""Phase 2.5 fine-tuning & ablation package — SOLID-architected pipeline.

Public API
----------
Phase2_5Config             — pydantic-validated configuration
SearchSpaceConfig          — 5 core parameter ranges
MultiSeedConfig            — multi-seed validation configuration
HyperparameterTuner       — Bayesian TPE search (Optuna)
AblationRunner             — component ablation study
QuickEvaluator             — two-stage train+eval
MultiSeedValidator         — retrain top-K with multiple seeds
TuningExporter             — export results, best config, metadata
TuningPipeline             — orchestrates all steps
render_tuning_report       — generates tuning report Markdown
compute_importance         — parameter importance analysis (fANOVA)
"""

from .ablation import AblationRunner
from .config import MultiSeedConfig, Phase2_5Config, SearchSpaceConfig
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
