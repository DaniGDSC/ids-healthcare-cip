"""Phase 3 classification engine package — SOLID-architected pipeline.

Public API
----------
BaseClassificationHead     — abstract head interface
Phase3Config               — pydantic-validated configuration
TrainingPhaseConfig        — per-phase training configuration
CrossDatasetConfig         — cross-dataset validation configuration
Phase2ArtifactReader       — loads Phase 2 outputs (DI)
AutoClassificationHead     — Dense→Dropout→Dense head
ProgressiveUnfreezer       — freeze/unfreeze layer groups
ClassificationTrainer      — compile + fit per phase
ModelEvaluator             — test-set-only metrics
ClassificationExporter     — weights, JSON, CSV export
ClassificationPipeline     — orchestrates all steps
CICIoMTLoader              — cross-dataset data adapter (CICIoMT2024)
render_classification_report  — generates §6.1 Markdown
render_cross_dataset_report   — generates §6.2 Cross-Dataset Markdown
build_comparison_report       — per-metric delta comparison
"""

from .artifact_reader import Phase2ArtifactReader
from .base import BaseClassificationHead
from .config import CrossDatasetConfig, Phase3Config, TrainingPhaseConfig
from .cross_dataset import CICIoMTLoader
from .cross_dataset_report import build_comparison_report, render_cross_dataset_report
from .evaluator import ModelEvaluator
from .exporter import ClassificationExporter
from .head import AutoClassificationHead
from .pipeline import ClassificationPipeline
from .report import render_classification_report
from .trainer import ClassificationTrainer
from .unfreezer import ProgressiveUnfreezer

__all__ = [
    "BaseClassificationHead",
    "Phase3Config",
    "TrainingPhaseConfig",
    "CrossDatasetConfig",
    "Phase2ArtifactReader",
    "AutoClassificationHead",
    "ProgressiveUnfreezer",
    "ClassificationTrainer",
    "ModelEvaluator",
    "ClassificationExporter",
    "ClassificationPipeline",
    "CICIoMTLoader",
    "render_classification_report",
    "render_cross_dataset_report",
    "build_comparison_report",
]
