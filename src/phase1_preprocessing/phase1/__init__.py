"""Phase 1 preprocessing package — SOLID-architected pipeline.

Public API
----------
Phase1Config             — pydantic-validated configuration
Phase0ArtifactReader     — reads Phase 0 outputs (DI)
HIPAASanitizer           — drops HIPAA identifier columns
MissingValueHandler      — ffill biometrics, dropna network
RedundancyRemover        — drops correlated features from Phase 0
VarianceFilter           — drops zero/near-zero variance features
DataSplitter             — stratified 70/30 train/test
SMOTEBalancer            — oversamples minority class (train only)
RobustScalerTransformer  — fit on train, transform both
PreprocessingExporter    — Parquet, pickle, JSON export
PreprocessingPipeline    — orchestrates all steps
render_preprocessing_report — generates §4.1 Markdown
"""

from .artifact_reader import Phase0ArtifactReader
from .base import BaseTransformer
from .config import Phase1Config
from .exporter import PreprocessingExporter
from .hipaa import HIPAASanitizer
from .missing import MissingValueHandler
from .pipeline import PreprocessingPipeline
from .redundancy import RedundancyRemover
from .report import render_preprocessing_report
from .variance import VarianceFilter
from .scaler import RobustScalerTransformer
from .smote import SMOTEBalancer
from .splitter import DataSplitter

__all__ = [
    "BaseTransformer",
    "Phase1Config",
    "Phase0ArtifactReader",
    "HIPAASanitizer",
    "MissingValueHandler",
    "RedundancyRemover",
    "VarianceFilter",
    "DataSplitter",
    "SMOTEBalancer",
    "RobustScalerTransformer",
    "PreprocessingExporter",
    "PreprocessingPipeline",
    "render_preprocessing_report",
]
