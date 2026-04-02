"""Phase 1 preprocessing package.

Public API
----------
Phase1Config             — pydantic-validated configuration
Phase0ArtifactReader     — reads Phase 0 outputs (DI)
HIPAASanitizer           — drops identifier columns
CategoricalEncoder       — label-encodes categoricals, parses string numerics
MissingValueHandler      — ffill biometrics, fill_zero network
VarianceFilter           — drops zero/near-zero variance features
RedundancyRemover        — drops correlated features from Phase 0
SHAPSelector             — RFECV + SHAP feature selection (train only)
RobustScalerTransformer  — fit on train, transform both
PreprocessingExporter    — Parquet, pickle, JSON export
PreprocessingPipeline    — orchestrates all steps
render_preprocessing_report — generates §4.1 Markdown

Dual-track outputs:
  Track A (supervised) — X_train + y_train; SMOTE applied inside CV pipeline
  Track B (novelty)    — X_train_benign for autoencoder training
"""

from .artifact_reader import Phase0ArtifactReader
from .base import BaseTransformer
from .config import Phase1Config
from .encoder import CategoricalEncoder
from .exporter import PreprocessingExporter
from .hipaa import HIPAASanitizer
from .missing import MissingValueHandler
from .pipeline import PreprocessingPipeline
from .redundancy import RedundancyRemover
from .report import render_preprocessing_report
from .scaler import RobustScalerTransformer
from .shap_selector import SHAPSelector
from .smote import SMOTEBalancer
from .splitter import DataSplitter
from .variance import VarianceFilter

__all__ = [
    "BaseTransformer",
    "Phase1Config",
    "Phase0ArtifactReader",
    "HIPAASanitizer",
    "CategoricalEncoder",
    "MissingValueHandler",
    "VarianceFilter",
    "RedundancyRemover",
    "SHAPSelector",
    "RobustScalerTransformer",
    "SMOTEBalancer",
    "DataSplitter",
    "PreprocessingExporter",
    "PreprocessingPipeline",
    "render_preprocessing_report",
]
