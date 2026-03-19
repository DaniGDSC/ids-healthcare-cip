"""Phase 4 risk-adaptive engine package — SOLID-architected pipeline.

Public API
----------
BaseDetector                — abstract detector interface
RiskLevel                   — five-level risk enum
Phase4Config                — pydantic-validated configuration
KScheduleEntry              — time-of-day k(t) schedule entry
Phase3ArtifactReader        — loads Phase 2/3 outputs (SHA-256 verified)
BaselineComputer            — Median + k*MAD baseline computation
DynamicThresholdUpdater     — rolling window threshold updates
ConceptDriftDetector        — baseline-relative drift detection
ThresholdFallbackManager    — lock/resume threshold fallback
CrossModalFusionDetector    — biometric + network anomaly fusion
RiskScorer                  — five-level risk classification
RiskAdaptiveExporter        — artifact export (JSON, CSV)
RiskAdaptivePipeline        — orchestrates all steps
render_risk_adaptive_report — generates §7.1 Markdown
"""

from .artifact_reader import Phase3ArtifactReader
from .base import BaseDetector
from .baseline import BaselineComputer
from .config import KScheduleEntry, Phase4Config
from .cross_modal import CrossModalFusionDetector
from .drift_detector import ConceptDriftDetector
from .dynamic_threshold import DynamicThresholdUpdater
from .exporter import RiskAdaptiveExporter
from .fallback_manager import ThresholdFallbackManager
from .pipeline import RiskAdaptivePipeline
from .report import render_risk_adaptive_report
from .risk_level import RiskLevel
from .risk_scorer import RiskScorer

__all__ = [
    "BaseDetector",
    "RiskLevel",
    "Phase4Config",
    "KScheduleEntry",
    "Phase3ArtifactReader",
    "BaselineComputer",
    "DynamicThresholdUpdater",
    "ConceptDriftDetector",
    "ThresholdFallbackManager",
    "CrossModalFusionDetector",
    "RiskScorer",
    "RiskAdaptiveExporter",
    "RiskAdaptivePipeline",
    "render_risk_adaptive_report",
]
