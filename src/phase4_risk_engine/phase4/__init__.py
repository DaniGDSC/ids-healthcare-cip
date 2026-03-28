"""Phase 4 risk-adaptive engine package — SOLID-architected pipeline.

Public API
----------
BaseDetector                — abstract detector interface
RiskLevel                   — five-level risk enum
Phase4Config                — pydantic-validated configuration
KScheduleEntry              — time-of-day k(t) schedule entry
Phase3ArtifactReader        — loads Phase 2/3 outputs (SHA-256 verified)
AttentionAnomalyDetector     — novelty/zero-day detection via attention distance
CIAThreatMapper             — maps attack types to CIA threat vectors
DeviceRegistry              — device profiles with FDA class + CIA priorities
CIARiskModifier             — adaptive CIA priority shifting by scenario
ClinicalImpactAssessor      — translates risk into clinical actions + protocols
ConditionalExplainer        — severity-based explainability (saves compute on NORMAL)
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

from .alert_fatigue import AlertFatigueManager
from .artifact_reader import Phase3ArtifactReader
from .attention_anomaly import AttentionAnomalyDetector
from .base import BaseDetector
from .baseline import BaselineComputer
from .cia_risk_modifier import CIARiskModifier, CIARiskAssessment, Scenario
from .cia_threat_mapper import CIAThreatMapper, CIAThreatVector
from .clinical_impact import ClinicalImpactAssessor, ClinicalSeverity, ClinicalAssessment, ResponseProtocol
from .cognitive_translator import CognitiveTranslator
from .conditional_explainer import ConditionalExplainer
from .device_registry import DeviceRegistry, DeviceProfile, CIAPriority
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
    "AlertFatigueManager",
    "AttentionAnomalyDetector",
    "CIAThreatMapper",
    "CIAThreatVector",
    "DeviceRegistry",
    "DeviceProfile",
    "CIAPriority",
    "CIARiskModifier",
    "CIARiskAssessment",
    "Scenario",
    "ClinicalImpactAssessor",
    "ClinicalSeverity",
    "ClinicalAssessment",
    "ResponseProtocol",
    "CognitiveTranslator",
    "ConditionalExplainer",
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
