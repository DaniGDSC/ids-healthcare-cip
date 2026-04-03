"""Phase 5 explanation engine package — SOLID-architected pipeline.

Public API
----------
BaseVisualizer              — abstract visualizer interface
Phase5Config                — pydantic-validated configuration
ExplanationTemplates        — nested config model for templates
Phase4ArtifactReader        — loads Phase 2/3/4 outputs (SHA-256 verified)
AlertFilter                 — filters non-NORMAL risk samples
SHAPComputer                — SHAP GradientExplainer + IG fallback
FeatureImportanceRanker     — global feature importance ranking
ContextEnricher             — per-sample SHAP context enrichment
ExplanationGenerator        — template-based explanation text
WaterfallVisualizer         — waterfall chart (BaseVisualizer)
BarChartVisualizer          — bar chart (BaseVisualizer)
LineGraphVisualizer         — timeline chart (BaseVisualizer)
ExplanationExporter         — artifact export (parquet, JSON)
ExplanationPipeline         — orchestrates all steps
render_explanation_report   — generates section 8.1 Markdown
"""

from .alert_filter import AlertFilter
from .artifact_reader import Phase4ArtifactReader
from .bar_chart_visualizer import BarChartVisualizer
from .base import BaseVisualizer
from .config import ExplanationTemplates, Phase5Config
from .context_enricher import ContextEnricher
from .explanation_generator import ExplanationGenerator
from .exporter import ExplanationExporter
from .feature_importance import FeatureImportanceRanker
from .line_graph_visualizer import LineGraphVisualizer
from .pipeline import ExplanationPipeline
from .report import render_explanation_report
from .shap_computer import SHAPComputer
from .waterfall_visualizer import WaterfallVisualizer

__all__ = [
    "BaseVisualizer",
    "Phase5Config",
    "ExplanationTemplates",
    "Phase4ArtifactReader",
    "AlertFilter",
    "SHAPComputer",
    "FeatureImportanceRanker",
    "ContextEnricher",
    "ExplanationGenerator",
    "WaterfallVisualizer",
    "BarChartVisualizer",
    "LineGraphVisualizer",
    "ExplanationExporter",
    "ExplanationPipeline",
    "render_explanation_report",
]
