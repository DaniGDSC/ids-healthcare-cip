"""Phase 0 analysis package — SOLID-architected EDA for WUSTL-EHMS-2020.

Public API
----------
Phase0Config        — validated configuration dataclass loaded from YAML
ConfigError         — structural config-file failure (separate from ValueError)
DataLoader          — load and validate the raw CSV dataset
StatisticsAnalyzer  — descriptive stats, missing values, class distribution
CorrelationAnalyzer — Pearson correlation matrix and high-correlation pairs
OutlierAnalyzer     — IQR-based outlier detection per feature
ReportExporter      — orchestrates JSON / CSV / Parquet / Markdown export
render_quality_report        — generates data-quality Markdown for thesis defence
render_reproducibility_report — generates reproducibility Markdown (§3.4)

Security controls (wired into DataLoader.load and Phase0Config.from_yaml):
IntegrityVerifier   — signed SHA-256 baseline; no auto-baseline footgun
IntegrityError      — raised on missing baseline / hash mismatch / forged metadata
PathValidator       — workspace containment, optional read-only enforcement
ColumnAllowlist     — column-name allowlist for required_columns
log_phase0_event    — routes audit events through Module 5's signed chain
"""

from .config import ConfigError, Phase0Config
from .loader import DataLoader
from .analyzer import CorrelationAnalyzer, OutlierAnalyzer, StatisticsAnalyzer
from .exporter import ReportExporter
from .quality_report import render_quality_report
from .reproducibility_report import render_reproducibility_report
from .security import (
    ColumnAllowlist,
    IntegrityError,
    IntegrityVerifier,
    PathValidator,
    log_phase0_event,
)

__all__ = [
    "Phase0Config",
    "ConfigError",
    "DataLoader",
    "StatisticsAnalyzer",
    "CorrelationAnalyzer",
    "OutlierAnalyzer",
    "ReportExporter",
    "render_quality_report",
    "render_reproducibility_report",
    "IntegrityVerifier",
    "IntegrityError",
    "PathValidator",
    "ColumnAllowlist",
    "log_phase0_event",
]
