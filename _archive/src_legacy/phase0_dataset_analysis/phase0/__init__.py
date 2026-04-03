"""Phase 0 analysis package — SOLID-architected EDA for WUSTL-EHMS-2020.

Public API
----------
Phase0Config        — validated configuration dataclass loaded from YAML
DataLoader          — load and validate the raw CSV dataset
StatisticsAnalyzer  — descriptive stats, missing values, class distribution
CorrelationAnalyzer — Pearson correlation matrix and high-correlation pairs
OutlierAnalyzer     — IQR-based outlier detection per feature
ReportExporter      — orchestrates JSON / CSV / Parquet / Markdown export
render_quality_report        — generates data-quality Markdown for thesis defence
render_reproducibility_report — generates reproducibility Markdown (§3.4)
"""

from .config import Phase0Config
from .loader import DataLoader
from .analyzer import CorrelationAnalyzer, OutlierAnalyzer, StatisticsAnalyzer
from .exporter import ReportExporter
from .quality_report import render_quality_report
from .reproducibility_report import render_reproducibility_report

__all__ = [
    "Phase0Config",
    "DataLoader",
    "StatisticsAnalyzer",
    "CorrelationAnalyzer",
    "OutlierAnalyzer",
    "ReportExporter",
    "render_quality_report",
    "render_reproducibility_report",
]
