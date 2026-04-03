"""Export layer for Phase 0 analysis artifacts.

Design (Open/Closed + Dependency Inversion)
-------------------------------------------
``BaseExporter`` is an abstract contract.  New output formats (HDF5, Feather,
Arrow IPC, …) are added by subclassing — existing exporters are never touched.

``ReportExporter`` orchestrates artifact writes.  Its concrete exporters are
injected via the constructor, making it fully testable with fakes/mocks and
decoupled from the underlying serialisation format.

Classes
-------
BaseExporter         — ABC defining the export(data, path) contract
JsonExporter         — writes a Python dict as indented JSON
CsvExporter          — writes a DataFrame as CSV (no index)
ParquetExporter      — writes a DataFrame as Parquet via pyarrow
MarkdownExporter     — writes a string to a UTF-8 .md file
ReportExporter       — orchestrator; accepts injected BaseExporter instances
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, List, Tuple

import pandas as pd

from .config import Phase0Config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base (Open/Closed Principle)
# ---------------------------------------------------------------------------


class BaseExporter(ABC):
    """Abstract contract for all artifact exporters.

    Subclass this to add a new output format without modifying any existing
    exporter or the ``ReportExporter`` orchestrator.

    Example::

        class FeatherExporter(BaseExporter):
            def export(self, data: pd.DataFrame, path: Path) -> None:
                path.parent.mkdir(parents=True, exist_ok=True)
                data.to_feather(path)
    """

    @abstractmethod
    def export(self, data: Any, path: Path) -> None:
        """Persist *data* to *path*.

        Args:
            data: Artifact to write (concrete type defined by each subclass).
            path: Destination file path.  Parent directories are created
                  automatically by each concrete implementation.

        Raises:
            IOError: On write failure.
        """


# ---------------------------------------------------------------------------
# Concrete exporters
# ---------------------------------------------------------------------------


class JsonExporter(BaseExporter):
    """Serialize a Python ``dict`` to an indented UTF-8 JSON file.

    Args:
        indent: JSON indentation width (default: 2).
    """

    def __init__(self, indent: int = 2) -> None:
        self._indent = indent

    def export(self, data: dict, path: Path) -> None:
        """Write *data* as JSON to *path*.

        Args:
            data: Serialisable Python dict.
            path: Destination ``.json`` file.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=self._indent))
        logger.info("JSON exported → %s", path)


class CsvExporter(BaseExporter):
    """Write a pandas DataFrame to a CSV file without the row index."""

    def export(self, data: pd.DataFrame, path: Path) -> None:
        """Write *data* as CSV to *path*.

        Args:
            data: DataFrame to serialise.
            path: Destination ``.csv`` file.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(path, index=False)
        logger.info("CSV exported → %s  (%d rows)", path, len(data))


class ParquetExporter(BaseExporter):
    """Write a pandas DataFrame to Parquet format.

    Requires ``pyarrow`` or ``fastparquet`` to be installed.
    """

    def export(self, data: pd.DataFrame, path: Path) -> None:
        """Write *data* as Parquet to *path*.

        Args:
            data: DataFrame to serialise.
            path: Destination ``.parquet`` file.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        data.to_parquet(path, index=False)
        logger.info("Parquet exported → %s  (%d rows)", path, len(data))


class MarkdownExporter(BaseExporter):
    """Write a plain-text string to a UTF-8 Markdown file."""

    def export(self, data: str, path: Path) -> None:
        """Write *data* as Markdown to *path*.

        Args:
            data: Markdown content string.
            path: Destination ``.md`` file.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(data, encoding="utf-8")
        logger.info("Markdown exported → %s", path)


# ---------------------------------------------------------------------------
# Orchestrator (Dependency Inversion)
# ---------------------------------------------------------------------------


class ReportExporter:
    """Orchestrate export of all Phase 0 analysis artifacts.

    Concrete exporters are injected via the constructor rather than
    instantiated internally, enabling substitution with fakes in tests
    and extension with new formats without changing this class.

    Args:
        config: Validated ``Phase0Config`` providing output directory and
                artifact filenames.
        json_exporter: Exporter for the JSON stats report.
                       Defaults to ``JsonExporter()``.
        csv_exporter: Exporter for the high-correlations CSV.
                      Defaults to ``CsvExporter()``.
        parquet_exporter: Exporter for the correlation matrix Parquet.
                          Defaults to ``ParquetExporter()``.

    Example::

        exporter = ReportExporter(config)
        exporter.export_stats_report(descriptive, missing, class_dist)
        exporter.export_high_correlations(pairs)
        exporter.export_correlation_matrix(matrix)
    """

    def __init__(
        self,
        config: Phase0Config,
        json_exporter: BaseExporter | None = None,
        csv_exporter: BaseExporter | None = None,
        parquet_exporter: BaseExporter | None = None,
        markdown_exporter: BaseExporter | None = None,
    ) -> None:
        self._config = config
        self._json = json_exporter or JsonExporter()
        self._csv = csv_exporter or CsvExporter()
        self._parquet = parquet_exporter or ParquetExporter()
        self._markdown = markdown_exporter or MarkdownExporter()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def export_stats_report(
        self,
        descriptive: dict,
        missing: dict,
        class_dist: dict,
    ) -> None:
        """Export descriptive stats, missing values, and class distribution.

        Combines the three analysis dicts into a single JSON report.

        Args:
            descriptive: Output of ``StatisticsAnalyzer.descriptive_stats()``.
            missing:     Output of ``StatisticsAnalyzer.missing_values()``.
            class_dist:  Output of ``StatisticsAnalyzer.class_distribution()``.
        """
        report = {
            "descriptive_statistics": descriptive,
            "missing_values":         missing,
            "class_distribution":     class_dist,
        }
        path = self._config.output_dir / self._config.stats_report_file
        self._json.export(report, path)

    def export_high_correlations(
        self,
        pairs: List[Tuple[str, str, float]],
    ) -> None:
        """Export high-correlation feature pairs to CSV.

        Columns written: ``feature_a``, ``feature_b``, ``correlation``.

        Args:
            pairs: Output of ``CorrelationAnalyzer.high_correlation_pairs()``.
        """
        df = pd.DataFrame(
            pairs, columns=["feature_a", "feature_b", "correlation"]
        )
        path = self._config.output_dir / self._config.high_correlations_file
        self._csv.export(df, path)

    def export_correlation_matrix(self, matrix: pd.DataFrame) -> None:
        """Export the full Pearson correlation matrix to Parquet.

        The feature-name index is reset to a regular column (``index``)
        so the Parquet file is self-describing without a separate schema.

        Args:
            matrix: Output of ``CorrelationAnalyzer.correlation_matrix()``.
        """
        path = self._config.output_dir / self._config.correlation_matrix_file
        self._parquet.export(matrix.reset_index(), path)

    def export_quality_report(self, content: str) -> None:
        """Export a Markdown data-quality report for the thesis manuscript.

        Args:
            content: Fully rendered Markdown string.
        """
        path = self._config.output_dir / self._config.quality_report_file
        self._markdown.export(content, path)
