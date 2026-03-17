"""Statistical analyzers for Phase 0 EDA.

Classes
-------
StatisticsAnalyzer
    Descriptive statistics, missing-value summary, and class distribution.
    Single responsibility: summarise the *values* in the dataset.

CorrelationAnalyzer
    Pearson correlation matrix and high-correlation pair detection.
    Single responsibility: measure *relationships* between features.

OutlierAnalyzer
    IQR-based outlier detection per numeric feature.
    Single responsibility: identify distributional extremes.

All classes receive their DataFrame and configuration via the constructor
(Dependency Inversion) and contain no file I/O.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from .config import Phase0Config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# StatisticsAnalyzer
# ---------------------------------------------------------------------------


class StatisticsAnalyzer:
    """Compute descriptive statistics, missing values, and class distribution.

    Args:
        df: Loaded (and validated) dataset DataFrame.
        config: Validated ``Phase0Config`` supplying label column and thresholds.

    Example::

        analyzer = StatisticsAnalyzer(df, config)
        stats   = analyzer.descriptive_stats()
        missing = analyzer.missing_values()
        dist    = analyzer.class_distribution()
    """

    def __init__(self, df: pd.DataFrame, config: Phase0Config) -> None:
        self._df = df
        self._config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def descriptive_stats(self) -> Dict[str, Dict[str, float]]:
        """Compute mean, median, std, min, and max per numeric feature.

        Non-numeric columns are silently ignored.  NaN values are excluded
        from each per-column computation before aggregation.

        Returns:
            Nested dict mapping ``feature_name → {mean, median, std, min, max}``.
            All values are rounded to six decimal places.
        """
        numeric_df = self._df.select_dtypes(include="number")
        stats: Dict[str, Dict[str, float]] = {}

        for col in numeric_df.columns:
            series = numeric_df[col].dropna()
            stats[col] = {
                "mean":   round(float(series.mean()),   6),
                "median": round(float(series.median()), 6),
                "std":    round(float(series.std()),    6),
                "min":    round(float(series.min()),    6),
                "max":    round(float(series.max()),    6),
            }

        logger.info(
            "Descriptive stats computed for %d numeric features", len(stats)
        )
        return stats

    def missing_values(self) -> Dict[str, Dict[str, float]]:
        """Count and quantify missing values per feature.

        Logs a ``WARNING`` for any feature whose missing percentage exceeds
        ``config.missing_value_warn_pct``.  Features with no missing values
        are omitted from the returned dict to keep the report concise.

        Returns:
            Dict mapping ``feature_name → {count, percentage}`` for features
            with at least one missing value.
        """
        total = len(self._df)
        result: Dict[str, Dict[str, float]] = {}

        for col in self._df.columns:
            n = int(self._df[col].isna().sum())
            if n == 0:
                continue

            pct = round(n / total * 100, 4)
            result[col] = {"count": n, "percentage": pct}

            if pct > self._config.missing_value_warn_pct:
                logger.warning(
                    "Feature '%s' has %.2f%% missing values "
                    "(threshold: %.1f%%)",
                    col, pct, self._config.missing_value_warn_pct,
                )

        logger.info(
            "Missing values: %d / %d features affected",
            len(result), len(self._df.columns),
        )
        return result

    def class_distribution(self) -> Dict[str, Dict[str, float]]:
        """Compute Normal vs Attack sample counts and class percentages.

        Returns:
            Dict with keys ``"Normal"`` and ``"Attack"``, each mapping to
            ``{count, percentage}``.

        Raises:
            KeyError: If ``config.label_column`` is absent from the DataFrame.
        """
        label_col = self._config.label_column
        if label_col not in self._df.columns:
            msg = f"Label column '{label_col}' not found in DataFrame"
            logger.error(msg)
            raise KeyError(msg)

        total = len(self._df)
        value_counts = self._df[label_col].value_counts()
        label_map: Dict[int, str] = {0: "Normal", 1: "Attack"}

        dist: Dict[str, Dict[str, float]] = {}
        for code, name in label_map.items():
            count = int(value_counts.get(code, 0))
            dist[name] = {
                "count":      count,
                "percentage": round(count / total * 100, 4),
            }

        logger.info(
            "Class distribution → Normal: %d (%.1f%%)  |  Attack: %d (%.1f%%)",
            dist["Normal"]["count"], dist["Normal"]["percentage"],
            dist["Attack"]["count"], dist["Attack"]["percentage"],
        )
        return dist


# ---------------------------------------------------------------------------
# CorrelationAnalyzer
# ---------------------------------------------------------------------------


class CorrelationAnalyzer:
    """Compute Pearson correlation matrix and identify high-correlation pairs.

    The correlation matrix is computed lazily on first access and cached for
    subsequent calls, avoiding redundant O(d²·n) work.

    Args:
        df: Loaded (and validated) dataset DataFrame.
        config: Validated ``Phase0Config`` supplying the correlation threshold.

    Example::

        analyzer = CorrelationAnalyzer(df, config)
        matrix = analyzer.correlation_matrix()
        pairs  = analyzer.high_correlation_pairs()
    """

    def __init__(self, df: pd.DataFrame, config: Phase0Config) -> None:
        self._df = df
        self._config = config
        self._matrix: pd.DataFrame | None = None  # lazy cache

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def correlation_matrix(self) -> pd.DataFrame:
        """Return the Pearson correlation matrix for all numeric features.

        The matrix is computed once and cached; subsequent calls are O(1).

        Returns:
            Square DataFrame indexed and columned by numeric feature names,
            values in [-1, 1].
        """
        if self._matrix is None:
            numeric_df = self._df.select_dtypes(include="number")
            self._matrix = numeric_df.corr(method="pearson")
            logger.info(
                "Correlation matrix computed: %d × %d",
                *self._matrix.shape,
            )
        return self._matrix

    def high_correlation_pairs(self) -> List[Tuple[str, str, float]]:
        """Identify numeric feature pairs with |r| > ``config.correlation_threshold``.

        Only the upper triangle of the correlation matrix is scanned so each
        pair is reported exactly once.  NaN entries (e.g. zero-variance
        features) are silently skipped.

        Returns:
            List of ``(feature_a, feature_b, correlation)`` tuples sorted by
            descending ``|correlation|``.
        """
        matrix = self.correlation_matrix()
        threshold = self._config.correlation_threshold
        cols = matrix.columns.tolist()

        pairs: List[Tuple[str, str, float]] = []
        for i, col_a in enumerate(cols):
            for col_b in cols[i + 1:]:
                r = matrix.loc[col_a, col_b]
                if not np.isnan(r) and abs(r) > threshold:
                    pairs.append((col_a, col_b, round(float(r), 6)))

        pairs.sort(key=lambda t: abs(t[2]), reverse=True)
        logger.info(
            "High-correlation pairs (|r| > %.2f): %d found",
            threshold, len(pairs),
        )
        return pairs


# ---------------------------------------------------------------------------
# OutlierAnalyzer
# ---------------------------------------------------------------------------


class OutlierAnalyzer:
    """IQR-based outlier detection per numeric feature.

    An observation is classified as an outlier if it falls outside
    ``[Q1 - k*IQR, Q3 + k*IQR]`` where *k* = ``config.outlier_iqr_multiplier``.

    Args:
        df: Loaded (and validated) dataset DataFrame.
        config: Validated ``Phase0Config`` supplying the IQR multiplier.

    Example::

        analyzer = OutlierAnalyzer(df, config)
        report = analyzer.outlier_report()
    """

    def __init__(self, df: pd.DataFrame, config: Phase0Config) -> None:
        self._df = df
        self._config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def outlier_report(self) -> List[Dict[str, Any]]:
        """Compute IQR-based outlier statistics for every numeric feature.

        Returns:
            List of dicts, one per numeric feature, each containing:
            ``{feature, q1, q3, iqr, lower_bound, upper_bound,
              outlier_count, outlier_pct, total}``.
            Sorted by descending ``outlier_pct``.
        """
        numeric_df = self._df.select_dtypes(include="number")
        k = self._config.outlier_iqr_multiplier
        total = len(numeric_df)
        report: List[Dict[str, Any]] = []

        for col in numeric_df.columns:
            series = numeric_df[col].dropna()
            q1 = float(series.quantile(0.25))
            q3 = float(series.quantile(0.75))
            iqr = q3 - q1
            lower = q1 - k * iqr
            upper = q3 + k * iqr

            outliers = int(((series < lower) | (series > upper)).sum())
            pct = round(outliers / total * 100, 4) if total else 0.0

            report.append({
                "feature":      col,
                "q1":           round(q1, 6),
                "q3":           round(q3, 6),
                "iqr":          round(iqr, 6),
                "lower_bound":  round(lower, 6),
                "upper_bound":  round(upper, 6),
                "outlier_count": outliers,
                "outlier_pct":  pct,
                "total":        total,
            })

        report.sort(key=lambda r: r["outlier_pct"], reverse=True)
        n_with = sum(1 for r in report if r["outlier_count"] > 0)
        logger.info(
            "Outlier analysis (k=%.1f): %d / %d features have outliers",
            k, n_with, len(report),
        )
        return report
