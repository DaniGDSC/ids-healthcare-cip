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

from common.phi import BIOMETRIC_COLUMNS

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
        """Compute descriptive statistics per numeric feature.

        Non-numeric columns are silently ignored.  NaN values are excluded
        from each per-column computation before aggregation.

        Network features get the full ``{mean, median, std, min, max}``.
        Biometric features (see ``common.phi.BIOMETRIC_COLUMNS``) are
        restricted to ``{mean, std}``: per-patient minima and maxima are
        quasi-identifiers under HIPAA Safe Harbor and are never published
        from this layer.

        Returns:
            Nested dict mapping ``feature_name → stats dict``. Stats values
            are rounded to six decimal places.
        """
        numeric_df = self._df.select_dtypes(include="number")

        # Partition columns once — O(F), avoids re-testing membership later.
        bio_cols = [c for c in numeric_df.columns if c in BIOMETRIC_COLUMNS]
        net_cols = [c for c in numeric_df.columns if c not in BIOMETRIC_COLUMNS]

        stats: Dict[str, Dict[str, float]] = {}

        if net_cols:
            net_agg = (
                numeric_df[net_cols]
                .agg(["mean", "median", "std", "min", "max"])
                .round(6)
            )
            for col in net_cols:
                v = net_agg[col]
                stats[col] = {
                    "mean":   float(v["mean"]),
                    "median": float(v["median"]),
                    "std":    float(v["std"]),
                    "min":    float(v["min"]),
                    "max":    float(v["max"]),
                }

        if bio_cols:
            # Population-level statistics only — no min/max/median.
            # See common/phi.py for the canonical column set.
            bio_agg = (
                numeric_df[bio_cols]
                .agg(["mean", "std"])
                .round(6)
            )
            for col in bio_cols:
                v = bio_agg[col]
                stats[col] = {
                    "mean": float(v["mean"]),
                    "std":  float(v["std"]),
                }

        logger.info(
            "Descriptive stats computed for %d numeric features "
            "(%d biometric channels published as mean/std only)",
            len(stats),
            len(bio_cols),
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
        warn_pct = self._config.missing_value_warn_pct

        null_counts = self._df.isna().sum()
        for col, n in null_counts[null_counts > 0].items():
            n = int(n)
            pct = round(n / total * 100, 4)
            result[col] = {"count": n, "percentage": pct}
            if pct > warn_pct:
                logger.warning(
                    "Feature '%s' has %.2f%% missing values (threshold: %.1f%%)",
                    col,
                    pct,
                    warn_pct,
                )

        logger.info(
            "Missing values: %d / %d features affected",
            len(result),
            len(self._df.columns),
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
                "count": count,
                "percentage": round(count / total * 100, 4),
            }

        logger.info(
            "Class distribution → Normal: %d (%.1f%%)  |  Attack: %d (%.1f%%)",
            dist["Normal"]["count"],
            dist["Normal"]["percentage"],
            dist["Attack"]["count"],
            dist["Attack"]["percentage"],
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
        cols_arr = np.array(matrix.columns.tolist())
        vals = matrix.values  # F×F float64 view — no copy

        rows_idx, cols_idx = np.triu_indices(len(cols_arr), k=1)
        r_vals = vals[rows_idx, cols_idx]

        # Vectorised filter: not NaN AND |r| > threshold.
        mask = (~np.isnan(r_vals)) & (np.abs(r_vals) > threshold)
        pairs: List[Tuple[str, str, float]] = sorted(
            zip(
                cols_arr[rows_idx[mask]].tolist(),
                cols_arr[cols_idx[mask]].tolist(),
                np.round(r_vals[mask], 6).tolist(),
            ),
            key=lambda t: abs(t[2]),
            reverse=True,
        )

        logger.info(
            "High-correlation pairs (|r| > %.2f): %d found",
            threshold,
            len(pairs),
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

        Network features get the full record (q1/q3/iqr/lower_bound/
        upper_bound/outlier_count/outlier_pct/total).

        Biometric features (see ``common.phi.BIOMETRIC_COLUMNS``) only
        publish ``outlier_count``/``outlier_pct``/``total``: q1/q3 and the
        derived fences are quantile-based quasi-identifiers under HIPAA
        Safe Harbor and must not leave this layer.

        Returns:
            List of dicts, one per numeric feature, sorted by descending
            ``outlier_pct``.
        """
        numeric_df = self._df.select_dtypes(include="number")
        k = self._config.outlier_iqr_multiplier
        total = len(numeric_df)
        report: List[Dict[str, Any]] = []

        quantiles = numeric_df.quantile([0.25, 0.75])
        q1_all = quantiles.loc[0.25]
        q3_all = quantiles.loc[0.75]
        iqr_all   = q3_all - q1_all
        lower_all = q1_all - k * iqr_all
        upper_all = q3_all + k * iqr_all

        outlier_counts = ((numeric_df < lower_all) | (numeric_df > upper_all)).sum()

        n_with = 0
        for col in numeric_df.columns:
            q1    = float(q1_all[col])
            q3    = float(q3_all[col])
            iqr   = float(iqr_all[col])
            lower = float(lower_all[col])
            upper = float(upper_all[col])
            outliers = int(outlier_counts[col])
            pct = round(outliers / total * 100, 4) if total else 0.0

            if outliers:
                n_with += 1

            if col in BIOMETRIC_COLUMNS:
                # Aggregate counts only — no quantile-derived fields (PHI).
                report.append(
                    {
                        "feature": col,
                        "outlier_count": outliers,
                        "outlier_pct": pct,
                        "total": total,
                    }
                )
            else:
                report.append(
                    {
                        "feature": col,
                        "q1": round(q1, 6),
                        "q3": round(q3, 6),
                        "iqr": round(iqr, 6),
                        "lower_bound": round(lower, 6),
                        "upper_bound": round(upper, 6),
                        "outlier_count": outliers,
                        "outlier_pct": pct,
                        "total": total,
                    }
                )

        report.sort(key=lambda r: r["outlier_pct"], reverse=True)
        logger.info(
            "Outlier analysis (k=%.1f): %d / %d features have outliers",
            k,
            n_with,
            len(report),
        )
        return report
