"""Cross-dataset loader — CICIoMT2024 to WUSTL-compatible format.

Adapts the CICIoMT2024 network traffic dataset for cross-dataset
validation against the WUSTL-trained classification model.

CICIoMT2024 lacks biometric features (Temp, SpO2, etc.), which are
imputed using median values from WUSTL Normal (y=0) training samples.

Cross-dataset disclosure:
    CICIoMT2024 biometric features imputed using WUSTL-EHMS-2020
    Normal sample medians. Scaler fitted on WUSTL train set only.
    Results reflect conservative generalization estimate.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from src.phase0_dataset_analysis.phase0.security import (
    BIOMETRIC_COLUMNS,
    AuditLogger,
)

logger = logging.getLogger(__name__)

# Canonical WUSTL feature order (29 features, verified from train_phase1.parquet)
WUSTL_FEATURE_ORDER: List[str] = [
    "SrcBytes",
    "DstBytes",
    "SrcLoad",
    "DstLoad",
    "SrcGap",
    "DstGap",
    "SIntPkt",
    "DIntPkt",
    "SIntPktAct",
    "DIntPktAct",
    "sMaxPktSz",
    "dMaxPktSz",
    "sMinPktSz",
    "dMinPktSz",
    "Dur",
    "Trans",
    "TotBytes",
    "Load",
    "pSrcLoss",
    "pDstLoss",
    "Packet_num",
    "Temp",
    "SpO2",
    "Pulse_Rate",
    "SYS",
    "DIA",
    "Heart_rate",
    "Resp_Rate",
    "ST",
]

N_FEATURES: int = 29

# HIPAA columns to drop from CICIoMT2024 (same as Phase 1)
_HIPAA_DROP_COLUMNS: List[str] = [
    "SrcAddr",
    "DstAddr",
    "Sport",
    "Dport",
    "SrcMac",
    "DstMac",
    "Dir",
    "Flgs",
]

# Cross-dataset disclosure (logged once per run)
_DISCLOSURE: str = (
    "CICIoMT2024 biometric features imputed using WUSTL-EHMS-2020 "
    "Normal sample medians. Scaler fitted on WUSTL train set only. "
    "Results reflect conservative generalization estimate."
)


class CICIoMTLoader:
    """Load and adapt CICIoMT2024 data for WUSTL-model inference.

    Steps:
        1. Load CSV from configurable path
        2. Map column names via configurable mapping dict
        3. Apply label mapping if provided
        4. Drop HIPAA-sensitive columns
        5. Impute missing biometric features with WUSTL Normal medians
        6. Reorder to canonical 29-feature order
        7. Apply pre-fitted RobustScaler (NO refit)

    Args:
        csv_path: Path to the CICIoMT2024 CSV file.
        column_mapping: Dict mapping CICIoMT2024 names to WUSTL names.
        label_column: Name of the label column in CICIoMT2024.
        label_mapping: Optional mapping of label values to int.
        scaler_path: Path to the fitted robust_scaler.pkl.
        wustl_train_path: Path to WUSTL train_phase1.parquet.
    """

    def __init__(
        self,
        csv_path: Path,
        column_mapping: Dict[str, str],
        label_column: str,
        label_mapping: Optional[Dict[str, int]],
        scaler_path: Path,
        wustl_train_path: Path,
    ) -> None:
        self._csv_path = csv_path
        self._column_mapping = column_mapping
        self._label_column = label_column
        self._label_mapping = label_mapping
        self._scaler_path = scaler_path
        self._wustl_train_path = wustl_train_path

    def is_available(self) -> bool:
        """Check whether the CICIoMT2024 CSV exists."""
        return self._csv_path.exists()

    def load_and_prepare(
        self,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
        """Execute the full load, adapt, scale pipeline.

        Returns:
            Tuple of (X_scaled, y, load_report) where X_scaled has
            shape (n_samples, 29) and is ready for reshaping.
            y is None if label column not found.

        Raises:
            FileNotFoundError: If CSV or scaler file not found.
            ValueError: If required features cannot be mapped.
        """
        logger.info("== CICIoMT2024 Cross-Dataset Loading ==")
        logger.info("  DISCLOSURE: %s", _DISCLOSURE)

        # 1. Load CSV
        df = self._load_csv()
        raw_rows = len(df)

        # 2. Map column names
        df = self._map_columns(df)

        # 3. Apply label mapping
        df = self._map_labels(df)

        # 4. Drop HIPAA-sensitive columns
        df, n_dropped = self._drop_hipaa_columns(df)

        # 5. Compute WUSTL Normal medians for biometric imputation
        bio_medians = self._compute_wustl_normal_medians()

        # 6. Impute missing biometric features
        df, n_imputed = self._impute_biometrics(df, bio_medians)

        # 7. Extract labels
        y: Optional[np.ndarray] = None
        if self._label_column in df.columns:
            y = df[self._label_column].values.astype(int)

        # 8. Enforce 29-feature order
        X_df = self._enforce_feature_order(df)

        # 9. Apply scaler (no refit)
        X_scaled = self._apply_scaler(X_df.values.astype(np.float32))

        logger.info(
            "  Imputed %d biometric features with WUSTL medians",
            n_imputed,
        )
        logger.info("  CICIoMT2024 prepared: %d samples", len(X_scaled))

        load_report: Dict[str, Any] = {
            "csv_path": str(self._csv_path),
            "raw_rows": raw_rows,
            "mapped_columns": len(self._column_mapping),
            "hipaa_columns_dropped": n_dropped,
            "biometric_features_imputed": n_imputed,
            "biometric_medians": {k: round(v, 4) for k, v in bio_medians.items()},
            "final_shape": list(X_scaled.shape),
            "has_labels": y is not None,
            "samples": len(X_scaled),
            "disclosure": _DISCLOSURE,
        }

        if y is not None:
            unique, counts = np.unique(y, return_counts=True)
            load_report["label_distribution"] = {
                str(int(u)): int(c) for u, c in zip(unique, counts)
            }

        return X_scaled, y, load_report

    def _load_csv(self) -> pd.DataFrame:
        """Load the CICIoMT2024 CSV file."""
        if not self._csv_path.exists():
            raise FileNotFoundError(f"CICIoMT2024 CSV not found: {self._csv_path}")
        df = pd.read_csv(self._csv_path)
        logger.info("  Loaded CSV: %d rows x %d cols", len(df), len(df.columns))
        AuditLogger.log_file_access("CROSS_DATASET_LOADED", self._csv_path)
        return df

    def _map_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply column name mapping from CICIoMT2024 to WUSTL names."""
        present = {k: v for k, v in self._column_mapping.items() if k in df.columns}
        if present:
            df = df.rename(columns=present)
            logger.info("  Mapped %d columns: %s", len(present), list(present.values()))
        return df

    def _map_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply label value mapping if provided."""
        if self._label_mapping and self._label_column in df.columns:
            df[self._label_column] = df[self._label_column].map(self._label_mapping)
            unmapped = df[self._label_column].isna().sum()
            if unmapped > 0:
                logger.warning("  %d rows have unmapped labels — dropped", unmapped)
                df = df.dropna(subset=[self._label_column])
            df[self._label_column] = df[self._label_column].astype(int)
            logger.info("  Label mapping applied: %s", self._label_mapping)
        return df

    def _drop_hipaa_columns(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """Drop HIPAA-sensitive columns (same as Phase 1)."""
        present = [c for c in _HIPAA_DROP_COLUMNS if c in df.columns]
        df = df.drop(columns=present, errors="ignore")
        logger.info("  HIPAA: dropped %d columns", len(present))
        return df, len(present)

    def _compute_wustl_normal_medians(self) -> Dict[str, float]:
        """Compute median biometric values from WUSTL Normal (y=0) samples."""
        if not self._wustl_train_path.exists():
            raise FileNotFoundError(f"WUSTL train data not found: {self._wustl_train_path}")
        train_df = pd.read_parquet(self._wustl_train_path)
        normal_df = train_df[train_df["Label"] == 0]

        medians: Dict[str, float] = {}
        for col in sorted(BIOMETRIC_COLUMNS):
            if col in normal_df.columns:
                medians[col] = float(normal_df[col].median())
            else:
                medians[col] = 0.0
        logger.info(
            "  WUSTL Normal medians (n=%d): %s",
            len(normal_df),
            {k: f"{v:.4f}" for k, v in medians.items()},
        )
        return medians

    def _impute_biometrics(
        self, df: pd.DataFrame, medians: Dict[str, float]
    ) -> Tuple[pd.DataFrame, int]:
        """Add missing biometric columns using WUSTL Normal medians."""
        n_imputed = 0
        for col in sorted(BIOMETRIC_COLUMNS):
            if col not in df.columns:
                df[col] = medians.get(col, 0.0)
                n_imputed += 1
                logger.info(
                    "    Imputed %s = %.4f (WUSTL Normal median)",
                    col,
                    medians.get(col, 0.0),
                )
        return df, n_imputed

    def _enforce_feature_order(self, df: pd.DataFrame) -> pd.DataFrame:
        """Verify and reorder to canonical 29-feature order."""
        missing = [c for c in WUSTL_FEATURE_ORDER if c not in df.columns]
        if missing:
            raise ValueError(f"CICIoMT2024 missing required features " f"after mapping: {missing}")
        return df[WUSTL_FEATURE_ORDER]

    def _apply_scaler(self, X: np.ndarray) -> np.ndarray:
        """Apply the pre-fitted RobustScaler (no refit)."""
        if not self._scaler_path.exists():
            raise FileNotFoundError(f"Scaler not found: {self._scaler_path}")
        scaler = joblib.load(self._scaler_path)
        X_scaled = scaler.transform(X)
        logger.info("  Scaler applied (transform only, no refit)")
        AuditLogger.log_file_access("SCALER_LOADED", self._scaler_path)
        return X_scaled.astype(np.float32)
