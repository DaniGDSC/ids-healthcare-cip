"""Feature alignment and imputation for MedSec-25 flows.

Aligns external dataset columns to the 29 WUSTL-EHMS feature schema,
applying imputation from Normal-class medians for unmapped features.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# 29 WUSTL features in canonical order
WUSTL_FEATURES: List[str] = [
    "SrcBytes", "DstBytes", "SrcLoad", "DstLoad", "SrcGap", "DstGap",
    "SIntPkt", "DIntPkt", "SIntPktAct", "DIntPktAct", "sMaxPktSz",
    "dMaxPktSz", "sMinPktSz", "dMinPktSz", "Dur", "Trans", "TotBytes",
    "Load", "pSrcLoss", "pDstLoss", "Packet_num", "Temp", "SpO2",
    "Pulse_Rate", "SYS", "DIA", "Heart_rate", "Resp_Rate", "ST",
]

# MedSec-25 (CIC-style) → WUSTL feature mapping
MEDSEC25_MAPPING: Dict[str, str] = {
    "Flow Duration": "Dur",
    "Tot Fwd Pkts": "Packet_num",
    "TotLen Fwd Pkts": "SrcBytes",
    "TotLen Bwd Pkts": "DstBytes",
    "Flow Byts/s": "Load",
    "Subflow Fwd Byts": "TotBytes",
    "Fwd IAT Mean": "SIntPkt",
    "Bwd IAT Mean": "DIntPkt",
    "Fwd Pkt Len Mean": "sMaxPktSz",
    "Bwd Pkt Len Mean": "dMaxPktSz",
    "Fwd Pkt Len Min": "sMinPktSz",
    "Bwd Pkt Len Min": "dMinPktSz",
}

# WUSTL Normal-class medians for imputation (from preprocessing baseline)
NORMAL_MEDIANS: Dict[str, float] = {
    "SrcBytes": 310.0, "DstBytes": 246.0, "SrcLoad": 6891.56,
    "DstLoad": 5576.42, "SrcGap": 36979.0, "DstGap": 36979.0,
    "SIntPkt": 39969.0, "DIntPkt": 39969.0, "SIntPktAct": 22912.0,
    "DIntPktAct": 26752.0, "sMaxPktSz": 60.0, "dMaxPktSz": 54.0,
    "sMinPktSz": 40.0, "dMinPktSz": 40.0, "Dur": 0.079,
    "Trans": 5.0, "TotBytes": 556.0, "Load": 12468.0,
    "pSrcLoss": 0.0, "pDstLoss": 0.0, "Packet_num": 10.0,
    "Temp": 36.8, "SpO2": 97.0, "Pulse_Rate": 75.0,
    "SYS": 120.0, "DIA": 80.0, "Heart_rate": 72.0,
    "Resp_Rate": 16.0, "ST": 0.1,
}

# Features that are biometric (not present in network-only datasets)
BIOMETRIC_FEATURES: List[str] = [
    "Temp", "SpO2", "Pulse_Rate", "SYS", "DIA",
    "Heart_rate", "Resp_Rate", "ST",
]

N_FEATURES: int = 29


def get_mapped_count() -> int:
    """Return number of directly mapped features."""
    return len(MEDSEC25_MAPPING)


def get_imputed_count() -> int:
    """Return number of imputed features."""
    mapped_wustl = set(MEDSEC25_MAPPING.values())
    return N_FEATURES - len(mapped_wustl)


def align_medsec25_row(
    row: pd.Series,
    medsec_columns: List[str],
) -> np.ndarray:
    """Align a single MedSec-25 row to 29 WUSTL features.

    Args:
        row: Single row from MedSec-25 DataFrame.
        medsec_columns: Column names of the source DataFrame.

    Returns:
        Aligned feature vector of shape (29,).
    """
    aligned = np.zeros(N_FEATURES, dtype=np.float32)

    for i, feat in enumerate(WUSTL_FEATURES):
        # Check if any MedSec-25 column maps to this WUSTL feature
        mapped = False
        for medsec_col, wustl_feat in MEDSEC25_MAPPING.items():
            if wustl_feat == feat and medsec_col in medsec_columns:
                try:
                    val = float(row[medsec_col])
                    if np.isfinite(val):
                        aligned[i] = val
                        mapped = True
                except (ValueError, TypeError):
                    pass
                break

        if not mapped:
            aligned[i] = NORMAL_MEDIANS.get(feat, 0.0)

    return aligned


def align_medsec25_batch(df: pd.DataFrame) -> pd.DataFrame:
    """Align a MedSec-25 DataFrame to WUSTL 29-feature schema.

    Args:
        df: MedSec-25 DataFrame with CIC-style columns.

    Returns:
        DataFrame with 29 WUSTL columns + Label.
    """
    cols = list(df.columns)
    aligned_rows = []

    for _, row in df.iterrows():
        vec = align_medsec25_row(row, cols)
        aligned_rows.append(vec)

    result = pd.DataFrame(aligned_rows, columns=WUSTL_FEATURES)

    # Map label
    if "Label" in df.columns:
        result["Label"] = (df["Label"].str.lower() != "benign").astype(int).values

    return result


def validate_schema(df: pd.DataFrame) -> Tuple[bool, str]:
    """Validate that a DataFrame has the required 29 WUSTL features.

    Args:
        df: Input DataFrame to validate.

    Returns:
        Tuple of (is_valid, message).
    """
    cols_lower = {c.lower(): c for c in df.columns}
    missing = []
    for feat in WUSTL_FEATURES:
        if feat.lower() not in cols_lower:
            missing.append(feat)

    if not missing:
        return True, "Schema valid: 29/29 features present"

    if len(missing) <= 5:
        return False, f"Missing {len(missing)} features: {', '.join(missing)}"

    return False, f"Missing {len(missing)}/29 features — alignment required"


def get_alignment_summary() -> Dict[str, Any]:
    """Return a summary of the feature alignment configuration."""
    mapped_wustl = set(MEDSEC25_MAPPING.values())
    imputed = [f for f in WUSTL_FEATURES if f not in mapped_wustl]
    return {
        "total_features": N_FEATURES,
        "mapped": len(mapped_wustl),
        "imputed": len(imputed),
        "imputed_features": imputed,
        "biometric_imputed": [f for f in imputed if f in BIOMETRIC_FEATURES],
        "network_imputed": [f for f in imputed if f not in BIOMETRIC_FEATURES],
    }
