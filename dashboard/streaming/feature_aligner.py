"""Feature alignment for streaming ingestion.

Supports two modes:
  1. WUSTL-native (production): All 24 features present, no imputation.
     Used on WUSTL-compatible hospital networks.
  2. MedSec-25 (simulation): 12 CIC columns mapped, 17 imputed from
     Normal-class medians. Used for demo/testing only.

After Phase 1 variance filtering, the model uses 24 features (not 29).
The 5 dropped features (SrcGap, DstGap, DIntPktAct, dMinPktSz, Trans)
are excluded from the canonical schema.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Canonical 24-feature schema (post-variance filtering) ──────────

MODEL_FEATURES: List[str] = [
    "SrcBytes", "DstBytes", "SrcLoad", "DstLoad",
    "SIntPkt", "DIntPkt", "SIntPktAct",
    "sMaxPktSz", "dMaxPktSz", "sMinPktSz",
    "Dur", "TotBytes", "Load", "pSrcLoss", "pDstLoss", "Packet_num",
    "Temp", "SpO2", "Pulse_Rate", "SYS", "DIA", "Heart_rate",
    "Resp_Rate", "ST",
]

N_FEATURES: int = len(MODEL_FEATURES)  # 24

# Features dropped by Phase 1 variance filtering (constant in WUSTL)
DROPPED_FEATURES: List[str] = ["SrcGap", "DstGap", "DIntPktAct", "dMinPktSz", "Trans"]

# Original 29 WUSTL features (pre-filtering, for legacy compatibility)
WUSTL_FEATURES: List[str] = MODEL_FEATURES  # alias for backward compat

BIOMETRIC_FEATURES: List[str] = [
    "Temp", "SpO2", "Pulse_Rate", "SYS", "DIA",
    "Heart_rate", "Resp_Rate", "ST",
]

NETWORK_FEATURES: List[str] = [f for f in MODEL_FEATURES if f not in BIOMETRIC_FEATURES]


# ── WUSTL-Native alignment (production) ───────────────────────────

def align_wustl_native(df: pd.DataFrame) -> pd.DataFrame:
    """Align a WUSTL-native DataFrame to the 24-feature model schema.

    Drops the 5 zero-variance features if present and reorders columns.
    No imputation — all 24 features must be present.

    Args:
        df: DataFrame with WUSTL features (24 or 29 columns).

    Returns:
        DataFrame with 24 model features in canonical order.

    Raises:
        ValueError: If required features are missing.
    """
    # Drop zero-variance features if present (from raw 29-feature data)
    df = df.drop(columns=[c for c in DROPPED_FEATURES if c in df.columns], errors="ignore")

    # Check all 24 required features are present
    col_map = {c.lower(): c for c in df.columns}
    ordered = []
    missing = []
    for feat in MODEL_FEATURES:
        key = feat.lower()
        if key in col_map:
            ordered.append(col_map[key])
        else:
            missing.append(feat)

    if missing:
        raise ValueError(f"Missing {len(missing)} required features: {', '.join(missing)}")

    return df[ordered].astype(np.float32)


def align_wustl_native_row(row: pd.Series) -> np.ndarray:
    """Align a single WUSTL-native row to 24-feature vector.

    Args:
        row: Single row with WUSTL features.

    Returns:
        Feature vector of shape (24,).
    """
    vec = np.zeros(N_FEATURES, dtype=np.float32)
    for i, feat in enumerate(MODEL_FEATURES):
        if feat in row.index:
            try:
                val = float(row[feat])
                vec[i] = val if np.isfinite(val) else 0.0
            except (ValueError, TypeError):
                vec[i] = 0.0
    return vec


# ── MedSec-25 alignment (simulation/demo) ─────────────────────────

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

# Normal-class medians for imputation (simulation only)
NORMAL_MEDIANS: Dict[str, float] = {
    "SrcBytes": 310.0, "DstBytes": 246.0, "SrcLoad": 6891.56,
    "DstLoad": 5576.42, "SIntPkt": 39969.0, "DIntPkt": 39969.0,
    "SIntPktAct": 22912.0, "sMaxPktSz": 60.0, "dMaxPktSz": 54.0,
    "sMinPktSz": 40.0, "Dur": 0.079, "TotBytes": 556.0,
    "Load": 12468.0, "pSrcLoss": 0.0, "pDstLoss": 0.0,
    "Packet_num": 10.0, "Temp": 36.8, "SpO2": 97.0,
    "Pulse_Rate": 75.0, "SYS": 120.0, "DIA": 80.0,
    "Heart_rate": 72.0, "Resp_Rate": 16.0, "ST": 0.1,
}


def align_medsec25_row(
    row: pd.Series,
    medsec_columns: List[str],
) -> np.ndarray:
    """Align a single MedSec-25 row to 24 model features.

    Maps 12 CIC columns directly, imputes remaining from Normal medians.
    Used for simulation only — not production.

    Args:
        row: Single row from MedSec-25 DataFrame.
        medsec_columns: Column names of the source DataFrame.

    Returns:
        Aligned feature vector of shape (24,).
    """
    aligned = np.zeros(N_FEATURES, dtype=np.float32)

    for i, feat in enumerate(MODEL_FEATURES):
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
    """Align a MedSec-25 DataFrame to 24-feature model schema.

    Args:
        df: MedSec-25 DataFrame with CIC-style columns.

    Returns:
        DataFrame with 24 model columns + Label.
    """
    cols = list(df.columns)
    aligned_rows = []

    for _, row in df.iterrows():
        vec = align_medsec25_row(row, cols)
        aligned_rows.append(vec)

    result = pd.DataFrame(aligned_rows, columns=MODEL_FEATURES)

    if "Label" in df.columns:
        result["Label"] = (df["Label"].str.lower() != "benign").astype(int).values

    return result


# ── Schema validation ──────────────────────────────────────────────

def validate_schema(df: pd.DataFrame) -> Tuple[bool, str]:
    """Validate that a DataFrame has the required 24 model features.

    Args:
        df: Input DataFrame to validate.

    Returns:
        Tuple of (is_valid, message).
    """
    cols_lower = {c.lower(): c for c in df.columns}
    missing = [f for f in MODEL_FEATURES if f.lower() not in cols_lower]

    if not missing:
        return True, f"Schema valid: {N_FEATURES}/{N_FEATURES} features present"

    if len(missing) <= 5:
        return False, f"Missing {len(missing)} features: {', '.join(missing)}"

    return False, f"Missing {len(missing)}/{N_FEATURES} features — alignment required"


def get_alignment_summary() -> Dict[str, Any]:
    """Return a summary of the feature alignment configuration."""
    mapped_wustl = set(MEDSEC25_MAPPING.values())
    imputed = [f for f in MODEL_FEATURES if f not in mapped_wustl]
    return {
        "total_features": N_FEATURES,
        "model_features": MODEL_FEATURES,
        "dropped_features": DROPPED_FEATURES,
        "medsec25_mapped": len(mapped_wustl),
        "medsec25_imputed": len(imputed),
        "biometric_features": BIOMETRIC_FEATURES,
        "network_features": NETWORK_FEATURES,
    }
