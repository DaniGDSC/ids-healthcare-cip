"""Per-row device-class derivation (closes GAP-A7).

Lifts the biometric-feature heuristic out of module6_evaluation so both
Module 1 (preprocessing exporter) and Module 6 (per-device metrics) call
the same code. The derivation runs against the standardised 25-feature
flow vector after Module 1's RobustScaler step.

Heuristic — based on which biometric features are non-trivially present:

    ventilator       Resp_Rate + SpO2 active + ≥4 of 6 bio features active
    patient_monitor  Pulse_Rate + Heart_rate active + ≥3 of 6 active
    infusion_pump    Temp active + ≥2 of 6 active
    ehr_workstation  ≤1 bio feature active AND Sport/SrcBytes activity
    other            otherwise

This is a heuristic, not a directory join. When a real device-inventory
join is available (per-device IP/MAC mapping), this file should be
replaced with that lookup.
"""
from __future__ import annotations

from typing import Iterable, List

import numpy as np
import pandas as pd

# Order is a contract: the bio-feature mask is constructed in this order
# and indexed by name elsewhere — do not reorder.
BIO_FEATS: tuple[str, ...] = (
    "Temp", "SpO2", "Pulse_Rate", "Heart_rate", "Resp_Rate", "ST",
)
_BIO_THRESHOLD = 0.5
_NET_THRESHOLD = 0.1


def derive_device_class_row(row: pd.Series) -> str:
    """Derive device_class for one standardised flow vector.

    Args:
        row: A pandas Series indexed by feature name. Missing features are
             treated as 0.0.

    Returns:
        One of: ventilator, patient_monitor, infusion_pump,
                ehr_workstation, other.
    """
    vals = {f: abs(float(row.get(f, 0.0))) > _BIO_THRESHOLD for f in BIO_FEATS}
    bio_active = sum(vals.values())
    if vals["Resp_Rate"] and vals["SpO2"] and bio_active >= 4:
        return "ventilator"
    if vals["Pulse_Rate"] and vals["Heart_rate"] and bio_active >= 3:
        return "patient_monitor"
    if vals["Temp"] and bio_active >= 2:
        return "infusion_pump"
    sport = abs(float(row.get("Sport", 0.0)))
    src = abs(float(row.get("SrcBytes", 0.0)))
    if bio_active <= 1 and (sport > _NET_THRESHOLD or src > _NET_THRESHOLD):
        return "ehr_workstation"
    return "other"


def derive_device_class_array(X: np.ndarray, feature_names: Iterable[str]) -> List[str]:
    """Vectorised derivation over a (n, n_features) numpy matrix.

    Args:
        X: standardised feature matrix from Module 1's scaler step.
        feature_names: column-name order matching X's columns.

    Returns:
        List of device-class strings, length = X.shape[0].
    """
    feats = list(feature_names)
    df = pd.DataFrame(X, columns=feats)
    return [derive_device_class_row(df.iloc[i]) for i in range(len(df))]
