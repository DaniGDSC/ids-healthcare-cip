"""Heuristic per-row device-class derivation from biometric feature activity.

Mirrors ``module6_evaluation._derive_device_class`` so Phase 1 parquets and
Module 6 evaluation tag rows the same way. Operates on the post-scaling
feature matrix produced by Phase 1; absent biometric columns contribute zero.
"""

from __future__ import annotations

from typing import Iterable, List

import numpy as np

_BIO_FEATS = ("Temp", "SpO2", "Pulse_Rate", "Heart_rate", "Resp_Rate", "ST")
_ACTIVATION = 0.5


def derive_device_class_array(
    X: np.ndarray,
    feature_names: Iterable[str],
) -> List[str]:
    """Classify each row of *X* into a device class.

    Args:
        X: 2-D feature matrix (rows × features) after scaling.
        feature_names: Column names aligned with ``X``'s second axis.

    Returns:
        List of device-class labels, one per row. Labels are one of
        ``ventilator``, ``patient_monitor``, ``infusion_pump``,
        ``ehr_workstation``, ``other``.
    """
    names = list(feature_names)
    idx = {name: i for i, name in enumerate(names)}

    def col(name: str) -> np.ndarray:
        i = idx.get(name)
        if i is None:
            return np.zeros(len(X), dtype=float)
        return np.abs(X[:, i].astype(float))

    bio_cols = {name: col(name) > _ACTIVATION for name in _BIO_FEATS}
    bio_active = np.sum(np.stack(list(bio_cols.values()), axis=1), axis=1)

    sport = col("Sport")
    src_bytes = col("SrcBytes")

    out = np.full(len(X), "other", dtype=object)
    out[(bio_active <= 1) & ((sport > 0.1) | (src_bytes > 0.1))] = "ehr_workstation"
    out[bio_cols["Temp"] & (bio_active >= 2)] = "infusion_pump"
    out[bio_cols["Pulse_Rate"] & bio_cols["Heart_rate"] & (bio_active >= 3)] = (
        "patient_monitor"
    )
    out[bio_cols["Resp_Rate"] & bio_cols["SpO2"] & (bio_active >= 4)] = "ventilator"
    return out.tolist()
