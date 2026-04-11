"""Canonical PHI column set for the IoMT IDS pipeline.

The WUSTL-EHMS-2020 dataset interleaves network-flow features with eight
biometric channels collected from real patients. Under HIPAA Safe Harbor,
patient-level minima/maxima of these channels can act as quasi-identifiers,
so every module that logs, exports, or visualises dataset values must
treat the columns listed here as protected.

This module is the single source of truth. Do NOT redefine the set
locally — import from here:

    from pipeline.common.phi import BIOMETRIC_COLUMNS
"""

from __future__ import annotations

# Frozen so accidental mutation in one caller cannot leak into another.
BIOMETRIC_COLUMNS: frozenset[str] = frozenset({
    "Temp",
    "SpO2",
    "Pulse_Rate",
    "SYS",
    "DIA",
    "Heart_rate",
    "Resp_Rate",
    "ST",
})
