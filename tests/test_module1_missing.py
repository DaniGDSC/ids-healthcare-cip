"""MissingValueHandler tests — context-aware missing-value defense.

Critical invariants:
  - ffill without session_column → ValueError (cross-patient leakage defense)
  - fill_zero emits WARNING (attacker-induced capture-loss mask defense)
  - median bio strategy is patient-safe (default)
  - dropna network strategy preserves missing/zero distinction (default)
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from module1_preprocessing.missing import MissingValueHandler


@pytest.fixture
def df():
    return pd.DataFrame({
        "Pulse_Rate": [70, np.nan, 75, 80, np.nan],
        "Temp":       [36.5, 37.0, np.nan, np.nan, 36.8],
        "Dur":        [0.1, 0.2, np.nan, 0.4, 0.5],
        "TotPkts":    [10, 20, 30, np.nan, 50],
        "Label":      [0, 1, 0, 1, 0],
    })


# ── Constructor validation ────────────────────────────────────────────


def test_ffill_without_session_column_raises():
    """ffill across patient sessions is the canonical cross-patient leak."""
    with pytest.raises(ValueError, match="requires a session_column"):
        MissingValueHandler(
            biometric_columns=["Pulse_Rate"],
            biometric_strategy="ffill",
            session_column=None,
        )


def test_ffill_with_session_column_constructs_ok():
    h = MissingValueHandler(
        biometric_columns=["Pulse_Rate"],
        biometric_strategy="ffill",
        session_column="patient_id",
    )
    assert h is not None


def test_fill_zero_emits_warning(caplog):
    caplog.set_level(logging.WARNING)
    MissingValueHandler(
        biometric_columns=["Pulse_Rate"],
        network_strategy="fill_zero",
    )
    msgs = " ".join(r.message for r in caplog.records)
    assert "fill_zero" in msgs
    assert "attacker" in msgs.lower() or "capture loss" in msgs.lower()


def test_unknown_biometric_strategy_rejected():
    with pytest.raises(ValueError, match="biometric_strategy must be"):
        MissingValueHandler(biometric_columns=["Pulse_Rate"], biometric_strategy="bogus")


def test_unknown_network_strategy_rejected():
    with pytest.raises(ValueError, match="network_strategy must be"):
        MissingValueHandler(biometric_columns=["Pulse_Rate"], network_strategy="bogus")


# ── median + dropna defaults ──────────────────────────────────────────


def test_median_fills_biometric_nans(df):
    h = MissingValueHandler(
        biometric_columns=["Pulse_Rate", "Temp"],
        biometric_strategy="median",
        network_strategy="dropna",
    )
    out = h.transform(df.copy())
    assert out["Pulse_Rate"].isna().sum() == 0
    assert out["Temp"].isna().sum() == 0


def test_dropna_removes_network_nans(df):
    """Rows with NaN in network cols must be dropped, not zero-filled."""
    h = MissingValueHandler(
        biometric_columns=["Pulse_Rate", "Temp"],
        biometric_strategy="median",
        network_strategy="dropna",
    )
    out = h.transform(df.copy())
    # Original had 2 rows with network NaN (Dur or TotPkts) → 3 rows remain
    assert out["Dur"].isna().sum() == 0
    assert out["TotPkts"].isna().sum() == 0


def test_dropna_report_records_drop_count(df):
    h = MissingValueHandler(
        biometric_columns=["Pulse_Rate", "Temp"],
        biometric_strategy="median",
        network_strategy="dropna",
    )
    h.transform(df.copy())
    report = h.get_report()
    assert report["rows_dropped"] >= 1
    assert report["biometric_strategy"] == "median"
    assert report["network_strategy"] == "dropna"


# ── ffill within session (safe path) ──────────────────────────────────


def test_ffill_within_session_no_cross_patient_contamination():
    """A NaN at the start of patient B must NOT be filled from patient A."""
    df = pd.DataFrame({
        "patient_id": ["A", "A", "A", "B", "B", "B"],
        "Pulse_Rate": [70, 72, 74, np.nan, 90, 92],
        "Temp":       [36.5, 36.6, 36.7, 38.0, 38.1, 38.2],
        "Dur":        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        "Label":      [0, 0, 0, 1, 1, 1],
    })
    h = MissingValueHandler(
        biometric_columns=["Pulse_Rate", "Temp"],
        biometric_strategy="ffill",
        network_strategy="dropna",
        session_column="patient_id",
    )
    out = h.transform(df.copy())
    # Patient B's first Pulse_Rate was NaN → bfill from B's next row (90), NOT
    # ffill from A's last row (74).
    b_first = out[out["patient_id"] == "B"].iloc[0]["Pulse_Rate"]
    assert b_first == 90, (
        f"Cross-patient leak: B's first reading became {b_first} "
        f"(expected 90 from bfill within B). 74 = A's last → contamination."
    )


def test_fill_zero_records_filled_count(df):
    h = MissingValueHandler(
        biometric_columns=["Pulse_Rate", "Temp"],
        biometric_strategy="median",
        network_strategy="fill_zero",
    )
    out = h.transform(df.copy())
    assert out["Dur"].isna().sum() == 0
    assert (out["Dur"] == 0).any()
    report = h.get_report()
    assert report["network_cells_filled_zero"] >= 1
