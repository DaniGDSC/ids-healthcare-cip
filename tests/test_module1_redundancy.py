"""RedundancyRemover tests — Phase 0 correlation CSV consumer.

Critical invariants:
  - Refuses to drop protected columns (Label, Attack Category) even if
    listed as feature_b in the correlations CSV → tampered-CSV defense
  - Rejects malformed corr_df schema at construction time
  - Honors threshold (drops only |r| >= threshold)
"""
from __future__ import annotations

import pandas as pd
import pytest

from module1_preprocessing.phase1.redundancy import RedundancyRemover


@pytest.fixture
def corr_df():
    return pd.DataFrame({
        "feature_a": ["A", "B", "C", "D"],
        "feature_b": ["A_dup", "B_dup", "C_dup", "D_dup"],
        "correlation": [0.98, 0.96, 0.50, -0.97],
    })


@pytest.fixture
def feature_df():
    return pd.DataFrame({
        "A": [1, 2, 3], "A_dup": [1, 2, 3],
        "B": [4, 5, 6], "B_dup": [4, 5, 6],
        "C": [7, 8, 9], "C_dup": [10, 11, 12],
        "D": [1, 2, 3], "D_dup": [-1, -2, -3],
    })


def test_drops_pairs_above_threshold(corr_df, feature_df):
    rem = RedundancyRemover(corr_df, threshold=0.95)
    out = rem.transform(feature_df.copy())
    # A_dup (0.98), B_dup (0.96), D_dup (-0.97 abs=0.97) dropped; C_dup kept (0.50)
    assert "A_dup" not in out.columns
    assert "B_dup" not in out.columns
    assert "D_dup" not in out.columns
    assert "C_dup" in out.columns


def test_protected_columns_refused_even_if_listed_as_feature_b():
    """Tampered-CSV defense: Label/Attack Category must never be dropped."""
    bad_corr = pd.DataFrame({
        "feature_a": ["Dur", "Pulse_Rate"],
        "feature_b": ["Label", "Attack Category"],
        "correlation": [0.99, 0.99],
    })
    df = pd.DataFrame({
        "Dur": [1, 2], "Pulse_Rate": [70, 80],
        "Label": [0, 1], "Attack Category": ["normal", "recon"],
    })
    rem = RedundancyRemover(
        bad_corr, threshold=0.95,
        protected_columns=("Label", "Attack Category"),
    )
    out = rem.transform(df.copy())
    assert "Label" in out.columns
    assert "Attack Category" in out.columns
    report = rem.get_report()
    assert "Label" in report["columns_refused"]
    assert "Attack Category" in report["columns_refused"]
    assert report["n_refused_protected"] == 2


def test_protected_refusal_logged_at_error_level(caplog):
    import logging
    bad_corr = pd.DataFrame({
        "feature_a": ["Dur"], "feature_b": ["Label"], "correlation": [0.99],
    })
    df = pd.DataFrame({"Dur": [1], "Label": [0]})
    caplog.set_level(logging.ERROR)
    rem = RedundancyRemover(bad_corr, threshold=0.95)
    rem.transform(df.copy())
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert errors
    assert any("REFUSED" in r.message for r in errors)


def test_malformed_corr_schema_rejected_at_construction():
    bad = pd.DataFrame({"a": [1], "b": [2]})  # missing required columns
    with pytest.raises(ValueError, match="missing required columns"):
        RedundancyRemover(bad, threshold=0.9)


def test_threshold_below_threshold_not_dropped(corr_df, feature_df):
    """Pairs with |r| < threshold remain."""
    rem = RedundancyRemover(corr_df, threshold=0.99)
    out = rem.transform(feature_df.copy())
    # Only A_dup (0.98)? No — 0.98 < 0.99 so kept too.
    assert "A_dup" in out.columns
    assert "B_dup" in out.columns


def test_dropping_nonexistent_column_silently_skipped(corr_df):
    """corr_df may list features not in the input — handled gracefully."""
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})  # no A_dup, B_dup
    out = RedundancyRemover(corr_df, threshold=0.95).transform(df.copy())
    # No crash; nothing dropped (those columns weren't there to begin with)
    assert list(out.columns) == ["A", "B"]


def test_report_records_dropped_and_refused(corr_df, feature_df):
    rem = RedundancyRemover(corr_df, threshold=0.95)
    rem.transform(feature_df.copy())
    report = rem.get_report()
    assert report["n_dropped"] == 3
    assert report["threshold"] == 0.95
    assert report["columns_refused"] == []
    assert report["n_refused_protected"] == 0
