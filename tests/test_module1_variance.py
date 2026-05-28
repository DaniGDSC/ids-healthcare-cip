"""VarianceFilter tests."""
from __future__ import annotations

import pandas as pd

from module1_preprocessing.variance import VarianceFilter


def test_drops_unary_columns():
    df = pd.DataFrame({
        "const": [1, 1, 1, 1],
        "binary": [0, 1, 0, 1],
        "diverse": [1, 2, 3, 4],
    })
    out = VarianceFilter(max_unique=1).transform(df)
    assert "const" not in out.columns
    assert "binary" in out.columns
    assert "diverse" in out.columns


def test_max_unique_threshold_configurable():
    """max_unique=2 drops binary AND unary, keeps diverse."""
    df = pd.DataFrame({
        "const": [1, 1, 1, 1],
        "binary": [0, 1, 0, 1],
        "diverse": [1, 2, 3, 4],
    })
    out = VarianceFilter(max_unique=2).transform(df)
    assert "const" not in out.columns
    assert "binary" not in out.columns
    assert "diverse" in out.columns


def test_report_records_dropped_columns():
    df = pd.DataFrame({"const": [1, 1, 1], "diverse": [1, 2, 3]})
    f = VarianceFilter(max_unique=1)
    f.transform(df)
    report = f.get_report()
    assert report["columns_dropped"] == ["const"]
    assert report["n_dropped"] == 1
    assert report["max_unique"] == 1


def test_empty_dataframe_no_crash():
    out = VarianceFilter().transform(pd.DataFrame())
    assert out.empty
