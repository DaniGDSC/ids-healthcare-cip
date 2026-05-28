"""HIPAASanitizer tests."""
from __future__ import annotations

import pandas as pd

from module1_preprocessing.hipaa import HIPAASanitizer


def test_drops_listed_columns():
    df = pd.DataFrame({
        "SrcAddr": ["10.0.0.1"],
        "DstAddr": ["10.0.0.2"],
        "Label": [0],
        "Dur": [0.5],
    })
    out = HIPAASanitizer(["SrcAddr", "DstAddr"]).transform(df)
    assert "SrcAddr" not in out.columns
    assert "DstAddr" not in out.columns
    assert "Label" in out.columns
    assert "Dur" in out.columns


def test_tolerates_missing_listed_columns():
    """Listed-but-absent columns must not raise — defensive."""
    df = pd.DataFrame({"Label": [0], "Dur": [0.5]})
    out = HIPAASanitizer(["NotPresent", "AlsoMissing"]).transform(df)
    assert list(out.columns) == ["Label", "Dur"]


def test_report_records_what_was_dropped():
    df = pd.DataFrame({"SrcAddr": [1], "Label": [0]})
    san = HIPAASanitizer(["SrcAddr", "MissingCol"])
    san.transform(df)
    report = san.get_report()
    assert report["columns_dropped"] == ["SrcAddr"]
    assert report["columns_requested"] == ["SrcAddr", "MissingCol"]
    assert report["n_dropped"] == 1


def test_empty_drop_list_is_noop():
    df = pd.DataFrame({"Label": [0], "Dur": [0.5]})
    out = HIPAASanitizer([]).transform(df)
    pd.testing.assert_frame_equal(out, df)


def test_transform_does_not_mutate_input():
    df = pd.DataFrame({"SrcAddr": [1], "Label": [0]})
    before_cols = list(df.columns)
    HIPAASanitizer(["SrcAddr"]).transform(df)
    assert list(df.columns) == before_cols, "Input frame must not be mutated"
