"""CategoricalEncoder tests — deterministic mapping + JSON sidecar.

Critical invariants:
  - Mappings are SORTED-ALPHABETICAL, never observation-order
    (this is the LabelEncoder leak defense)
  - JSON sidecar round-trip is byte-identical
  - Unknown values at inference map to unknown_value (default -1)
  - Sentinel for unparseable strings is configurable (-99999 in spec)
"""
from __future__ import annotations


import pandas as pd
import pytest

from module1_preprocessing.phase1.encoder import CategoricalEncoder


def test_mapping_is_sorted_alphabetical_not_observation_order():
    """LabelEncoder defense: encoding must be independent of row order."""
    df = pd.DataFrame({"Dir": ["b", "a", "c"]})
    enc1 = CategoricalEncoder(label_encode=["Dir"])
    enc1.transform(df.copy())
    df_reversed = pd.DataFrame({"Dir": ["c", "b", "a"]})
    enc2 = CategoricalEncoder(label_encode=["Dir"])
    enc2.transform(df_reversed)
    # Same alphabetical mapping regardless of observation order
    assert enc1._mappings["Dir"] == enc2._mappings["Dir"]
    # And specifically: 'a' → 0, 'b' → 1, 'c' → 2
    assert enc1._mappings["Dir"] == {"a": 0, "b": 1, "c": 2}


def test_label_encode_replaces_strings_with_ints():
    df = pd.DataFrame({"Flgs": ["x", "y", "x", "z"], "Other": [1, 2, 3, 4]})
    out = CategoricalEncoder(label_encode=["Flgs"]).transform(df.copy())
    assert out["Flgs"].dtype.kind in ("i", "u")  # integer
    assert set(out["Flgs"].tolist()) == {0, 1, 2}  # 3 unique → {0,1,2}


def test_parse_numeric_with_sentinel():
    """Non-parseable strings become sentinel, not NaN."""
    df = pd.DataFrame({"Sport": ["443", "80", "bogus", "8080"]})
    enc = CategoricalEncoder(parse_numeric=["Sport"], sentinel=-99999)
    out = enc.transform(df.copy())
    assert out.loc[2, "Sport"] == -99999
    assert out.loc[0, "Sport"] == 443


def test_parse_numeric_sentinel_is_outside_valid_port_range():
    """-99999 must be well outside [0, 65535] so the model can't learn it
    as a meaningful value next to legitimate port codes.
    """
    df = pd.DataFrame({"Sport": ["bogus"]})
    enc = CategoricalEncoder(parse_numeric=["Sport"], sentinel=-99999)
    out = enc.transform(df.copy())
    assert out.loc[0, "Sport"] < 0  # outside any valid port
    assert abs(out.loc[0, "Sport"]) > 65535  # outside max port too


def test_sidecar_round_trip_preserves_mapping(tmp_path):
    """Loading a saved sidecar must reproduce the same mapping."""
    df = pd.DataFrame({"Dir": ["b", "a", "c"]})
    enc_before = CategoricalEncoder(label_encode=["Dir"])
    enc_before.transform(df.copy())
    sidecar = tmp_path / "encoder.json"
    enc_before.save(sidecar)

    enc_after = CategoricalEncoder.from_json(sidecar)
    assert enc_after._mappings == enc_before._mappings


def test_sidecar_rejects_pkl_path_writes_json(tmp_path):
    """Legacy .pkl request must be rewritten to .json on disk."""
    df = pd.DataFrame({"Dir": ["a", "b"]})
    enc = CategoricalEncoder(label_encode=["Dir"])
    enc.transform(df.copy())
    pkl_path = tmp_path / "encoder.pkl"
    json_path = enc.save(pkl_path)
    assert json_path.suffix == ".json"
    assert json_path.exists()
    assert not pkl_path.exists()


def test_sidecar_load_rejects_wrong_format(tmp_path):
    p = tmp_path / "fake.json"
    p.write_text('{"format": "not-an-encoder"}')
    with pytest.raises(ValueError, match="not a phase1.encoder.v1 sidecar"):
        CategoricalEncoder.from_json(p)


def test_sidecar_load_fails_on_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        CategoricalEncoder.from_json(tmp_path / "nope.json")


def test_report_records_per_column_class_count():
    df = pd.DataFrame({"Dir": ["a", "b", "c"], "Flgs": ["x", "x", "y"]})
    enc = CategoricalEncoder(label_encode=["Dir", "Flgs"])
    enc.transform(df.copy())
    report = enc.get_report()
    assert report["label_encoded"] == {"Dir": 3, "Flgs": 2}
    assert report["mapping_classes"] == {"Dir": 3, "Flgs": 2}


def test_missing_columns_silently_skipped():
    df = pd.DataFrame({"Dir": ["a", "b"]})
    enc = CategoricalEncoder(label_encode=["Dir", "NotPresent"])
    out = enc.transform(df.copy())
    assert list(out.columns) == ["Dir"]
    assert "Dir" in enc._mappings
    assert "NotPresent" not in enc._mappings
