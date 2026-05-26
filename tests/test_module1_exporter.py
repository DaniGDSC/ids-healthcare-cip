"""PreprocessingExporter tests — atomic parquet + JSON, no pickle.

Critical invariants:
  - Parquet write is atomic (.tmp + rename)
  - export_scaler refuses non-sidecar scalers (no joblib.dump fallback)
  - export_report strict-fails on non-JSON-serialisable values
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from module1_preprocessing.phase1.exporter import PreprocessingExporter
from module1_preprocessing.phase1.scaler import RobustScalerTransformer


@pytest.fixture
def exporter(tmp_path):
    return PreprocessingExporter(
        output_dir=tmp_path / "out",
        scaler_dir=tmp_path / "scalers",
    )


@pytest.fixture
def X():
    return np.array([[1.0, 100], [2.0, 200], [3.0, 300]])


@pytest.fixture
def y():
    return np.array([0, 1, 0])


def test_parquet_atomic_write(exporter, X, y, tmp_path):
    p = exporter.export_parquet(X, y, ["Dur", "TotPkts"], "x.parquet")
    assert p.exists()
    # No leftover .tmp
    assert not list((tmp_path / "out").glob("*.tmp"))


def test_parquet_includes_row_id_and_device_class(exporter, X, y):
    p = exporter.export_parquet(X, y, ["Dur", "TotPkts"], "x.parquet")
    df = pd.read_parquet(p)
    assert "row_id" in df.columns
    assert "device_class" in df.columns


def test_parquet_with_multi_label(exporter, X, y):
    y_multi = np.array(["normal", "recon", "normal"], dtype=object)
    p = exporter.export_parquet(
        X, y, ["Dur", "TotPkts"], "x.parquet", y_multi=y_multi,
    )
    df = pd.read_parquet(p)
    assert "Attack Category" in df.columns


def test_export_scaler_via_sidecar(exporter, X):
    scaler = RobustScalerTransformer().fit(X)
    p = exporter.export_scaler(scaler, "scaler.json")
    assert p.exists()
    body = json.loads(p.read_text())
    assert body["format"] == "phase1.scaler.v1"


def test_export_scaler_legacy_pkl_filename_rewritten(exporter, X):
    """Legacy .pkl filename in config must still produce .json on disk."""
    scaler = RobustScalerTransformer().fit(X)
    p = exporter.export_scaler(scaler, "scaler.pkl")
    assert p.suffix == ".json"
    assert p.exists()


def test_export_scaler_refuses_non_sidecar_object(exporter):
    """A raw sklearn scaler (no .save method) must be rejected loudly."""
    from sklearn.preprocessing import RobustScaler
    raw = RobustScaler()
    with pytest.raises(TypeError, match="refuses to pickle"):
        exporter.export_scaler(raw, "scaler.json")


def test_export_report_writes_json(exporter):
    p = exporter.export_report({"a": 1, "b": [2, 3]}, "report.json")
    assert p.exists()
    assert json.loads(p.read_text()) == {"a": 1, "b": [2, 3]}


def test_export_report_strict_fails_on_non_serialisable(exporter):
    """The exporter no longer silently coerces with default=str."""
    bad = {"a": object()}
    with pytest.raises(TypeError, match="non-JSON-serialisable"):
        exporter.export_report(bad, "report.json")


def test_export_report_atomic_no_tmp_left(exporter, tmp_path):
    exporter.export_report({"a": 1}, "report.json")
    assert not list((tmp_path / "out").glob("*.tmp"))
