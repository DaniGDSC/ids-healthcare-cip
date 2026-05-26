"""Module 4 io — paths, loading, saving, strict JSON, NumpyEncoder."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from module4_explanations.io import (
    NumpyJSONEncoder,
    _split_paths,
    export_feature_concepts,
    export_nlg_templates,
    load_predictions,
    load_test_data,
    save_dae_errors,
    save_global_importance,
    save_shap_values,
    write_json_strict,
)


# ── _split_paths ────────────────────────────────────────────────────


def test_split_paths_test_suffix_empty():
    paths = _split_paths("test")
    assert paths["suffix"] == ""
    assert paths["parquet"].name == "test_phase1.parquet"


def test_split_paths_demo_suffix_set():
    paths = _split_paths("demo")
    assert paths["suffix"] == "_demo"
    assert paths["parquet"].name == "demo_phase1.parquet"


# ── load_test_data ──────────────────────────────────────────────────


def test_load_test_data_drops_label_attack_category(tmp_path):
    df = pd.DataFrame({
        "f1": [0.1, 0.2],
        "f2": [1.0, 2.0],
        "Label": [0, 1],
        "Attack Category": ["normal", "Spoofing"],
        "row_id": [0, 1],
        "device_class": ["patient_monitor", "patient_monitor"],
    })
    pq = tmp_path / "test.parquet"
    df.to_parquet(pq, index=False)
    X, y, cats, feat_names = load_test_data(pq)
    assert X.shape == (2, 2)
    assert feat_names == ["f1", "f2"]
    assert list(cats) == ["normal", "Spoofing"]


def test_load_test_data_handles_missing_attack_category(tmp_path):
    df = pd.DataFrame({"f1": [0.1], "Label": [0]})
    pq = tmp_path / "test.parquet"
    df.to_parquet(pq, index=False)
    _, _, cats, _ = load_test_data(pq)
    assert cats is None


# ── load_predictions ────────────────────────────────────────────────


def test_load_predictions_round_trip(tmp_path):
    arr = np.array([0, 1, 0, 1])
    proba = np.array([0.1, 0.9, 0.2, 0.8])
    np.savez(tmp_path / "p.npz", y_pred=arr, y_proba=proba)
    out = load_predictions(tmp_path / "p.npz")
    np.testing.assert_array_equal(out["y_pred"], arr)
    np.testing.assert_array_almost_equal(out["y_proba"], proba)


# ── Strict JSON ─────────────────────────────────────────────────────


def test_write_json_strict_atomic_no_tmp(tmp_path):
    write_json_strict(tmp_path / "x.json", {"a": 1, "b": [2, 3]})
    assert (tmp_path / "x.json").exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_write_json_strict_rejects_non_serialisable(tmp_path):
    with pytest.raises(TypeError, match="non-JSON-serialisable"):
        write_json_strict(tmp_path / "x.json", {"a": object()})


# ── NumpyJSONEncoder ────────────────────────────────────────────────


def test_numpy_encoder_handles_numpy_int():
    out = json.dumps({"a": np.int64(42)}, cls=NumpyJSONEncoder)
    assert json.loads(out) == {"a": 42}


def test_numpy_encoder_handles_numpy_float():
    out = json.dumps({"a": np.float32(3.5)}, cls=NumpyJSONEncoder)
    assert abs(json.loads(out)["a"] - 3.5) < 1e-5


def test_numpy_encoder_handles_numpy_array():
    out = json.dumps({"a": np.array([1, 2, 3])}, cls=NumpyJSONEncoder)
    assert json.loads(out) == {"a": [1, 2, 3]}


# ── save_* round-trips ──────────────────────────────────────────────


def test_save_shap_values_round_trip(tmp_path):
    sv = np.random.randn(5, 4).astype(np.float32)
    save_shap_values("test_model", sv, 0.5, ["f1", "f2", "f3", "f4"],
                     output_dir=tmp_path)
    data = np.load(tmp_path / "shap_values_test_model.npz")
    assert data["shap_values"].shape == (5, 4)
    assert float(data["expected_value"]) == 0.5


def test_save_global_importance_writes_json(tmp_path):
    imp = [{"rank": 1, "feature": "f1", "mean_abs_shap": 0.5}]
    save_global_importance("m", imp, output_dir=tmp_path)
    data = json.loads((tmp_path / "global_importance_m.json").read_text())
    assert data["model"] == "m"
    assert data["features"] == imp


def test_save_dae_errors_round_trip(tmp_path):
    sq = np.zeros((3, 5))
    werr = np.ones((3, 5))
    fw = np.array([0.1, 0.2, 0.3, 0.2, 0.2])
    save_dae_errors(sq, werr, fw, ["a", "b", "c", "d", "e"], output_dir=tmp_path)
    data = np.load(tmp_path / "dae_feature_errors.npz")
    assert "per_feature_error" in data
    assert "weighted_per_feature_error" in data
    assert "feature_weights" in data


# ── Config exports ──────────────────────────────────────────────────


def test_export_feature_concepts(tmp_path):
    export_feature_concepts(output_dir=tmp_path)
    assert (tmp_path / "feature_concepts.json").exists()
    content = json.loads((tmp_path / "feature_concepts.json").read_text())
    assert "Pulse_Rate" in content


def test_export_nlg_templates(tmp_path):
    export_nlg_templates(output_dir=tmp_path)
    assert (tmp_path / "nlg_templates.json").exists()
    content = json.loads((tmp_path / "nlg_templates.json").read_text())
    assert "severity_header" in content
