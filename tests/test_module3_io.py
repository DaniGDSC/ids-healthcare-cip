"""Module 3 io — save_outputs + export_config_jsons + load_test_data."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from module3_risk_scoring.io import (
    _split_paths,
    export_config_jsons,
    load_test_data,
    save_outputs,
)


@pytest.fixture
def synthetic_state():
    rng = np.random.default_rng(42)
    n = 50
    R = rng.uniform(0, 1, n)
    levels = np.array(["LOW", "MEDIUM", "HIGH", "CRITICAL"] * 12 + ["LOW"] * 2)
    y = np.array([0] * 25 + [1] * 25)
    cats = np.array(["normal"] * 25 + ["Spoofing"] * 12 + ["Data Alteration"] * 13)
    c_detect = rng.uniform(0, 1, n)
    c_track_a = rng.uniform(0, 1, n)
    c_track_b = rng.uniform(0, 1, n)
    d_crit = rng.uniform(0, 1, n)
    s_data = rng.uniform(0, 1, n)
    d_clinical_tier = rng.uniform(0, 1, n)
    fusion = {
        "quadrants": {
            "both_flag": {"total": 10, "true_attacks": 8, "true_benign": 2, "attack_categories": {"Spoofing": 4}},
            "only_xgboost": {"total": 5, "true_attacks": 3, "true_benign": 2, "attack_categories": {}},
            "only_dae": {"total": 3, "true_attacks": 1, "true_benign": 2, "attack_categories": {}},
            "neither": {"total": 32, "true_attacks": 13, "true_benign": 19, "attack_categories": {}},
        },
        "xgb_threshold": 0.5,
        "dae_threshold": 0.5,
        "recall": {"xgboost_alone": 0.5, "dae_alone": 0.4, "union_fusion": 0.6, "best_single_track": 0.5, "fusion_gain": 0.1},
        "total_attacks": 25,
    }
    contributions = {"per_level": {}, "overall_dominant": {}}
    sensitivity = {"best_weights": {"w1": 0.4, "w2": 0.25, "w3": 0.15, "w4": 0.2}, "best_auroc": 0.95}
    worked = []
    return dict(
        R=R, c_detect=c_detect, d_crit=d_crit, s_data=s_data,
        d_clinical_tier=d_clinical_tier, c_track_a=c_track_a, c_track_b=c_track_b,
        levels=levels, y_true=y, attack_cats=cats,
        fusion=fusion, contributions=contributions, sensitivity=sensitivity,
        worked_examples=worked,
    )


# ── save_outputs ─────────────────────────────────────────────────────


def test_save_outputs_writes_npz_csv_json(synthetic_state, tmp_path):
    save_outputs(**synthetic_state, out_npz=tmp_path / "scores.npz",
                 output_dir=tmp_path)
    assert (tmp_path / "scores.npz").exists()
    assert (tmp_path / "risk_scores_detail.csv").exists()
    assert (tmp_path / "risk_report.json").exists()


def test_save_outputs_npz_arrays_round_trip(synthetic_state, tmp_path):
    save_outputs(**synthetic_state, out_npz=tmp_path / "scores.npz",
                 output_dir=tmp_path)
    data = np.load(tmp_path / "scores.npz")
    for key in ("R", "c_detect", "d_crit", "s_data", "d_clinical_tier",
                "c_track_a", "c_track_b", "risk_levels", "y_true"):
        assert key in data


def test_save_outputs_report_json_shape(synthetic_state, tmp_path):
    save_outputs(**synthetic_state, out_npz=tmp_path / "scores.npz",
                 output_dir=tmp_path)
    report = json.loads((tmp_path / "risk_report.json").read_text())
    assert "formula" in report
    assert "weights" in report
    assert "risk_thresholds" in report
    assert "risk_level_distribution" in report
    assert "dual_track_fusion" in report
    assert "weight_sensitivity" in report
    assert "limitations" in report


def test_save_outputs_per_category_derived_from_data(synthetic_state, tmp_path):
    """Y2 fix: per_category_stats includes every attack category present."""
    save_outputs(**synthetic_state, out_npz=tmp_path / "scores.npz",
                 output_dir=tmp_path)
    report = json.loads((tmp_path / "risk_report.json").read_text())
    cats_in_report = set(report["per_category_stats"].keys())
    # Expected: normal + Spoofing + Data Alteration
    assert "normal" in cats_in_report
    assert "Spoofing" in cats_in_report
    assert "Data Alteration" in cats_in_report


def test_save_outputs_handles_new_attack_category(synthetic_state, tmp_path):
    """Y2 — new category in data is reported, not silently dropped."""
    # Add a row with a new category
    state = dict(synthetic_state)
    state["attack_cats"] = np.append(state["attack_cats"], ["Reconnaissance"])
    for key in ("R", "c_detect", "d_crit", "s_data", "d_clinical_tier",
                "c_track_a", "c_track_b"):
        state[key] = np.append(state[key], [0.5])
    state["levels"] = np.append(state["levels"], ["MEDIUM"])
    state["y_true"] = np.append(state["y_true"], [1])
    save_outputs(**state, out_npz=tmp_path / "scores.npz", output_dir=tmp_path)
    report = json.loads((tmp_path / "risk_report.json").read_text())
    assert "Reconnaissance" in report["per_category_stats"]


def test_save_outputs_strict_json_no_default_coerce(synthetic_state, tmp_path):
    """Producer bug surfaces as TypeError, not silent coercion."""
    state = dict(synthetic_state)
    state["worked_examples"] = [object()]  # non-serialisable
    with pytest.raises(TypeError, match="non-JSON-serialisable"):
        save_outputs(**state, out_npz=tmp_path / "scores.npz",
                     output_dir=tmp_path)


# ── export_config_jsons ─────────────────────────────────────────────


def test_export_config_jsons_writes_three_files(tmp_path):
    export_config_jsons(output_dir=tmp_path)
    assert (tmp_path / "device_criticality.json").exists()
    assert (tmp_path / "data_sensitivity.json").exists()
    assert (tmp_path / "risk_config.json").exists()


def test_export_config_jsons_shape(tmp_path):
    export_config_jsons(output_dir=tmp_path)
    risk_cfg = json.loads((tmp_path / "risk_config.json").read_text())
    assert "formula" in risk_cfg
    assert "weights" in risk_cfg
    assert "biometric_features" in risk_cfg
    assert "sigma_threshold" in risk_cfg


# ── load_test_data ──────────────────────────────────────────────────


def test_load_test_data_returns_expected_shape(tmp_path):
    df = pd.DataFrame({
        "f1": [0.1, 0.2, 0.3],
        "f2": [1.0, 2.0, 3.0],
        "Label": [0, 1, 0],
        "Attack Category": ["normal", "Spoofing", "normal"],
        "row_id": [0, 1, 2],
        "device_class": ["patient_monitor"] * 3,
    })
    pq = tmp_path / "test.parquet"
    df.to_parquet(pq, index=False)
    X, y, cats, feat_names = load_test_data(pq)
    assert X.shape == (3, 2)
    assert list(y) == [0, 1, 0]
    assert list(cats) == ["normal", "Spoofing", "normal"]
    assert feat_names == ["f1", "f2"]


def test_load_test_data_handles_missing_attack_category(tmp_path):
    df = pd.DataFrame({
        "f1": [0.1, 0.2],
        "Label": [0, 1],
    })
    pq = tmp_path / "test.parquet"
    df.to_parquet(pq, index=False)
    X, y, cats, feat_names = load_test_data(pq)
    assert cats is None


# ── _split_paths ────────────────────────────────────────────────────


def test_split_paths_test():
    paths = _split_paths("test")
    assert "parquet" in paths
    assert "out_npz" in paths
    assert paths["parquet"].name == "test_phase1.parquet"


def test_split_paths_demo():
    paths = _split_paths("demo")
    assert paths["parquet"].name == "demo_phase1.parquet"
