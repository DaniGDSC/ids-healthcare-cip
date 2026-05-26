"""module2_train_models tests — load_data delegation, leakage guard,
strip_prefix, _resolve_random_state, evaluate metrics shape.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from module2_detection.module2_train_models import (
    DEFAULT_RANDOM_STATE,
    RANDOM_STATE,
    _FORBIDDEN_TRAINING_PARQUETS,
    _assert_no_demo_leakage,
    _resolve_random_state,
    evaluate,
    strip_prefix,
)


# ── strip_prefix ──────────────────────────────────────────────────────


def test_strip_prefix_removes_classifier_prefix():
    raw = {"classifier__n_estimators": 100, "classifier__max_depth": 5}
    out = strip_prefix(raw)
    assert out == {"n_estimators": 100, "max_depth": 5}


def test_strip_prefix_custom_prefix():
    raw = {"smote__k": 5, "classifier__lr": 0.1}
    out = strip_prefix(raw, prefix="smote__")
    assert out == {"k": 5, "classifier__lr": 0.1}


def test_strip_prefix_empty_dict():
    assert strip_prefix({}) == {}


def test_strip_prefix_no_match_preserves_keys():
    raw = {"foo": 1, "bar": 2}
    assert strip_prefix(raw) == {"foo": 1, "bar": 2}


# ── Re-exported leakage guard ─────────────────────────────────────────


def test_re_exports_match_tuning_data():
    """module2_train_models should re-export the canonical guard."""
    from module2_detection.tuning._data import (
        _FORBIDDEN_TRAINING_PARQUETS as canonical,
    )
    assert _FORBIDDEN_TRAINING_PARQUETS is canonical


def test_re_exported_assert_no_demo_leakage_blocks_demo(tmp_path):
    """Re-exported guard must work identically."""
    with pytest.raises(RuntimeError, match="demo_phase1"):
        _assert_no_demo_leakage(tmp_path / "demo_phase1.parquet")


# ── _resolve_random_state (Y6 fix) ────────────────────────────────────


def test_resolve_random_state_falls_back_when_report_missing(tmp_path):
    fake_params = tmp_path / "xgboost_best_params.json"
    fake_params.write_text("{}")  # exists but no report next to it
    seed = _resolve_random_state(fake_params)
    assert seed == DEFAULT_RANDOM_STATE == 42


def test_resolve_random_state_reads_seed_from_report(tmp_path):
    """When tuning report carries data.random_state, propagate it."""
    fake_params = tmp_path / "xgboost_best_params.json"
    fake_params.write_text("{}")
    # Runner writes data.random_state into the report alongside best_params.
    report = tmp_path / "xgboost_report.json"
    report.write_text(json.dumps({
        "data": {"random_state": 1337},
        "best_params": {},
    }))
    seed = _resolve_random_state(fake_params)
    assert seed == 1337


def test_resolve_random_state_corrupt_report_falls_back(tmp_path):
    fake_params = tmp_path / "xgboost_best_params.json"
    fake_params.write_text("{}")
    report = tmp_path / "xgboost_report.json"
    report.write_text("{ not valid json")
    seed = _resolve_random_state(fake_params)
    assert seed == DEFAULT_RANDOM_STATE


def test_legacy_random_state_alias_still_42():
    """`RANDOM_STATE` retained for any external consumer."""
    assert RANDOM_STATE == DEFAULT_RANDOM_STATE == 42


# ── evaluate() metrics shape ──────────────────────────────────────────


def test_evaluate_returns_canonical_metrics():
    y_test = np.array([0, 0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1, 0, 0])
    y_proba = np.array([0.1, 0.6, 0.7, 0.8, 0.2, 0.4])
    m = evaluate("test_model", y_test, y_pred, y_proba, threshold=0.5)
    for key in ("attack_f1", "attack_f2", "weighted_f1", "macro_f1",
                "auc_roc", "optimal_threshold"):
        assert key in m
    assert m["optimal_threshold"] == 0.5


# ── load_data delegation ──────────────────────────────────────────────


def test_load_data_delegates_to_tuning_data_module(monkeypatch, tmp_path):
    """Verify module2_train_models.load_data is now a thin wrapper around
    tuning._data.load_data (post-C3 consolidation)."""
    from module2_detection import module2_train_models as m2

    captured: dict = {}

    def fake_load(train_path, test_path, label_col="Label"):
        captured["train_path"] = train_path
        captured["test_path"] = test_path
        captured["label_col"] = label_col
        return (
            np.zeros((1, 1)), np.zeros((1, 1)),
            np.zeros(1), np.zeros(1),
            ["f1"],
        )

    monkeypatch.setattr(m2, "_load_data_shared", fake_load)
    m2.load_data(label_col="Label")
    assert captured["train_path"].name == "train_phase1.parquet"
    assert captured["test_path"].name == "test_phase1.parquet"
    assert captured["label_col"] == "Label"
