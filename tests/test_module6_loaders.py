"""Module 6 loaders — pure (non-Streamlit) variants of dashboard load_*."""
from __future__ import annotations

import json

import pytest

from module6_evaluation.loaders import (
    ENRICH_KEYS,
    LoaderError,
    enrich_with_device_context,
    load_provenance_inner,
    load_responses_inner,
)


def test_load_responses_none_returns_empty():
    assert load_responses_inner(None) == []


def test_load_responses_invalid_split_raises():
    with pytest.raises(RuntimeError, match="Refusing"):
        load_responses_inner("staging")


def test_load_responses_missing_file_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setattr("module6_evaluation.loaders.EVAL_DIR", tmp_path)
    out = load_responses_inner("test")
    assert out == []


def test_load_responses_legacy_list_shape(tmp_path, monkeypatch):
    monkeypatch.setattr("module6_evaluation.loaders.EVAL_DIR", tmp_path)
    (tmp_path / "alert_responses.json").write_text(
        json.dumps([{"sample_index": 0, "alert_id": "A0", "risk_level": "LOW",
                     "risk_score": 0.1, "attack_category": "normal",
                     "ground_truth": "benign", "response": {"actions": []}}])
    )
    out = load_responses_inner("test")
    assert len(out) == 1
    assert out[0]["alert_id"] == "A0"


def test_load_responses_unknown_shape_raises(tmp_path, monkeypatch):
    monkeypatch.setattr("module6_evaluation.loaders.EVAL_DIR", tmp_path)
    # Neither bare list nor envelope.
    (tmp_path / "alert_responses.json").write_text('{"unexpected": "shape"}')
    with pytest.raises(LoaderError):
        load_responses_inner("test")


# ── load_provenance_inner ──────────────────────────────────────────────


def test_load_provenance_none_returns_none():
    assert load_provenance_inner(None) is None


def test_load_provenance_invalid_split_returns_none():
    assert load_provenance_inner("staging") is None


def test_load_provenance_missing_file(tmp_path, monkeypatch):
    monkeypatch.setattr("module6_evaluation.loaders.EVAL_DIR", tmp_path)
    assert load_provenance_inner("test") is None


def test_load_provenance_legacy_list_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr("module6_evaluation.loaders.EVAL_DIR", tmp_path)
    (tmp_path / "alert_responses.json").write_text("[]")
    assert load_provenance_inner("test") is None


def test_load_provenance_envelope_returns_block(tmp_path, monkeypatch):
    monkeypatch.setattr("module6_evaluation.loaders.EVAL_DIR", tmp_path)
    (tmp_path / "alert_responses.json").write_text(json.dumps({
        "_provenance": {"split": "test", "n_alerts_emitted": 12},
        "records": [],
    }))
    prov = load_provenance_inner("test")
    assert prov is not None
    assert prov["split"] == "test"
    assert prov["n_alerts_emitted"] == 12


# ── enrich_with_device_context ─────────────────────────────────────────


def test_enrich_no_eval_alerts_file_passes_through(tmp_path, monkeypatch):
    monkeypatch.setattr("module6_evaluation.loaders.EVAL_DIR", tmp_path)
    responses = [{"sample_index": 0}]
    out = enrich_with_device_context(responses, split="test")
    assert out == [{"sample_index": 0}]


def test_enrich_adds_device_fields(tmp_path, monkeypatch):
    monkeypatch.setattr("module6_evaluation.loaders.EVAL_DIR", tmp_path)
    (tmp_path / "evaluation_alerts.json").write_text(json.dumps([{
        "sample_index": 0,
        "device_class": "ECG monitor",
        "device_criticality": "HIGH",
        "affected_system": "vital monitoring",
        "patient_care_impact": "loss of vitals",
        "active_device": True,
        "correct_action": "isolate",
    }]))
    responses = [{"sample_index": 0}]
    out = enrich_with_device_context(responses, split="test")
    assert out[0]["device_class"] == "ECG monitor"
    assert out[0]["device_criticality"] == "HIGH"
    assert out[0]["correct_action"] == "isolate"


def test_enrich_does_not_overwrite_existing_fields(tmp_path, monkeypatch):
    monkeypatch.setattr("module6_evaluation.loaders.EVAL_DIR", tmp_path)
    (tmp_path / "evaluation_alerts.json").write_text(json.dumps([{
        "sample_index": 0,
        "device_class": "FROM_EVAL",
    }]))
    responses = [{"sample_index": 0, "device_class": "PRESERVED"}]
    out = enrich_with_device_context(responses, split="test")
    assert out[0]["device_class"] == "PRESERVED"


def test_enrich_keys_stable():
    """Schema guard — ENRICH_KEYS controls what fields cross over from
    evaluation_alerts.json into the dashboard's alert_responses payload.
    """
    expected = {
        "device_class", "device_criticality", "affected_system",
        "patient_care_impact", "active_device", "correct_action",
    }
    assert set(ENRICH_KEYS) == expected
