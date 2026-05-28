"""Loader tests for simulation_stream — the M6 dashboard's Full-stream input.

Covers the pure (non-Streamlit) layer in :mod:`module6_evaluation.loaders`:

* missing-file → empty list (no crash, lets UI fall back to alerts-only)
* unknown split → empty list (guarded against typos)
* malformed payload (no 'stream' key) → :class:`LoaderError`
* meta accessor returns the artefact's ``_meta`` block verbatim
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from module6_evaluation.loaders import (
    EVAL_DIR,
    LoaderError,
    load_simulation_stream_inner,
    load_simulation_stream_meta_inner,
)


# ── Fallback semantics ────────────────────────────────────────────────────


def test_unknown_split_returns_empty_list():
    """Refuse silently rather than raising so the UI can default to
    alerts-only mode without an error toast."""
    assert load_simulation_stream_inner("validation") == []
    assert load_simulation_stream_inner(None) == []


def test_meta_unknown_split_returns_none():
    assert load_simulation_stream_meta_inner("validation") is None
    assert load_simulation_stream_meta_inner(None) is None


def test_missing_artefact_returns_empty_list(tmp_path, monkeypatch):
    """When simulation_stream_<split>.json is absent the loader must
    return an empty list — the freshly-cloned repo path."""
    # Redirect EVAL_DIR to an empty tmp dir so the file is guaranteed absent.
    from module6_evaluation import loaders as _loaders_mod
    monkeypatch.setattr(_loaders_mod, "EVAL_DIR", tmp_path)
    assert _loaders_mod.load_simulation_stream_inner("demo") == []
    assert _loaders_mod.load_simulation_stream_meta_inner("demo") is None


# ── Schema enforcement ───────────────────────────────────────────────────


def test_malformed_payload_raises_loader_error(tmp_path, monkeypatch):
    """A simulation_stream JSON missing the ``stream`` key is a build bug,
    not a benign missing file. The loader raises LoaderError so the UI
    can show an actionable error (run the builder again) instead of
    silently falling back to alerts-only and hiding the regression.
    """
    from module6_evaluation import loaders as _loaders_mod
    bad_path = tmp_path / "simulation_stream_demo.json"
    bad_path.write_text(json.dumps({"_meta": {"split": "demo"}, "records": []}))
    monkeypatch.setattr(_loaders_mod, "EVAL_DIR", tmp_path)
    with pytest.raises(LoaderError, match="missing 'stream' array"):
        _loaders_mod.load_simulation_stream_inner("demo")


# ── Live artefact (skip cleanly when absent) ─────────────────────────────


def test_demo_stream_matches_meta_when_present():
    if not (EVAL_DIR / "simulation_stream_demo.json").exists():
        pytest.skip(
            "simulation_stream_demo.json not built — run "
            "`python -m tools.build_simulation_stream --split demo` first."
        )
    stream = load_simulation_stream_inner("demo")
    meta = load_simulation_stream_meta_inner("demo")
    assert meta is not None
    assert len(stream) == meta["n_total"]
    # Each entry has sample_index and risk_level — these are the two
    # fields the dashboard's iteration loop depends on.
    assert all("sample_index" in e and "risk_level" in e for e in stream[:10])
