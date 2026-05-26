"""_sidecar_io tests — atomic write, pkl migration, format-tag check."""
from __future__ import annotations

import json

import pytest

from module1_preprocessing.phase1._sidecar_io import (
    atomic_write_json,
    load_sidecar,
    migrate_legacy_pkl,
)


# ── atomic_write_json ────────────────────────────────────────────────


def test_atomic_write_creates_file(tmp_path):
    p = tmp_path / "x.json"
    atomic_write_json(p, {"a": 1, "b": 2})
    assert p.exists()
    assert json.loads(p.read_text()) == {"a": 1, "b": 2}


def test_atomic_write_leaves_no_tmp(tmp_path):
    p = tmp_path / "x.json"
    atomic_write_json(p, {"a": 1})
    assert not (tmp_path / "x.json.tmp").exists()


def test_atomic_write_creates_parent_dirs(tmp_path):
    p = tmp_path / "deeply" / "nested" / "x.json"
    atomic_write_json(p, {"a": 1})
    assert p.exists()


# ── migrate_legacy_pkl ───────────────────────────────────────────────


def test_migrate_no_op_on_json_path(tmp_path):
    p = tmp_path / "x.json"
    assert migrate_legacy_pkl(p, "encoder") == p


def test_migrate_rewrites_pkl_to_json(tmp_path):
    p = tmp_path / "x.pkl"
    out = migrate_legacy_pkl(p, "encoder")
    assert out.suffix == ".json"
    assert out.name == "x.json"


def test_migrate_deletes_existing_pkl(tmp_path):
    pkl = tmp_path / "x.pkl"
    pkl.write_bytes(b"legacy garbage")
    out = migrate_legacy_pkl(pkl, "encoder")
    assert not pkl.exists()
    assert out.suffix == ".json"


# ── load_sidecar ─────────────────────────────────────────────────────


def test_load_sidecar_validates_format(tmp_path):
    p = tmp_path / "x.json"
    p.write_text(json.dumps({"format": "phase1.encoder.v1", "data": 1}))
    body = load_sidecar(p, "phase1.encoder.v1", "encoder")
    assert body["data"] == 1


def test_load_sidecar_rejects_wrong_format(tmp_path):
    p = tmp_path / "x.json"
    p.write_text(json.dumps({"format": "wrong"}))
    with pytest.raises(ValueError, match="not a phase1.encoder.v1 sidecar"):
        load_sidecar(p, "phase1.encoder.v1", "encoder")


def test_load_sidecar_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="Encoder sidecar"):
        load_sidecar(tmp_path / "missing.json", "phase1.encoder.v1", "encoder")
