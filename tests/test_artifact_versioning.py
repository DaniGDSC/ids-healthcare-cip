"""Tests for ``common.artifact_versioning`` (Sprint 6 / Tầng 3.5).

Pins the version-registry semantics and the version-gate behaviour:

  - read_version handles json + npz formats
  - check_compatibility distinguishes match / mismatch / missing
  - normalisation strips per-split / per-model suffixes
  - embed_version_in_dict produces idempotent output
  - version_kwarg_for produces a serialisable scalar
  - the gate's pending list keeps "expected migration" artifacts inert
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


# ── Registry shape ───────────────────────────────────────────────


def test_artifact_versions_use_semver_strings():
    """Every registered version must be parseable as ``X.Y`` so the
    semver-bump rules in the docstring stay true."""
    from common.artifact_versioning import ARTIFACT_VERSIONS
    for name, version in ARTIFACT_VERSIONS.items():
        parts = version.split(".")
        assert len(parts) in {2, 3}, (
            f"{name}: version {version!r} should be X.Y or X.Y.Z"
        )
        assert all(p.isdigit() for p in parts), (
            f"{name}: version {version!r} parts must be digits"
        )


def test_pending_envelope_migration_is_disjoint_from_registry():
    """A file is either *registered* (versioned) or *pending migration*
    (gate-skipped) — never both. Otherwise the gate's intent is
    ambiguous."""
    from common.artifact_versioning import (
        ARTIFACT_VERSIONS, PENDING_ENVELOPE_MIGRATION,
    )
    overlap = set(ARTIFACT_VERSIONS.keys()) & PENDING_ENVELOPE_MIGRATION
    assert not overlap, (
        f"Artifacts {overlap} appear in both ARTIFACT_VERSIONS and "
        f"PENDING_ENVELOPE_MIGRATION"
    )


# ── Normalisation ─────────────────────────────────────────────────


def test_normalise_strips_demo_suffix():
    from common.artifact_versioning import _normalise_artifact_name
    assert _normalise_artifact_name("results/reports/alert_responses_demo.json") \
        == "alert_responses.json"


def test_normalise_strips_test_suffix():
    from common.artifact_versioning import _normalise_artifact_name
    assert _normalise_artifact_name("shap_values_xgboost_test.npz") \
        == "shap_values_xgboost.npz"


def test_normalise_maps_demo_scores_to_risk_scores():
    from common.artifact_versioning import _normalise_artifact_name
    assert _normalise_artifact_name("results/reports/demo_scores.npz") \
        == "risk_scores.npz"


def test_normalise_returns_bare_name_when_no_suffix():
    from common.artifact_versioning import _normalise_artifact_name
    assert _normalise_artifact_name("foo.json") == "foo.json"


# ── read_version ──────────────────────────────────────────────────


def test_read_version_json_top_level(tmp_path: Path):
    from common.artifact_versioning import read_version
    p = tmp_path / "x.json"
    p.write_text(json.dumps({"_schema_version": "2.1", "data": 1}))
    assert read_version(p) == "2.1"


def test_read_version_json_in_provenance(tmp_path: Path):
    """Module 5's envelope nests the version under ``_provenance``.
    The reader must look in both places."""
    from common.artifact_versioning import read_version
    p = tmp_path / "alert_responses.json"
    p.write_text(json.dumps({
        "_provenance": {"_schema_version": "3.2", "split": "test"},
        "records": [],
    }))
    assert read_version(p) == "3.2"


def test_read_version_npz(tmp_path: Path):
    from common.artifact_versioning import read_version
    p = tmp_path / "x.npz"
    np.savez(p, schema_version=np.array("1.0", dtype=str), data=np.zeros(5))
    assert read_version(p) == "1.0"


def test_read_version_returns_none_for_missing_field(tmp_path: Path):
    from common.artifact_versioning import read_version
    p = tmp_path / "x.json"
    p.write_text(json.dumps({"foo": "bar"}))
    assert read_version(p) is None


def test_read_version_returns_none_for_nonexistent_file(tmp_path: Path):
    from common.artifact_versioning import read_version
    assert read_version(tmp_path / "does_not_exist.json") is None


# ── check_compatibility ──────────────────────────────────────────


def test_check_compatibility_match(tmp_path: Path):
    from common.artifact_versioning import (
        ARTIFACT_VERSIONS, check_compatibility,
    )
    # Pick any registered artifact and synthesise it
    name = next(iter(ARTIFACT_VERSIONS))
    p = tmp_path / name
    expected = ARTIFACT_VERSIONS[name]
    if name.endswith(".json"):
        p.write_text(json.dumps({"_schema_version": expected}))
    else:
        np.savez(p, schema_version=np.array(expected, dtype=str))
    check = check_compatibility(p)
    assert check.ok
    assert check.on_disk == expected


def test_check_compatibility_mismatch(tmp_path: Path):
    from common.artifact_versioning import check_compatibility
    p = tmp_path / "phase0_baseline.json"
    p.write_text(json.dumps({"_schema_version": "1.0"}))  # registry says "2.1"
    check = check_compatibility(p)
    assert not check.ok
    assert "stale" in check.reason.lower() or "≠" in check.reason


def test_check_compatibility_missing_version_field(tmp_path: Path):
    from common.artifact_versioning import check_compatibility
    p = tmp_path / "phase0_baseline.json"
    p.write_text(json.dumps({"foo": "bar"}))  # no version field
    check = check_compatibility(p)
    assert not check.ok
    assert "no _schema_version" in check.reason.lower() or "rerun" in check.reason.lower()


def test_check_compatibility_unregistered_artifact(tmp_path: Path):
    """Files not in the registry should pass — they're opted out."""
    from common.artifact_versioning import check_compatibility
    p = tmp_path / "some_random_file.json"
    p.write_text("{}")
    check = check_compatibility(p)
    assert check.ok
    assert "opt-out" in check.reason or "no expected" in check.reason


# ── assert_compatible ────────────────────────────────────────────


def test_assert_compatible_raises_on_mismatch(tmp_path: Path):
    from common.artifact_versioning import (
        ArtifactVersionMismatch, assert_compatible,
    )
    p = tmp_path / "phase0_baseline.json"
    p.write_text(json.dumps({"_schema_version": "0.0"}))
    with pytest.raises(ArtifactVersionMismatch):
        assert_compatible(p)


def test_assert_compatible_passes_on_match(tmp_path: Path):
    from common.artifact_versioning import (
        ARTIFACT_VERSIONS, assert_compatible,
    )
    p = tmp_path / "phase0_baseline.json"
    p.write_text(json.dumps({
        "_schema_version": ARTIFACT_VERSIONS["phase0_baseline.json"],
    }))
    assert_compatible(p)  # must not raise


# ── embed_version_in_dict ────────────────────────────────────────


def test_embed_version_top_level():
    from common.artifact_versioning import embed_version_in_dict
    out = embed_version_in_dict({"data": 1}, "phase0_baseline.json")
    assert out["_schema_version"]
    assert out["data"] == 1


def test_embed_version_nests_in_provenance_envelope():
    from common.artifact_versioning import embed_version_in_dict
    payload = {"_provenance": {"split": "test"}, "records": []}
    out = embed_version_in_dict(payload, "alert_responses.json")
    assert out["_provenance"]["_schema_version"]
    assert "_schema_version" not in out  # top-level not touched


def test_embed_version_is_idempotent():
    from common.artifact_versioning import embed_version_in_dict
    payload = {"data": 1}
    once  = embed_version_in_dict(payload, "phase0_baseline.json")
    twice = embed_version_in_dict(once,   "phase0_baseline.json")
    assert once == twice


def test_embed_version_unregistered_passes_through():
    from common.artifact_versioning import embed_version_in_dict
    payload = {"foo": "bar"}
    out = embed_version_in_dict(payload, "totally_unknown.json")
    assert out == payload


# ── version_kwarg_for ────────────────────────────────────────────


def test_version_kwarg_for_known_npz():
    from common.artifact_versioning import (
        ARTIFACT_VERSIONS, version_kwarg_for,
    )
    out = version_kwarg_for("risk_scores.npz")
    assert "schema_version" in out
    assert str(out["schema_version"]) == ARTIFACT_VERSIONS["risk_scores.npz"]


def test_version_kwarg_for_unknown_returns_empty():
    from common.artifact_versioning import version_kwarg_for
    assert version_kwarg_for("unregistered_artifact.npz") == {}


def test_npz_round_trip_via_version_kwarg(tmp_path: Path):
    """Round-trip ``np.savez(... **version_kwarg_for(name))`` →
    ``read_version`` must produce the same string."""
    from common.artifact_versioning import (
        ARTIFACT_VERSIONS, read_version, version_kwarg_for,
    )
    p = tmp_path / "risk_scores.npz"
    np.savez(p, data=np.zeros(5), **version_kwarg_for(p.name))
    assert read_version(p) == ARTIFACT_VERSIONS["risk_scores.npz"]


# ── End-to-end: the version gate behaves on real artifacts ──────


def test_version_gate_passes_on_current_artifacts():
    """Sprint 6 acceptance: after re-running the producers once with
    the embed helpers, every registered artifact must carry the
    expected version. A failure here means a producer didn't get
    the embed call wired up."""
    import subprocess, sys
    result = subprocess.run(
        [sys.executable, "-m", "tools.version_gate", "--check"],
        cwd=Path(__file__).resolve().parent.parent,
        capture_output=True, text=True,
    )
    assert result.returncode == 0, (
        f"version gate failing:\n{result.stdout}\n{result.stderr}"
    )
