"""Reproducibility-manifest integrity (chapter §4.5 — frozen evaluation pool).

The manifest at ``results/reports/evaluation_manifest.json`` pins the
raw WUSTL-EHMS-2020 CSV + the three downstream "frozen" artefacts the
dashboard reads from. These tests rehash each file from disk and
assert the manifest agrees, then assert structural invariants:

  * format string is ``evaluation_manifest.v1``
  * splitter seed equals the seed recorded in
    ``data/processed/split_metadata.yaml``
  * each artifact record carries ``sha256``, ``size``, ``mtime``
  * the source-dataset SHA matches ``split_metadata.yaml``'s
    ``source_dataset_sha256`` field (one chain, one truth)

Run::

    python3 -m pytest tests/test_evaluation_manifest_integrity.py -v

If the manifest is missing or stale, the helper test
``test_manifest_can_be_rebuilt`` rebuilds it in-memory and asserts the
freshly-computed hashes equal the on-disk hashes — i.e. the artefacts
haven't drifted since the manifest was written.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tools.build_evaluation_manifest import (
    ARTIFACTS,
    MANIFEST_PATH,
    MANIFEST_VERSION,
    SPLIT_METADATA,
    build_manifest,
)


@pytest.fixture(scope="module")
def manifest() -> dict:
    if not MANIFEST_PATH.exists():
        pytest.skip(
            f"{MANIFEST_PATH} missing — run "
            "`python3 -m tools.build_evaluation_manifest` first."
        )
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def test_manifest_format_version(manifest):
    assert manifest.get("format") == MANIFEST_VERSION


def test_manifest_has_required_top_level_keys(manifest):
    for k in ("format", "generated_at", "splitter_seed", "artifacts"):
        assert k in manifest, f"manifest missing top-level key {k!r}"


def test_manifest_covers_all_expected_artifacts(manifest):
    expected = {name for name, _ in ARTIFACTS}
    actual = set(manifest["artifacts"].keys())
    assert actual == expected, (
        f"manifest artifacts {actual} != expected {expected}"
    )


def test_artifact_records_carry_required_fields(manifest):
    required = {"path", "exists", "sha256", "size", "mtime"}
    for name, record in manifest["artifacts"].items():
        missing = required - set(record.keys())
        assert not missing, f"{name}: missing fields {missing}"


@pytest.mark.parametrize("name,_path", ARTIFACTS)
def test_artifact_sha256_matches_disk(manifest, name, _path):
    """Rehash each file from disk and assert manifest agrees."""
    record = manifest["artifacts"][name]
    if not record["exists"]:
        pytest.skip(f"{name} missing on disk — manifest correctly records absence")
    disk_path = PROJECT_ROOT / record["path"]
    actual_sha = hashlib.sha256(disk_path.read_bytes()).hexdigest()
    assert actual_sha == record["sha256"], (
        f"{name}: on-disk SHA {actual_sha[:16]}… != manifest "
        f"{record['sha256'][:16]}… — artefact drifted; rebuild manifest."
    )


def test_manifest_splitter_seed_matches_split_metadata(manifest):
    """The seed in the manifest is copied from split_metadata.yaml — they
    must agree, otherwise the manifest's reproducibility claim is wrong."""
    if not SPLIT_METADATA.exists():
        pytest.skip("split_metadata.yaml missing")
    body = yaml.safe_load(SPLIT_METADATA.read_text(encoding="utf-8"))
    assert manifest["splitter_seed"] == body.get("random_state")


def test_source_sha_matches_split_metadata(manifest):
    """The source-dataset SHA in the manifest must equal the
    ``source_dataset_sha256`` field in split_metadata.yaml — single chain
    of custody from raw bytes → splits → dashboard."""
    if not SPLIT_METADATA.exists():
        pytest.skip("split_metadata.yaml missing")
    body = yaml.safe_load(SPLIT_METADATA.read_text(encoding="utf-8"))
    yaml_sha = body.get("source_dataset_sha256", "")
    manifest_sha = manifest["artifacts"]["source_dataset"]["sha256"]
    if not yaml_sha:
        pytest.skip(
            "split_metadata.yaml has empty source_dataset_sha256; run "
            "`python3 -m tools.build_evaluation_manifest --patch-split-metadata`"
        )
    assert yaml_sha == manifest_sha, (
        f"chain-of-custody broken: split_metadata.yaml SHA "
        f"{yaml_sha[:16]}… != manifest source-dataset SHA "
        f"{manifest_sha[:16]}…"
    )


def test_manifest_can_be_rebuilt():
    """Rebuilding the manifest in-memory must produce the same SHAs as
    the on-disk manifest — i.e. nothing has drifted between writes."""
    fresh = build_manifest()
    if not MANIFEST_PATH.exists():
        pytest.skip("on-disk manifest missing — nothing to compare against")
    on_disk = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    for name in fresh["artifacts"]:
        fresh_sha = fresh["artifacts"][name]["sha256"]
        disk_sha = on_disk["artifacts"][name]["sha256"]
        if fresh["artifacts"][name]["exists"]:
            assert fresh_sha == disk_sha, (
                f"{name}: rebuilt SHA {fresh_sha[:16]}… differs from "
                f"on-disk manifest {disk_sha[:16]}… — artefact changed "
                f"since manifest was written; re-run "
                f"`python3 -m tools.build_evaluation_manifest`."
            )
