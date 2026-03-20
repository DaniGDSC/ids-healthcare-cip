"""Tests for ArtifactIntegrityChecker."""

from __future__ import annotations

import json

from src.phase7_monitoring_engine.phase7.alert_dispatcher import AlertDispatcher
from src.phase7_monitoring_engine.phase7.config import EngineEntry, Phase7Config
from src.phase7_monitoring_engine.phase7.integrity_checker import (
    ArtifactIntegrityChecker,
)


def _make_config(**overrides):
    defaults = {
        "engines": [
            EngineEntry(
                id="test_eng",
                heartbeat_topic="test.hb",
                artifact_dir="artifacts",
                metadata_path="metadata.json",
            )
        ],
    }
    defaults.update(overrides)
    return Phase7Config(**defaults)


class TestArtifactIntegrityChecker:
    """Test SHA-256 artifact verification."""

    def test_compute_sha256(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello world")
        h = ArtifactIntegrityChecker.compute_sha256(f)
        assert len(h) == 64
        assert isinstance(h, str)

    def test_compute_sha256_deterministic(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("deterministic content")
        h1 = ArtifactIntegrityChecker.compute_sha256(f)
        h2 = ArtifactIntegrityChecker.compute_sha256(f)
        assert h1 == h2

    def test_verify_all_artifacts_metadata_missing(self, tmp_path):
        cfg = _make_config()
        d = AlertDispatcher()
        checker = ArtifactIntegrityChecker(cfg, d, tmp_path)
        events = checker.verify_all_artifacts()
        assert len(events) == 1
        assert events[0].event_type == "METADATA_MISSING"

    def test_verify_all_artifacts_hash_verified(self, tmp_path):
        # Create artifact and metadata
        art_dir = tmp_path / "artifacts"
        art_dir.mkdir()
        artifact = art_dir / "model.h5"
        artifact.write_text("model data")

        expected_hash = ArtifactIntegrityChecker.compute_sha256(artifact)
        metadata = {"artifact_hashes": {"model.h5": {"sha256": expected_hash}}}
        meta_path = tmp_path / "metadata.json"
        meta_path.write_text(json.dumps(metadata))

        cfg = _make_config()
        d = AlertDispatcher()
        checker = ArtifactIntegrityChecker(cfg, d, tmp_path)
        events = checker.verify_all_artifacts()
        assert len(events) == 1
        assert events[0].event_type == "HASH_VERIFIED"

    def test_verify_all_artifacts_hash_mismatch(self, tmp_path):
        art_dir = tmp_path / "artifacts"
        art_dir.mkdir()
        artifact = art_dir / "model.h5"
        artifact.write_text("model data")

        metadata = {"artifact_hashes": {"model.h5": {"sha256": "wrong_hash_value"}}}
        meta_path = tmp_path / "metadata.json"
        meta_path.write_text(json.dumps(metadata))

        cfg = _make_config()
        d = AlertDispatcher()
        checker = ArtifactIntegrityChecker(cfg, d, tmp_path)
        events = checker.verify_all_artifacts()
        assert events[0].event_type == "HASH_MISMATCH"
        assert events[0].severity == "CRITICAL"

    def test_verify_all_artifacts_artifact_missing(self, tmp_path):
        art_dir = tmp_path / "artifacts"
        art_dir.mkdir()
        metadata = {"artifact_hashes": {"nonexistent.h5": {"sha256": "abc"}}}
        meta_path = tmp_path / "metadata.json"
        meta_path.write_text(json.dumps(metadata))

        cfg = _make_config()
        d = AlertDispatcher()
        checker = ArtifactIntegrityChecker(cfg, d, tmp_path)
        events = checker.verify_all_artifacts()
        assert events[0].event_type == "ARTIFACT_MISSING"

    def test_verify_baseline_config_init(self, tmp_path):
        bl = tmp_path / "data" / "phase4" / "baseline_config.json"
        bl.parent.mkdir(parents=True)
        bl.write_text('{"key": "value"}')

        cfg = _make_config()
        d = AlertDispatcher()
        checker = ArtifactIntegrityChecker(cfg, d, tmp_path)
        event = checker.verify_baseline_config()
        assert event is not None
        assert event.event_type == "BASELINE_INITIALIZED"

    def test_verify_baseline_config_no_change(self, tmp_path):
        bl = tmp_path / "data" / "phase4" / "baseline_config.json"
        bl.parent.mkdir(parents=True)
        bl.write_text('{"key": "value"}')

        cfg = _make_config()
        d = AlertDispatcher()
        checker = ArtifactIntegrityChecker(cfg, d, tmp_path)
        checker.verify_baseline_config()  # init
        event = checker.verify_baseline_config()  # no change
        assert event is None

    def test_verify_baseline_config_tampered(self, tmp_path):
        bl = tmp_path / "data" / "phase4" / "baseline_config.json"
        bl.parent.mkdir(parents=True)
        bl.write_text('{"key": "value"}')

        cfg = _make_config()
        d = AlertDispatcher()
        checker = ArtifactIntegrityChecker(cfg, d, tmp_path)
        checker.verify_baseline_config()  # init
        bl.write_text('{"key": "tampered"}')
        event = checker.verify_baseline_config()
        assert event is not None
        assert event.event_type == "BASELINE_TAMPER_DETECTED"
        assert event.severity == "CRITICAL"

    def test_verify_baseline_config_missing(self, tmp_path):
        cfg = _make_config()
        d = AlertDispatcher()
        checker = ArtifactIntegrityChecker(cfg, d, tmp_path)
        event = checker.verify_baseline_config()
        assert event is None

    def test_get_status(self, tmp_path):
        cfg = _make_config()
        d = AlertDispatcher()
        checker = ArtifactIntegrityChecker(cfg, d, tmp_path)
        status = checker.get_status()
        assert status.name == "ArtifactIntegrityChecker"
        assert status.details["verified"] == 0

    def test_metadata_read_error(self, tmp_path):
        meta_path = tmp_path / "metadata.json"
        meta_path.write_text("not valid json{{{")

        cfg = _make_config()
        d = AlertDispatcher()
        checker = ArtifactIntegrityChecker(cfg, d, tmp_path)
        events = checker.verify_all_artifacts()
        assert events[0].event_type == "METADATA_READ_ERROR"
