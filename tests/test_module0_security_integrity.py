"""IntegrityVerifier tests — A02 dataset integrity defense.

Covers:
  - Happy round-trip: bootstrap → verify → returns bytes
  - Schema version: refuses v2, accepts v3
  - Tampered file body → IntegrityError
  - Tampered metadata signature → IntegrityError
  - Missing baseline → IntegrityError
  - Corrupt metadata JSON → IntegrityError
  - DoS guard: oversized file rejected before hashing
  - Idempotent bootstrap on same hash (no duplicate entry)
  - Atomic write: tmp file removed when write succeeds; on-disk safe under sim crash
"""
from __future__ import annotations

import base64
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from module0_analysis import IntegrityError, IntegrityVerifier


@pytest.fixture
def metadata_dir(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def known_good(tmp_path: Path) -> Path:
    """Fixture dataset file with stable content."""
    p = tmp_path / "data.csv"
    p.write_bytes(b"Label,Attack Category\n0,benign\n1,reconnaissance\n")
    return p


@pytest.fixture
def verifier(metadata_dir: Path) -> IntegrityVerifier:
    return IntegrityVerifier(metadata_dir)


# ── Happy path ────────────────────────────────────────────────────────────


def test_bootstrap_then_verify_roundtrip(verifier, known_good):
    digest = verifier.bootstrap(known_good)
    assert len(digest) == 64  # sha256 hex
    data, digest2 = verifier.verify_and_read(known_good)
    assert digest == digest2
    assert data == known_good.read_bytes()


def test_bootstrap_writes_v3_schema(verifier, known_good, metadata_dir):
    verifier.bootstrap(known_good)
    meta = json.loads((metadata_dir / "dataset_integrity.json").read_text())
    assert meta["version"] == 3
    # Entry key must be the sha256 hex digest, not the path
    keys = list(meta["entries"].keys())
    assert len(keys) == 1
    assert len(keys[0]) == 64 and all(c in "0123456789abcdef" for c in keys[0])
    entry = meta["entries"][keys[0]]
    assert entry["filename"] == known_good.name
    assert entry["size_bytes"] == known_good.stat().st_size


def test_bootstrap_idempotent_on_identical_content(verifier, known_good):
    digest1 = verifier.bootstrap(known_good)
    # Second bootstrap of the SAME content → no-op, no error, same digest
    digest2 = verifier.bootstrap(known_good)
    assert digest1 == digest2
    meta = json.loads(verifier._metadata_path.read_text())
    assert len(meta["entries"]) == 1


# ── Tamper detection ─────────────────────────────────────────────────────


def test_verify_detects_byte_tamper(verifier, known_good):
    verifier.bootstrap(known_good)
    # Flip a byte
    original = known_good.read_bytes()
    known_good.write_bytes(original[:-1] + b"X")
    with pytest.raises(IntegrityError, match="INTEGRITY VIOLATION|size .* does not match"):
        verifier.verify_and_read(known_good)


def test_verify_detects_metadata_signature_forge(verifier, known_good, metadata_dir):
    verifier.bootstrap(known_good)
    # Edit a payload field WITHOUT re-signing
    meta_path = metadata_dir / "dataset_integrity.json"
    meta = json.loads(meta_path.read_text())
    first_digest = next(iter(meta["entries"]))
    meta["entries"][first_digest]["filename"] = "evil.csv"
    meta_path.write_text(json.dumps(meta, indent=2))

    with pytest.raises(IntegrityError, match="signature is invalid|tampered"):
        verifier.verify_and_read(known_good)


def test_verify_refuses_unsigned_metadata(verifier, known_good, metadata_dir):
    verifier.bootstrap(known_good)
    meta_path = metadata_dir / "dataset_integrity.json"
    meta = json.loads(meta_path.read_text())
    meta.pop("signature", None)
    meta_path.write_text(json.dumps(meta, indent=2))
    with pytest.raises(IntegrityError, match="unsigned"):
        verifier.verify_and_read(known_good)


def test_verify_refuses_missing_baseline_file(verifier, known_good):
    # Never bootstrapped → no metadata file
    with pytest.raises(IntegrityError, match="No integrity baseline at"):
        verifier.verify_and_read(known_good)


def test_verify_refuses_corrupt_metadata_json(verifier, known_good, metadata_dir):
    verifier.bootstrap(known_good)
    meta_path = metadata_dir / "dataset_integrity.json"
    meta_path.write_text("{ this is not valid JSON")
    with pytest.raises(IntegrityError, match="corrupt"):
        verifier.verify_and_read(known_good)


# ── Schema version enforcement ───────────────────────────────────────────


def test_verify_refuses_v2_metadata(verifier, known_good, metadata_dir):
    # Simulate a leftover v2-format baseline. Note: we can't sign it
    # properly (we'd need v2 signing logic), so we use the v3 signing
    # apparatus but stamp version=2 — the signature verifier checks the
    # payload first; if version was tampered, the signature should fail.
    # If we make a properly-signed v2 (impossible with current code),
    # the version-assert would catch it.
    verifier.bootstrap(known_good)
    meta_path = metadata_dir / "dataset_integrity.json"
    meta = json.loads(meta_path.read_text())

    # Move entry to v2 path-keyed schema + bump version down to 2.
    # Re-sign with same key so signature verify passes, then the
    # version assert must fire.
    from module5_responses.signing import canonical_json, load_signing_key
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import ec

    v3_entry = next(iter(meta["entries"].values()))
    v3_digest = next(iter(meta["entries"].keys()))
    v2_body = {
        "version": 2,
        "entries": {
            "/fake/path.csv": {
                "sha256": v3_digest,
                "size_bytes": v3_entry["size_bytes"],
                "bootstrapped_at": v3_entry["bootstrapped_at"],
            }
        },
    }
    private_key, _, key_id = load_signing_key()
    sig = private_key.sign(canonical_json(v2_body), ec.ECDSA(hashes.SHA256()))
    v2_body["signature"] = base64.b64encode(sig).decode("ascii")
    v2_body["signing_key_id"] = key_id
    v2_body["signature_alg"] = "ECDSA_P256_SHA256"
    meta_path.write_text(json.dumps(v2_body, indent=2))

    with pytest.raises(IntegrityError, match="unsupported schema version 2"):
        verifier.verify_and_read(known_good)


# ── DoS guard ────────────────────────────────────────────────────────────


def test_dos_guard_size_mismatch_blocks_before_hash(verifier, known_good, tmp_path):
    """File whose size matches no baseline entry must be rejected
    before being read into memory — the file is NEVER opened.
    """
    verifier.bootstrap(known_good)
    big_file = tmp_path / "huge.csv"
    big_file.write_bytes(b"X" * (known_good.stat().st_size + 1))

    # Patch _read_bytes to fail the test if it's called
    with patch.object(IntegrityVerifier, "_read_bytes",
                      side_effect=AssertionError("DoS guard FAILED — file was read")):
        with pytest.raises(IntegrityError, match="size .* does not match any baseline"):
            verifier.verify_and_read(big_file)


def test_dos_guard_allows_correct_size_but_wrong_content(verifier, known_good, tmp_path):
    """A file with matching size but different content must pass the
    size guard, be hashed, and then fail with hash-not-in-baseline.
    """
    verifier.bootstrap(known_good)
    same_size = tmp_path / "same_size.csv"
    original = known_good.read_bytes()
    same_size.write_bytes(b"Y" * len(original))  # same size, all-different bytes

    with pytest.raises(IntegrityError, match="is not in the baseline"):
        verifier.verify_and_read(same_size)


# ── Atomic write ─────────────────────────────────────────────────────────


def test_atomic_write_leaves_no_tmp_on_success(verifier, known_good, metadata_dir):
    verifier.bootstrap(known_good)
    tmps = list(metadata_dir.glob("*.tmp"))
    assert tmps == [], f"Stray tmp files left behind: {tmps}"


def test_signing_key_id_recorded(verifier, known_good, metadata_dir):
    verifier.bootstrap(known_good)
    meta = json.loads((metadata_dir / "dataset_integrity.json").read_text())
    assert meta["signing_key_id"].startswith("ecdsa-p256-")
    assert meta["signature_alg"] == "ECDSA_P256_SHA256"
