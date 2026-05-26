"""PathValidator tests — A01 workspace-containment defense.

Covers:
  - Files inside workspace pass
  - Traversal via ``..`` rejected
  - Symlink that escapes workspace rejected (linux only)
  - Missing input file → FileNotFoundError
  - check_read_only enforces in production mode
"""
from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from module0_analysis import PathValidator


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    (tmp_path / "data").mkdir()
    return tmp_path


def test_contained_path_resolves(workspace):
    f = workspace / "data" / "x.csv"
    f.touch()
    validator = PathValidator(workspace)
    assert validator.validate_input_path(f) == f.resolve()


def test_traversal_dotdot_rejected(workspace):
    """A relative path that resolves outside the workspace must be blocked."""
    outside_dir = workspace.parent / "outside-workspace-xyz"
    outside_dir.mkdir(exist_ok=True)
    outside_file = outside_dir / "secret.csv"
    outside_file.touch()
    try:
        validator = PathValidator(workspace)
        with pytest.raises(PermissionError, match="Path escapes workspace"):
            validator.validate_input_path(Path("../outside-workspace-xyz/secret.csv"))
    finally:
        outside_file.unlink()
        outside_dir.rmdir()


def test_symlink_escape_rejected(workspace):
    """Symlink inside workspace pointing outside must be blocked."""
    outside_dir = workspace.parent / "outside-symlink-xyz"
    outside_dir.mkdir(exist_ok=True)
    outside_target = outside_dir / "target.csv"
    outside_target.touch()
    link = workspace / "evil_link.csv"
    try:
        link.symlink_to(outside_target)
        validator = PathValidator(workspace)
        with pytest.raises(PermissionError, match="Path escapes workspace"):
            validator.validate_input_path(link)
    finally:
        if link.exists() or link.is_symlink():
            link.unlink()
        outside_target.unlink()
        outside_dir.rmdir()


def test_nonexistent_input_raises_fnf(workspace):
    validator = PathValidator(workspace)
    with pytest.raises(FileNotFoundError):
        validator.validate_input_path(Path("data/missing.csv"))


def test_validate_output_dir_creates_dir(workspace):
    validator = PathValidator(workspace)
    out = workspace / "results" / "deeply" / "nested"
    validator.validate_output_dir(out)
    assert out.exists() and out.is_dir()


def test_output_dir_escape_rejected(workspace):
    validator = PathValidator(workspace)
    with pytest.raises(PermissionError):
        validator.validate_output_dir(Path("../../escape"))


def test_read_only_warns_in_dev_mode(workspace, caplog):
    f = workspace / "writable.csv"
    f.touch()
    # Ensure mode includes user-write (default umask)
    os.chmod(f, 0o644)
    validator = PathValidator(workspace)
    # Make sure PROD env is off
    os.environ.pop("PHASE0_PROD", None)
    is_ro = validator.check_read_only(f)
    assert is_ro is False
    assert any("writable" in r.message.lower() for r in caplog.records)


def test_read_only_enforced_in_prod_mode(workspace, monkeypatch):
    f = workspace / "writable.csv"
    f.touch()
    os.chmod(f, 0o644)
    monkeypatch.setenv("PHASE0_PROD", "1")
    validator = PathValidator(workspace)
    with pytest.raises(PermissionError, match="writable"):
        validator.check_read_only(f)


def test_read_only_passes_when_file_is_chmod_444(workspace):
    f = workspace / "readonly.csv"
    f.touch()
    os.chmod(f, 0o444)
    validator = PathValidator(workspace)
    assert validator.check_read_only(f) is True
    # Restore for cleanup
    os.chmod(f, 0o644)
