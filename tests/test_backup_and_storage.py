from types import SimpleNamespace

import pytest

from src.utils.backup_manager import BackupManager
from src.utils.storage_validator import StorageValidator


def test_backup_verify_and_cleanup(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "file.txt").write_text("hello")

    backup_dir = tmp_path / "backups"
    manager = BackupManager(backup_dir=str(backup_dir), retention_count=1, retention_days=999, compression=False)

    meta_old = manager.backup_directory(str(source_dir), "checkpoints", tag="old")
    meta_new = manager.backup_directory(str(source_dir), "checkpoints", tag="new")

    archive_new = backup_dir / meta_new["archive_path"]

    ok, msg = manager.verify_backup(str(archive_new))
    assert ok, msg

    # Corrupt the archive to trigger verification failure
    archive_new.write_bytes(b"corrupt")
    ok, msg = manager.verify_backup(str(archive_new))
    assert not ok
    assert "Checksum" in msg or "Failed" in msg

    # Cleanup should keep only the newest (meta_new) because retention_count=1
    deleted = manager.cleanup_old_backups("checkpoints", dry_run=False)
    archive_old = backup_dir / meta_old["archive_path"]
    assert archive_old.name in deleted
    assert not archive_old.exists()
    assert not archive_old.with_suffix('.json').exists()


def test_storage_validator_insufficient_space(monkeypatch, tmp_path):
    # Force disk usage to a low-free scenario
    def fake_disk_usage(path):
        return SimpleNamespace(total=200 * 1024 * 1024, used=150 * 1024 * 1024, free=50 * 1024 * 1024)

    monkeypatch.setattr("shutil.disk_usage", fake_disk_usage)

    validator = StorageValidator(project_root=tmp_path)

    with pytest.raises(RuntimeError):
        validator.check_space_for_phase("phase1", fail_on_insufficient=True)

    has_space, details = validator.check_space_for_phase("phase1", fail_on_insufficient=False)
    assert not has_space
    assert details["has_sufficient_space"] is False
    assert details["available_mb"] == 50


def test_storage_validator_cleanup_needed(monkeypatch, tmp_path):
    def fake_disk_usage(path):
        return SimpleNamespace(total=100 * 1024 * 1024, used=90 * 1024 * 1024, free=10 * 1024 * 1024)

    monkeypatch.setattr("shutil.disk_usage", fake_disk_usage)

    validator = StorageValidator(project_root=tmp_path)
    cleanup_needed, mb_to_free = validator.check_cleanup_needed(target_free_mb=50)

    assert cleanup_needed is True
    assert mb_to_free == 40
