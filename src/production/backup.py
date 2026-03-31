"""Database backup utility — periodic SQLite snapshots.

Creates timestamped copies of the production database.
Old backups beyond --keep count are automatically pruned.

Usage::

    # One-shot backup
    python -m src.production.backup

    # Continuous (every hour, keep 48 backups = 2 days)
    python -m src.production.backup --interval 3600 --keep 48
"""

from __future__ import annotations

import argparse
import logging
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def backup_database(
    db_path: Path | None = None,
    backup_dir: Path | None = None,
    keep: int = 48,
) -> Path | None:
    """Create a timestamped backup of the SQLite database.

    Args:
        db_path: Path to source database.
        backup_dir: Directory for backup files.
        keep: Maximum number of backups to retain.

    Returns:
        Path to created backup, or None if source doesn't exist.
    """
    from config.production_loader import cfg

    if db_path is None:
        db_path = PROJECT_ROOT / cfg("database.path", "data/production/iomt_ids.db")
    if backup_dir is None:
        backup_dir = db_path.parent / "backups"

    if not db_path.exists():
        logger.warning("Database not found: %s — skipping backup", db_path)
        return None

    backup_dir.mkdir(parents=True, exist_ok=True)

    # Create timestamped copy
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup_path = backup_dir / f"iomt_ids_{ts}.db"
    shutil.copy2(str(db_path), str(backup_path))

    size_kb = backup_path.stat().st_size / 1024
    logger.info("Backup created: %s (%.1f KB)", backup_path.name, size_kb)

    # Prune old backups
    backups = sorted(backup_dir.glob("iomt_ids_*.db"), reverse=True)
    for old in backups[keep:]:
        old.unlink()
        logger.info("Pruned old backup: %s", old.name)

    remaining = min(len(backups), keep)
    logger.info("Backups: %d retained (max %d)", remaining, keep)
    return backup_path


def restore_database(
    backup_path: Path,
    db_path: Path | None = None,
) -> bool:
    """Restore database from a backup file.

    Args:
        backup_path: Path to backup file to restore.
        db_path: Target database path (overwrites existing).

    Returns:
        True if restore succeeded.
    """
    from config.production_loader import cfg

    if db_path is None:
        db_path = PROJECT_ROOT / cfg("database.path", "data/production/iomt_ids.db")

    if not backup_path.exists():
        logger.error("Backup not found: %s", backup_path)
        return False

    # Safety: backup current DB before overwriting
    if db_path.exists():
        safety = db_path.with_suffix(".db.pre_restore")
        shutil.copy2(str(db_path), str(safety))
        logger.info("Pre-restore safety copy: %s", safety.name)

    shutil.copy2(str(backup_path), str(db_path))
    logger.info("Database restored from: %s", backup_path.name)
    return True


def list_backups(backup_dir: Path | None = None) -> list[dict]:
    """List available backups with metadata."""
    from config.production_loader import cfg

    if backup_dir is None:
        db_path = PROJECT_ROOT / cfg("database.path", "data/production/iomt_ids.db")
        backup_dir = db_path.parent / "backups"

    if not backup_dir.exists():
        return []

    backups = sorted(backup_dir.glob("iomt_ids_*.db"), reverse=True)
    return [
        {
            "filename": b.name,
            "path": str(b),
            "size_kb": round(b.stat().st_size / 1024, 1),
            "modified": datetime.fromtimestamp(
                b.stat().st_mtime, tz=timezone.utc
            ).isoformat(),
        }
        for b in backups
    ]


def main() -> None:
    """CLI entry point for backup operations."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  [%(name)s]  %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="IoMT IDS database backup")
    parser.add_argument("--interval", type=int, default=0,
                        help="Seconds between backups (0 = one-shot)")
    parser.add_argument("--keep", type=int, default=48,
                        help="Max backups to retain (default: 48)")
    parser.add_argument("--restore", type=str, default=None,
                        help="Restore from backup file path")
    parser.add_argument("--list", action="store_true",
                        help="List available backups")
    args = parser.parse_args()

    if args.list:
        for b in list_backups():
            print(f"  {b['filename']}  {b['size_kb']:>8.1f} KB  {b['modified']}")
        return

    if args.restore:
        ok = restore_database(Path(args.restore))
        print("Restore OK" if ok else "Restore FAILED")
        return

    if args.interval > 0:
        logger.info("Continuous backup: every %ds, keep %d", args.interval, args.keep)
        while True:
            backup_database(keep=args.keep)
            time.sleep(args.interval)
    else:
        backup_database(keep=args.keep)


if __name__ == "__main__":
    main()
