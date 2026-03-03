#!/usr/bin/env python3
"""
Backup management utility for IDS Healthcare CIP pipeline.

Usage:
    python scripts/backup_utils.py list                      # List all backups
    python scripts/backup_utils.py list --type models        # List specific type
    python scripts/backup_utils.py backup --dir models/phase5 --type models --tag phase5
    python scripts/backup_utils.py restore --archive backups/models/models_phase5_*.tar.gz --dest models/
    python scripts/backup_utils.py verify --archive backups/models/models_phase5_*.tar.gz
    python scripts/backup_utils.py cleanup --keep 3          # Keep 3 latest
    python scripts/backup_utils.py sizes                     # Show backup sizes
    python scripts/backup_utils.py storage                   # Check disk usage
"""

import sys
import argparse
import json
from pathlib import Path
from typing import Optional
import glob

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.backup_manager import BackupManager
from src.utils.storage_validator import StorageValidator


def list_backups(backup_mgr: BackupManager, backup_type: Optional[str] = None, tag: Optional[str] = None):
    """List available backups."""
    backups = backup_mgr.list_backups(backup_type, tag)
    
    if not backups:
        print("No backups found.")
        return
    
    print("\n" + "=" * 80)
    print("BACKUPS")
    print("=" * 80)
    
    for type_name, backup_list in backups.items():
        print(f"\n{type_name.upper()}:")
        if not backup_list:
            print("  (no backups)")
            continue
        
        for idx, backup in enumerate(backup_list):
            version = backup.get('version', 'unknown')
            timestamp = backup.get('timestamp', 'unknown')
            size_mb = backup.get('archive_size_mb', 0)
            backup_tag = backup.get('tag', '')
            checksum = backup.get('checksum', '')[:8]
            
            marker = " [LATEST]" if idx == 0 else ""
            tag_str = f" ({backup_tag})" if backup_tag else ""
            
            print(f"  {idx+1}. v{version}{tag_str} - {size_mb:.1f}MB - {timestamp[:10]} - {checksum}...{marker}")
    
    print("=" * 80 + "\n")


def create_backup(
    backup_mgr: BackupManager,
    source_dir: str,
    backup_type: str,
    tag: Optional[str] = None
):
    """Create a new backup."""
    print(f"\nCreating backup of {source_dir}...")
    
    try:
        metadata = backup_mgr.backup_directory(source_dir, backup_type, tag)
        print(f"✓ Backup created successfully:")
        print(f"  Archive: {metadata['archive_path']}")
        print(f"  Size: {metadata['archive_size_mb']:.1f}MB")
        print(f"  Checksum: {metadata['checksum'][:16]}...")
    except Exception as e:
        print(f"✗ Backup failed: {e}")
        sys.exit(1)


def restore_backup(
    backup_mgr: BackupManager,
    archive_path: str,
    restore_dir: str,
    verify: bool = True
):
    """Restore a backup."""
    # Handle glob patterns
    if '*' in archive_path:
        matches = glob.glob(archive_path)
        if not matches:
            print(f"No archives matching: {archive_path}")
            sys.exit(1)
        archive_path = sorted(matches)[-1]  # Use latest match
        print(f"Selected: {archive_path}")
    
    print(f"\nRestoring backup from {archive_path}...")
    
    try:
        backup_mgr.restore_backup(archive_path, restore_dir, verify_checksum=verify)
        print(f"✓ Backup restored successfully to {restore_dir}")
    except Exception as e:
        print(f"✗ Restore failed: {e}")
        sys.exit(1)


def verify_backup(backup_mgr: BackupManager, archive_path: str):
    """Verify backup integrity."""
    # Handle glob patterns
    if '*' in archive_path:
        matches = glob.glob(archive_path)
        if not matches:
            print(f"No archives matching: {archive_path}")
            sys.exit(1)
        archives = sorted(matches)
    else:
        archives = [archive_path]
    
    print(f"\nVerifying {len(archives)} backup(s)...")
    
    failed = []
    for archive in archives:
        is_valid, message = backup_mgr.verify_backup(archive)
        status = "✓" if is_valid else "✗"
        print(f"{status} {Path(archive).name}: {message}")
        if not is_valid:
            failed.append(archive)
    
    if failed:
        print(f"\n✗ {len(failed)} backup(s) failed verification")
        sys.exit(1)
    else:
        print(f"\n✓ All backups verified successfully")


def cleanup_backups(
    backup_mgr: BackupManager,
    keep: int,
    backup_type: Optional[str] = None,
    dry_run: bool = False
):
    """Clean up old backups."""
    action = "Would delete" if dry_run else "Deleting"
    print(f"\n{action} old backups (keeping {keep} latest)...")
    
    deleted = backup_mgr.cleanup_old_backups(backup_type, dry_run)
    
    if deleted:
        print(f"\n{action.split()[0] + 'ed'} {len(deleted)} backup(s):")
        for name in deleted:
            print(f"  - {name}")
    else:
        print("No backups to clean up.")


def show_sizes(backup_mgr: BackupManager):
    """Show backup sizes by type."""
    sizes = backup_mgr.get_backup_size()
    
    print("\n" + "=" * 80)
    print("BACKUP SIZES")
    print("=" * 80)
    
    if not sizes:
        print("No backups found.")
        return
    
    total = 0
    for backup_type, size_mb in sizes.items():
        print(f"{backup_type:20s}: {size_mb:8.1f} MB")
        total += size_mb
    
    print("-" * 80)
    print(f"{'TOTAL':20s}: {total:8.1f} MB")
    print("=" * 80 + "\n")


def check_storage(project_root: Path):
    """Check disk usage and storage requirements."""
    validator = StorageValidator(project_root)
    
    print("\n" + "=" * 80)
    print("STORAGE STATUS")
    print("=" * 80)
    
    # Current disk usage
    usage = validator.get_disk_usage(project_root)
    print(f"\nDisk Usage:")
    print(f"  Total: {usage['total_mb']:.0f} MB ({usage['total_mb']/1024:.1f} GB)")
    print(f"  Used:  {usage['used_mb']:.0f} MB ({usage['percent_used']:.1f}%)")
    print(f"  Free:  {usage['free_mb']:.0f} MB ({usage['free_mb']/1024:.1f} GB)")
    
    # Pipeline requirements
    print(f"\nPipeline Space Requirements:")
    phases = ["phase1", "phase2", "phase3", "phase4", "phase5"]
    all_ok, details = validator.check_space_for_pipeline(phases, fail_on_insufficient=False)
    
    for phase in phases:
        detail = details[phase]
        required = detail['required_mb']
        status = "✓" if detail['has_sufficient_space'] else "✗"
        margin = detail['margin_mb']
        print(f"  {status} {phase}: {required:.0f} MB required (margin: {margin:+.0f} MB)")
    
    # Cleanup suggestions
    print(f"\nDirectory Sizes:")
    suggestions = validator.suggest_cleanup_targets(project_root)
    
    if suggestions:
        for name, info in sorted(suggestions.items(), key=lambda x: x[1]['size_mb'], reverse=True):
            safe_marker = " [safe to delete]" if info['safe_to_delete'] else ""
            print(f"  {name:30s}: {info['size_mb']:8.1f} MB{safe_marker}")
    
    print("=" * 80 + "\n")
    
    if not all_ok:
        print("⚠️  Warning: Insufficient disk space for some phases")
        return 1
    else:
        print("✓ Sufficient disk space available")
        return 0


def main():
    parser = argparse.ArgumentParser(
        description="Manage IDS Healthcare CIP pipeline backups",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List backups")
    list_parser.add_argument("--type", choices=["checkpoints", "models", "results"], help="Filter by type")
    list_parser.add_argument("--tag", help="Filter by tag")
    
    # Backup command
    backup_parser = subparsers.add_parser("backup", help="Create backup")
    backup_parser.add_argument("--dir", required=True, help="Directory to backup")
    backup_parser.add_argument("--type", required=True, choices=["checkpoints", "models", "results"], help="Backup type")
    backup_parser.add_argument("--tag", help="Tag for the backup")
    
    # Restore command
    restore_parser = subparsers.add_parser("restore", help="Restore backup")
    restore_parser.add_argument("--archive", required=True, help="Archive path (supports glob patterns)")
    restore_parser.add_argument("--dest", required=True, help="Restore destination")
    restore_parser.add_argument("--no-verify", action="store_true", help="Skip checksum verification")
    
    # Verify command
    verify_parser = subparsers.add_parser("verify", help="Verify backup integrity")
    verify_parser.add_argument("--archive", required=True, help="Archive path (supports glob patterns)")
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser("cleanup", help="Clean up old backups")
    cleanup_parser.add_argument("--keep", type=int, default=7, help="Number of backups to keep")
    cleanup_parser.add_argument("--type", choices=["checkpoints", "models", "results"], help="Filter by type")
    cleanup_parser.add_argument("--dry-run", action="store_true", help="Preview without deleting")
    
    # Sizes command
    subparsers.add_parser("sizes", help="Show backup sizes")
    
    # Storage command
    subparsers.add_parser("storage", help="Check disk usage")
    
    # Global options
    parser.add_argument("--backup-dir", default="backups", help="Backup directory")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Initialize managers
    backup_mgr = BackupManager(args.backup_dir)
    
    # Execute command
    if args.command == "list":
        list_backups(backup_mgr, args.type, args.tag)
    
    elif args.command == "backup":
        create_backup(backup_mgr, args.dir, args.type, args.tag)
    
    elif args.command == "restore":
        restore_backup(backup_mgr, args.archive, args.dest, not args.no_verify)
    
    elif args.command == "verify":
        verify_backup(backup_mgr, args.archive)
    
    elif args.command == "cleanup":
        cleanup_backups(backup_mgr, args.keep, args.type, args.dry_run)
    
    elif args.command == "sizes":
        show_sizes(backup_mgr)
    
    elif args.command == "storage":
        sys.exit(check_storage(PROJECT_ROOT))


if __name__ == "__main__":
    main()
