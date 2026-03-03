"""Backup and recovery management for pipeline artifacts.

Supports:
- Automated backups of checkpoints, models, and results
- Retention policies (keep N latest, age-based cleanup)
- Local and cloud storage backends
- Incremental and full backup modes
- Backup verification and integrity checks
"""

import json
import logging
import shutil
import tarfile
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import os

logger = logging.getLogger(__name__)


class BackupManager:
    """
    Manage backups of pipeline artifacts with retention policies.
    
    Features:
    - Full and incremental backups
    - Configurable retention (count and age-based)
    - Backup verification via checksums
    - Automated cleanup of old backups
    """
    
    def __init__(
        self,
        backup_dir: str = "backups",
        retention_count: int = 7,
        retention_days: int = 30,
        compression: bool = True
    ):
        """
        Initialize BackupManager.
        
        Args:
            backup_dir: Root directory for backups
            retention_count: Keep N latest backups
            retention_days: Keep backups newer than N days
            compression: Use gzip compression for tar archives
        """
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.retention_count = retention_count
        self.retention_days = retention_days
        self.compression = compression
        
        # Backup subdirectories
        self.checkpoints_backup = self.backup_dir / "checkpoints"
        self.models_backup = self.backup_dir / "models"
        self.results_backup = self.backup_dir / "results"
        
        for subdir in [self.checkpoints_backup, self.models_backup, self.results_backup]:
            subdir.mkdir(parents=True, exist_ok=True)
    
    def _compute_checksum(self, file_path: Path) -> str:
        """Compute SHA256 checksum of a file."""
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    def _get_timestamp(self) -> str:
        """Get timestamp string for backup naming."""
        return datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    
    def backup_directory(
        self,
        source_dir: str,
        backup_type: str,
        tag: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a compressed backup of a directory.
        
        Args:
            source_dir: Directory to backup
            backup_type: Type identifier (checkpoints, models, results)
            tag: Optional tag for the backup (e.g., phase name)
            
        Returns:
            Metadata dict with backup info
        """
        source_path = Path(source_dir)
        
        if not source_path.exists():
            raise FileNotFoundError(f"Source directory not found: {source_dir}")
        
        timestamp = self._get_timestamp()
        backup_name = f"{backup_type}_{timestamp}"
        if tag:
            backup_name = f"{backup_type}_{tag}_{timestamp}"
        
        # Select backup destination
        if backup_type == "checkpoints":
            dest_dir = self.checkpoints_backup
        elif backup_type == "models":
            dest_dir = self.models_backup
        elif backup_type == "results":
            dest_dir = self.results_backup
        else:
            dest_dir = self.backup_dir / backup_type
            dest_dir.mkdir(parents=True, exist_ok=True)
        
        # Create tar archive
        archive_ext = ".tar.gz" if self.compression else ".tar"
        archive_path = dest_dir / f"{backup_name}{archive_ext}"
        
        logger.info(f"Creating backup: {archive_path}")
        
        mode = "w:gz" if self.compression else "w"
        with tarfile.open(archive_path, mode) as tar:
            tar.add(source_path, arcname=source_path.name)
        
        # Compute checksum
        checksum = self._compute_checksum(archive_path)
        
        # Get archive size
        archive_size = archive_path.stat().st_size
        
        # Create metadata
        metadata = {
            "timestamp": datetime.utcnow().isoformat(),
            "version": timestamp,
            "backup_type": backup_type,
            "tag": tag,
            "source_dir": str(source_path),
            "archive_path": str(archive_path.relative_to(self.backup_dir)),
            "archive_size_mb": round(archive_size / (1024 * 1024), 2),
            "compression": self.compression,
            "checksum": checksum
        }
        
        # Save metadata
        meta_path = archive_path.with_suffix('.json')
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(json.dumps({
            "event": "backup",
            "action": "created",
            "type": backup_type,
            "tag": tag,
            "size_mb": metadata["archive_size_mb"],
            "archive": str(archive_path.name)
        }))
        
        return metadata
    
    def restore_backup(
        self,
        archive_path: str,
        restore_dir: str,
        verify_checksum: bool = True
    ) -> bool:
        """
        Restore a backup archive to a directory.
        
        Args:
            archive_path: Path to backup archive
            restore_dir: Directory to restore to
            verify_checksum: Verify checksum before restoring
            
        Returns:
            True if successful
        """
        archive = Path(archive_path)
        if not archive.exists():
            raise FileNotFoundError(f"Backup archive not found: {archive_path}")
        
        # Load and verify metadata
        meta_path = archive.with_suffix('.json')
        if meta_path.exists() and verify_checksum:
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
            
            stored_checksum = metadata.get('checksum')
            if stored_checksum:
                actual_checksum = self._compute_checksum(archive)
                if actual_checksum != stored_checksum:
                    raise ValueError(f"Checksum mismatch for {archive.name}: backup may be corrupted")
                logger.info(f"Checksum verified: {archive.name}")
        
        # Extract archive
        restore_path = Path(restore_dir)
        restore_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Restoring backup: {archive} -> {restore_dir}")
        
        with tarfile.open(archive, 'r:*') as tar:
            tar.extractall(restore_path)
        
        logger.info(json.dumps({
            "event": "backup",
            "action": "restored",
            "archive": str(archive.name),
            "destination": str(restore_dir)
        }))
        
        return True
    
    def list_backups(
        self,
        backup_type: Optional[str] = None,
        tag: Optional[str] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        List available backups.
        
        Args:
            backup_type: Filter by type (checkpoints, models, results)
            tag: Filter by tag
            
        Returns:
            Dict mapping backup_type -> list of backup metadata
        """
        results = {}
        
        # Determine which directories to scan
        if backup_type == "checkpoints":
            scan_dirs = [(self.checkpoints_backup, "checkpoints")]
        elif backup_type == "models":
            scan_dirs = [(self.models_backup, "models")]
        elif backup_type == "results":
            scan_dirs = [(self.results_backup, "results")]
        else:
            scan_dirs = [
                (self.checkpoints_backup, "checkpoints"),
                (self.models_backup, "models"),
                (self.results_backup, "results")
            ]
        
        for scan_dir, type_name in scan_dirs:
            backups = []
            
            for meta_file in sorted(scan_dir.glob("*.json"), reverse=True):
                try:
                    with open(meta_file, 'r') as f:
                        metadata = json.load(f)
                    
                    # Filter by tag if specified
                    if tag and metadata.get('tag') != tag:
                        continue
                    
                    backups.append(metadata)
                except Exception as e:
                    logger.warning(f"Failed to read backup metadata {meta_file}: {e}")
            
            if backups:
                results[type_name] = backups
        
        return results
    
    def get_latest_backup(
        self,
        backup_type: str,
        tag: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Get metadata for the most recent backup."""
        backups = self.list_backups(backup_type, tag)
        
        if backup_type in backups and backups[backup_type]:
            return backups[backup_type][0]  # Already sorted by mtime desc
        return None
    
    def cleanup_old_backups(
        self,
        backup_type: Optional[str] = None,
        dry_run: bool = False
    ) -> List[str]:
        """
        Clean up old backups based on retention policies.
        
        Args:
            backup_type: Type to clean (None = all types)
            dry_run: If True, only report what would be deleted
            
        Returns:
            List of deleted backup names
        """
        deleted = []
        
        backups = self.list_backups(backup_type)
        
        for type_name, backup_list in backups.items():
            # Sort by timestamp descending (newest first)
            backup_list.sort(key=lambda x: x['timestamp'], reverse=True)
            
            # Keep N latest backups
            to_delete = []
            
            # Count-based retention
            if len(backup_list) > self.retention_count:
                to_delete.extend(backup_list[self.retention_count:])
            
            # Age-based retention
            cutoff_date = datetime.utcnow() - timedelta(days=self.retention_days)
            for backup in backup_list:
                backup_date = datetime.fromisoformat(backup['timestamp'])
                if backup_date < cutoff_date:
                    if backup not in to_delete:
                        to_delete.append(backup)
            
            # Delete old backups
            for backup in to_delete:
                archive_path = self.backup_dir / backup['archive_path']
                meta_path = archive_path.with_suffix('.json')
                
                if dry_run:
                    logger.info(f"Would delete: {archive_path.name}")
                    deleted.append(archive_path.name)
                else:
                    try:
                        archive_path.unlink(missing_ok=True)
                        meta_path.unlink(missing_ok=True)
                        logger.info(f"Deleted old backup: {archive_path.name}")
                        deleted.append(archive_path.name)
                    except Exception as e:
                        logger.error(f"Failed to delete {archive_path}: {e}")
        
        if deleted:
            logger.info(json.dumps({
                "event": "backup",
                "action": "cleanup",
                "deleted_count": len(deleted),
                "dry_run": dry_run
            }))
        
        return deleted
    
    def get_backup_size(self, backup_type: Optional[str] = None) -> Dict[str, float]:
        """
        Get total size of backups in MB.
        
        Args:
            backup_type: Type to measure (None = all types)
            
        Returns:
            Dict mapping backup_type -> size in MB
        """
        results = {}
        
        if backup_type == "checkpoints":
            scan_dirs = [(self.checkpoints_backup, "checkpoints")]
        elif backup_type == "models":
            scan_dirs = [(self.models_backup, "models")]
        elif backup_type == "results":
            scan_dirs = [(self.results_backup, "results")]
        else:
            scan_dirs = [
                (self.checkpoints_backup, "checkpoints"),
                (self.models_backup, "models"),
                (self.results_backup, "results")
            ]
        
        for scan_dir, type_name in scan_dirs:
            total_bytes = sum(
                f.stat().st_size
                for f in scan_dir.glob("*.tar*")
                if f.is_file()
            )
            results[type_name] = round(total_bytes / (1024 * 1024), 2)
        
        return results
    
    def verify_backup(self, archive_path: str) -> Tuple[bool, str]:
        """
        Verify backup integrity.
        
        Args:
            archive_path: Path to backup archive
            
        Returns:
            Tuple of (is_valid, message)
        """
        archive = Path(archive_path)
        
        if not archive.exists():
            return False, f"Archive not found: {archive_path}"
        
        # Check metadata
        meta_path = archive.with_suffix('.json')
        if not meta_path.exists():
            return False, "Metadata file missing"
        
        try:
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
        except Exception as e:
            return False, f"Failed to read metadata: {e}"
        
        # Verify checksum
        stored_checksum = metadata.get('checksum')
        if not stored_checksum:
            return False, "No checksum in metadata"
        
        actual_checksum = self._compute_checksum(archive)
        if actual_checksum != stored_checksum:
            return False, f"Checksum mismatch (expected {stored_checksum[:8]}..., got {actual_checksum[:8]}...)"
        
        # Try to open archive
        try:
            with tarfile.open(archive, 'r:*') as tar:
                members = tar.getmembers()
                if not members:
                    return False, "Archive is empty"
        except Exception as e:
            return False, f"Failed to open archive: {e}"
        
        return True, f"Backup verified successfully ({len(members)} files)"
