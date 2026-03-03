"""Prefect orchestration for IDS Healthcare CIP pipeline.

Features:
 - Per-phase retries and timeouts
 - Resume from checkpoints (skip completed phases)
 - Centralized configuration via config/orchestration.yaml
 - Integration with CheckpointManager for artifact tracking

Usage:
  python scripts/prefect_pipeline.py
  python scripts/prefect_pipeline.py --resume  # Resume from last checkpoint
  python scripts/prefect_pipeline.py --fresh   # Force restart all phases
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Dict, Any
import yaml

from prefect import flow, task, get_run_logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.checkpoint_manager import CheckpointManager
from src.utils.backup_manager import BackupManager
from src.utils.storage_validator import StorageValidator
from src.utils.config_safety import validate_orchestration_config

CONFIG_PATH = PROJECT_ROOT / "config" / "orchestration.yaml"


def load_config() -> Dict[str, Any]:
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)
    orchestration_cfg = cfg.get("orchestration", {}) if cfg else {}
    return validate_orchestration_config(orchestration_cfg)


def create_managers(
    config: Dict[str, Any],
    project_root: Path = PROJECT_ROOT
) -> tuple[CheckpointManager, BackupManager, StorageValidator]:
    """
    Factory function to create manager instances from config.
    
    Args:
        config: Validated orchestration config
        project_root: Project root directory
        
    Returns:
        Tuple of (checkpoint_mgr, backup_mgr, storage_validator)
    """
    checkpoints_dir = project_root / config.get("checkpoints_dir", "results/checkpoints")
    
    backup_cfg = config.get("backup", {})
    backup_dir = backup_cfg.get("backup_dir", "backups")
    retention_count = backup_cfg.get("retention_count", 7)
    retention_days = backup_cfg.get("retention_days", 30)
    
    checkpoint_mgr = CheckpointManager(str(checkpoints_dir))
    backup_mgr = BackupManager(
        backup_dir=backup_dir,
        retention_count=retention_count,
        retention_days=retention_days
    )
    storage_validator = StorageValidator(project_root)
    
    return checkpoint_mgr, backup_mgr, storage_validator


@task
def run_phase(
    phase_name: str,
    command: str,
    checkpoint: Path,
    timeout_seconds: int,
    resume: bool,
    checkpoint_mgr: CheckpointManager,
    backup_mgr: BackupManager,
    storage_validator: StorageValidator,
    backup_enabled: bool
) -> None:
    """
    Execute a pipeline phase with checkpoint tracking, storage validation, and backups.
    
    Args:
        phase_name: Phase identifier (e.g., "phase1", "phase2")
        command: Shell command to execute
        checkpoint: Path to checkpoint file marker
        timeout_seconds: Execution timeout
        resume: If True, skip phases with existing checkpoints
        checkpoint_mgr: CheckpointManager for artifact tracking
        backup_mgr: BackupManager for creating backups
        storage_validator: StorageValidator for space checks
        backup_enabled: If True, create backup after phase completion
    """
    logger = get_run_logger()

    # Check for existing checkpoint (legacy marker file)
    if resume and checkpoint.exists():
        logger.info(f"Legacy checkpoint found, skipping: {checkpoint}")
        return
    
    # Check for existing artifacts in CheckpointManager
    if resume and checkpoint_mgr.has_checkpoint(phase_name, artifact_name="data"):
        latest = checkpoint_mgr.get_latest_checkpoint(phase_name)
        if latest:
            logger.info(f"CheckpointManager artifact found for {phase_name}, skipping:")
            logger.info(f"  Version: {latest.get('version', 'unknown')}")
            logger.info(f"  Timestamp: {latest.get('timestamp', 'unknown')}")
            checkpoint.touch()  # Create legacy marker for consistency
            return

    # Storage validation before execution
    logger.info(f"Checking storage space for {phase_name}...")
    try:
        has_space, details = storage_validator.check_space_for_phase(
            phase_name,
            fail_on_insufficient=True
        )
        logger.info(f"Storage check passed: {details['available_mb']:.0f}MB available, "
                   f"{details['required_mb']:.0f}MB required")
    except RuntimeError as e:
        logger.error(f"Storage check failed for {phase_name}: {e}")
        raise

    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Running {phase_name}: {command} (timeout={timeout_seconds}s)")

    try:
        subprocess.run(command, shell=True, check=True, timeout=timeout_seconds, cwd=PROJECT_ROOT)
    except subprocess.TimeoutExpired:
        logger.error(f"Timeout after {timeout_seconds}s for {phase_name}: {command}")
        raise
    except subprocess.CalledProcessError as exc:
        logger.error(f"Command failed (exit {exc.returncode}) for {phase_name}: {command}")
        raise

    checkpoint.touch()
    logger.info(f"Checkpoint written: {checkpoint}")
    
    # Verify checkpoint was created by the phase
    if checkpoint_mgr.has_checkpoint(phase_name):
        latest = checkpoint_mgr.get_latest_checkpoint(phase_name)
        logger.info(f"Verified artifact checkpoint for {phase_name}:")
        logger.info(f"  Version: {latest.get('version', 'unknown')}")
    else:
        logger.warning(f"No artifact checkpoint found for {phase_name} - phase may not be using CheckpointManager")
    
    # Create backup if enabled
    if backup_enabled:
        logger.info(f"Creating backup for {phase_name}...")
        try:
            # Backup checkpoints
            checkpoint_dir = PROJECT_ROOT / "results" / "checkpoints" / phase_name
            if checkpoint_dir.exists():
                backup_mgr.backup_directory(
                    str(checkpoint_dir),
                    backup_type="checkpoints",
                    tag=phase_name
                )
            
            # Backup models if this is phase3 or phase5
            if phase_name in ["phase3", "phase5"]:
                model_dir = PROJECT_ROOT / "models" / phase_name
                if model_dir.exists():
                    backup_mgr.backup_directory(
                        str(model_dir),
                        backup_type="models",
                        tag=phase_name
                    )
            
            # Backup results
            results_dir = PROJECT_ROOT / "results" / phase_name
            if results_dir.exists():
                backup_mgr.backup_directory(
                    str(results_dir),
                    backup_type="results",
                    tag=phase_name
                )
            
            logger.info(f"Backup completed for {phase_name}")
        except Exception as e:
            logger.warning(f"Backup failed for {phase_name}: {e}")


@flow(name="ids-cip-pipeline")
def orchestrate(
    resume: bool = True,
    checkpoint_mgr: CheckpointManager | None = None,
    backup_mgr: BackupManager | None = None,
    storage_validator: StorageValidator | None = None,
    config: Dict[str, Any] | None = None
) -> None:
    """
    Execute the complete IDS Healthcare CIP pipeline.
    
    Args:
        resume: If True, skip phases with existing checkpoints
        checkpoint_mgr: Optional CheckpointManager instance (for testing/injection)
        backup_mgr: Optional BackupManager instance (for testing/injection)
        storage_validator: Optional StorageValidator instance (for testing/injection)
        config: Optional config dict (defaults to loading from orchestration.yaml)
    """
    cfg = config if config is not None else load_config()
    resume = resume if resume is not None else cfg.get("resume", True)
    checkpoints_dir = PROJECT_ROOT / cfg.get("checkpoints_dir", "results/checkpoints")
    phases = cfg.get("phases", [])
    
    # Backup configuration
    backup_cfg = cfg.get("backup", {})
    backup_enabled = backup_cfg.get("enabled", True)
    
    # Storage validation configuration
    storage_cfg = cfg.get("storage_validation", {})
    storage_check_enabled = storage_cfg.get("enabled", True)
    
    # Initialize managers via factory or use injected instances
    if checkpoint_mgr is None or backup_mgr is None or storage_validator is None:
        _checkpoint_mgr, _backup_mgr, _storage_validator = create_managers(cfg, PROJECT_ROOT)
        checkpoint_mgr = checkpoint_mgr or _checkpoint_mgr
        backup_mgr = backup_mgr or _backup_mgr
        storage_validator = storage_validator or _storage_validator
    
    logger = get_run_logger()
    logger.info("=" * 80)
    logger.info("IDS Healthcare CIP Pipeline - Orchestration Started")
    logger.info("=" * 80)
    logger.info(f"Resume mode: {resume}")
    logger.info(f"Checkpoints directory: {checkpoints_dir}")
    logger.info(f"Backup enabled: {backup_enabled}")
    logger.info(f"Storage validation: {storage_check_enabled}")
    logger.info(f"Phases configured: {len(phases)}")
    
    # Pre-flight storage check for entire pipeline
    if storage_check_enabled:
        logger.info("\nPre-flight storage validation...")
        try:
            phase_names = [p["name"] for p in phases]
            all_ok, details = storage_validator.check_space_for_pipeline(
                phases=phase_names,
                fail_on_insufficient=False
            )
            
            if not all_ok:
                logger.warning("Some phases may have insufficient disk space:")
                for phase, detail in details.items():
                    if not detail["has_sufficient_space"]:
                        logger.warning(f"  {phase}: short by {abs(detail['margin_mb']):.0f}MB")
            else:
                logger.info("Storage validation passed for all phases")
        except Exception as e:
            logger.error(f"Storage validation failed: {e}")
            if storage_cfg.get("fail_on_error", False):
                raise
    
    # List existing checkpoints
    if resume:
        existing = checkpoint_mgr.list_checkpoints()
        if existing:
            logger.info("\nExisting checkpoints found:")
            for phase_name, checkpoints in existing.items():
                if checkpoints:
                    latest = checkpoints[0]
                    logger.info(f"  {phase_name}: {latest.get('version', 'unknown')} ({latest.get('timestamp', 'unknown')})")
        else:
            logger.info("\nNo existing checkpoints found - starting from scratch")
    else:
        logger.info("\nForced restart - ignoring existing checkpoints")
    
    # List existing backups
    if backup_enabled:
        backup_sizes = backup_mgr.get_backup_size()
        if backup_sizes:
            logger.info("\nExisting backups:")
            for backup_type, size_mb in backup_sizes.items():
                logger.info(f"  {backup_type}: {size_mb:.0f}MB")
        
        # Cleanup old backups before starting
        logger.info(f"\nCleaning up old backups (keeping {retention_count} latest, {retention_days} days)...")
        deleted = backup_mgr.cleanup_old_backups(dry_run=False)
        if deleted:
            logger.info(f"Deleted {len(deleted)} old backups")

    for phase in phases:
        name = phase["name"]
        command = phase["command"]
        timeout_seconds = int(phase.get("timeout_seconds", 3600))
        retries = int(phase.get("retries", 0))
        retry_delay_seconds = int(phase.get("retry_delay_seconds", 60))

        checkpoint = checkpoints_dir / f"{name}.done"

        # Prefect task with dynamic retry settings
        run_phase.with_options(
            name=f"run-{name}",
            retries=retries,
            retry_delay_seconds=retry_delay_seconds,
        )(
            phase_name=name,
            command=command,
            checkpoint=checkpoint,
            timeout_seconds=timeout_seconds,
            resume=resume,
            checkpoint_mgr=checkpoint_mgr,
            backup_mgr=backup_mgr,
            storage_validator=storage_validator,
            backup_enabled=backup_enabled
        )
    
    logger.info("=" * 80)
    logger.info("Pipeline Execution Complete")
    logger.info("=" * 80)
    
    # Final backup summary
    if backup_enabled:
        final_sizes = backup_mgr.get_backup_size()
        logger.info("\nFinal backup sizes:")
        for backup_type, size_mb in final_sizes.items():
            logger.info(f"  {backup_type}: {size_mb:.0f}MB")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="IDS Healthcare CIP Pipeline Orchestration")
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Resume from last checkpoint (default: True)"
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Force restart from beginning, ignore checkpoints"
    )
    
    args = parser.parse_args()
    
    resume = not args.fresh if args.fresh else args.resume
    
    orchestrate(resume=resume)
