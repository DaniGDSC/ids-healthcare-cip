"""Storage space validation utilities for pipeline execution.

Provides pre-flight checks to ensure sufficient disk space before running
data-intensive pipeline phases.
"""

import shutil
import logging
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class StorageValidator:
    """Validate available storage space before pipeline execution."""
    
    # Estimated space requirements per phase (in MB)
    PHASE_ESTIMATES = {
        "phase1": {
            "input": 5000,      # Raw data ~5GB
            "working": 10000,   # Processing overhead ~10GB
            "output": 3000      # Processed data ~3GB
        },
        "phase2": {
            "input": 3000,
            "working": 2000,
            "output": 500
        },
        "phase3": {
            "input": 500,
            "working": 2000,
            "output": 1000
        },
        "phase4": {
            "input": 1000,
            "working": 1000,
            "output": 500
        },
        "phase5": {
            "input": 1500,
            "working": 1000,
            "output": 500
        }
    }
    
    # Safety margin multiplier
    SAFETY_MARGIN = 1.2  # 20% extra
    
    def __init__(self, project_root: Optional[Path] = None):
        """
        Initialize StorageValidator.
        
        Args:
            project_root: Project root directory (defaults to CWD)
        """
        self.project_root = project_root or Path.cwd()
    
    def get_disk_usage(self, path: Path) -> Dict[str, float]:
        """
        Get disk usage statistics for a path.
        
        Args:
            path: Directory or file path
            
        Returns:
            Dict with total, used, free space in MB
        """
        try:
            usage = shutil.disk_usage(path)
            return {
                "total_mb": round(usage.total / (1024 * 1024), 2),
                "used_mb": round(usage.used / (1024 * 1024), 2),
                "free_mb": round(usage.free / (1024 * 1024), 2),
                "percent_used": round((usage.used / usage.total) * 100, 1)
            }
        except Exception as e:
            logger.error(f"Failed to get disk usage for {path}: {e}")
            return {
                "total_mb": 0,
                "used_mb": 0,
                "free_mb": 0,
                "percent_used": 0
            }
    
    def estimate_phase_requirements(self, phase: str) -> float:
        """
        Estimate total space required for a phase (in MB).
        
        Args:
            phase: Phase name (e.g., "phase1")
            
        Returns:
            Estimated space in MB with safety margin
        """
        if phase not in self.PHASE_ESTIMATES:
            logger.warning(f"No estimate for {phase}, using default 5GB")
            return 5000 * self.SAFETY_MARGIN
        
        requirements = self.PHASE_ESTIMATES[phase]
        total = requirements["input"] + requirements["working"] + requirements["output"]
        return total * self.SAFETY_MARGIN
    
    def check_space_for_phase(
        self,
        phase: str,
        data_dir: Optional[Path] = None,
        fail_on_insufficient: bool = True
    ) -> Tuple[bool, Dict[str, any]]:
        """
        Check if sufficient space is available for a phase.
        
        Args:
            phase: Phase name
            data_dir: Directory to check (defaults to project root)
            fail_on_insufficient: Raise exception if insufficient space
            
        Returns:
            Tuple of (has_sufficient_space, details_dict)
            
        Raises:
            RuntimeError: If insufficient space and fail_on_insufficient=True
        """
        check_dir = data_dir or self.project_root
        
        disk_usage = self.get_disk_usage(check_dir)
        required_mb = self.estimate_phase_requirements(phase)
        available_mb = disk_usage["free_mb"]
        
        has_space = available_mb >= required_mb
        
        details = {
            "phase": phase,
            "required_mb": round(required_mb, 2),
            "available_mb": available_mb,
            "total_mb": disk_usage["total_mb"],
            "percent_used": disk_usage["percent_used"],
            "has_sufficient_space": has_space,
            "margin_mb": round(available_mb - required_mb, 2) if has_space else round(required_mb - available_mb, 2)
        }
        
        if has_space:
            logger.info(json.dumps({
                "event": "storage_check",
                "phase": phase,
                "status": "ok",
                "required_mb": details["required_mb"],
                "available_mb": details["available_mb"],
                "margin_mb": details["margin_mb"]
            }))
        else:
            message = (
                f"Insufficient disk space for {phase}: "
                f"required {required_mb:.0f}MB, available {available_mb:.0f}MB "
                f"(short by {details['margin_mb']:.0f}MB)"
            )
            logger.error(json.dumps({
                "event": "storage_check",
                "phase": phase,
                "status": "insufficient",
                "required_mb": details["required_mb"],
                "available_mb": details["available_mb"],
                "shortage_mb": abs(details["margin_mb"])
            }))
            
            if fail_on_insufficient:
                raise RuntimeError(message)
        
        return has_space, details
    
    def check_space_for_pipeline(
        self,
        phases: Optional[list] = None,
        fail_on_insufficient: bool = True
    ) -> Tuple[bool, Dict[str, Dict]]:
        """
        Check space requirements for entire pipeline.
        
        Args:
            phases: List of phase names (defaults to all phases)
            fail_on_insufficient: Raise exception if any phase has insufficient space
            
        Returns:
            Tuple of (all_phases_ok, phase_details_dict)
        """
        if phases is None:
            phases = ["phase1", "phase2", "phase3", "phase4", "phase5"]
        
        results = {}
        all_ok = True
        
        for phase in phases:
            has_space, details = self.check_space_for_phase(
                phase,
                fail_on_insufficient=False
            )
            results[phase] = details
            if not has_space:
                all_ok = False
        
        if not all_ok and fail_on_insufficient:
            failed_phases = [p for p, d in results.items() if not d["has_sufficient_space"]]
            raise RuntimeError(
                f"Insufficient disk space for phases: {', '.join(failed_phases)}. "
                f"See logs for details."
            )
        
        return all_ok, results
    
    def get_directory_size(self, directory: Path) -> float:
        """
        Calculate total size of a directory in MB.
        
        Args:
            directory: Directory path
            
        Returns:
            Size in MB
        """
        if not directory.exists():
            return 0.0
        
        total_bytes = sum(
            f.stat().st_size
            for f in directory.rglob('*')
            if f.is_file()
        )
        return round(total_bytes / (1024 * 1024), 2)
    
    def check_cleanup_needed(
        self,
        target_free_mb: float = 10000,
        data_dir: Optional[Path] = None
    ) -> Tuple[bool, float]:
        """
        Check if cleanup is needed to reach target free space.
        
        Args:
            target_free_mb: Target free space in MB
            data_dir: Directory to check
            
        Returns:
            Tuple of (cleanup_needed, mb_to_free)
        """
        check_dir = data_dir or self.project_root
        disk_usage = self.get_disk_usage(check_dir)
        
        current_free = disk_usage["free_mb"]
        
        if current_free >= target_free_mb:
            return False, 0.0
        
        needed = target_free_mb - current_free
        return True, round(needed, 2)
    
    def suggest_cleanup_targets(
        self,
        project_root: Optional[Path] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Suggest directories that could be cleaned up to free space.
        
        Args:
            project_root: Project root directory
            
        Returns:
            Dict mapping directory -> size info
        """
        root = project_root or self.project_root
        
        cleanup_targets = {
            "logs": root / "logs",
            "results/phase1": root / "results" / "phase1",
            "results/phase2": root / "results" / "phase2",
            "results/phase3": root / "results" / "phase3",
            "results/phase4": root / "results" / "phase4",
            "results/phase5": root / "results" / "phase5",
            "data/processed": root / "data" / "processed",
            "data/latent": root / "data" / "latent",
            "data/features": root / "data" / "features",
            "__pycache__": root / "src"  # Will scan for all __pycache__ dirs
        }
        
        suggestions = {}
        
        for name, path in cleanup_targets.items():
            if path.exists():
                if name == "__pycache__":
                    # Scan for all __pycache__ directories
                    total_size = sum(
                        self.get_directory_size(p)
                        for p in root.rglob("__pycache__")
                    )
                else:
                    total_size = self.get_directory_size(path)
                
                if total_size > 0:
                    suggestions[name] = {
                        "path": str(path),
                        "size_mb": total_size,
                        "safe_to_delete": name in ["logs", "__pycache__"]
                    }
        
        return suggestions


def check_storage_before_run(
    phase: str,
    project_root: Optional[Path] = None,
    fail_on_insufficient: bool = True
) -> Dict[str, any]:
    """
    Convenience function to check storage before running a phase.
    
    Args:
        phase: Phase name
        project_root: Project root directory
        fail_on_insufficient: Raise exception if insufficient space
        
    Returns:
        Storage check details dict
    """
    validator = StorageValidator(project_root)
    has_space, details = validator.check_space_for_phase(phase, fail_on_insufficient=fail_on_insufficient)
    return details
