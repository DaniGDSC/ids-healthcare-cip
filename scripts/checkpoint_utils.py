#!/usr/bin/env python3
"""
Checkpoint management utility for IDS Healthcare CIP pipeline.

Usage:
    python scripts/checkpoint_utils.py list                    # List all checkpoints
    python scripts/checkpoint_utils.py list --phase phase1     # List phase1 checkpoints
    python scripts/checkpoint_utils.py clean --keep 2          # Keep only 2 latest per phase
    python scripts/checkpoint_utils.py clear --phase phase3    # Clear all phase3 checkpoints
    python scripts/checkpoint_utils.py info --phase phase1     # Show latest phase1 info
"""

import sys
import argparse
import json
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.checkpoint_manager import CheckpointManager


def list_checkpoints(checkpoint_mgr: CheckpointManager, phase: Optional[str] = None):
    """List all or phase-specific checkpoints."""
    checkpoints = checkpoint_mgr.list_checkpoints(phase)
    
    if not checkpoints:
        print("No checkpoints found.")
        return
    
    print("\n" + "=" * 80)
    print("CHECKPOINTS")
    print("=" * 80)
    
    for phase_name, checkpoint_list in checkpoints.items():
        print(f"\n{phase_name}:")
        if not checkpoint_list:
            print("  (no checkpoints)")
            continue
        
        for idx, ckpt in enumerate(checkpoint_list):
            version = ckpt.get('version', 'unknown')
            timestamp = ckpt.get('timestamp', 'unknown')
            artifact_type = ckpt.get('artifact_type', ckpt.get('model_name', 'unknown'))
            
            shape_info = ""
            if 'dataframe_shape' in ckpt:
                shape_info = f" shape={ckpt['dataframe_shape']}"
            elif 'model_type' in ckpt:
                shape_info = f" type={ckpt['model_type']}"
            
            metrics_info = ""
            if 'metrics' in ckpt:
                metrics = ckpt['metrics']
                if 'test_accuracy' in metrics:
                    metrics_info = f" acc={metrics['test_accuracy']:.4f}"
                if 'test_f1_weighted' in metrics:
                    metrics_info += f" f1={metrics['test_f1_weighted']:.4f}"
            
            marker = "  [LATEST]" if idx == 0 else ""
            print(f"  {idx+1}. {artifact_type} v{version} ({timestamp}){shape_info}{metrics_info}{marker}")
    
    print("=" * 80 + "\n")


def show_checkpoint_info(checkpoint_mgr: CheckpointManager, phase: str):
    """Show detailed info for latest checkpoint in a phase."""
    latest = checkpoint_mgr.get_latest_checkpoint(phase)
    
    if not latest:
        print(f"No checkpoints found for {phase}")
        return
    
    print("\n" + "=" * 80)
    print(f"LATEST CHECKPOINT: {phase}")
    print("=" * 80)
    print(json.dumps(latest, indent=2, default=str))
    print("=" * 80 + "\n")


def clean_checkpoints(checkpoint_mgr: CheckpointManager, keep: int, phase: Optional[str] = None):
    """Clean old checkpoints, keeping N latest."""
    print(f"\nCleaning checkpoints (keeping {keep} latest per phase)...")
    checkpoint_mgr.clear_checkpoints(phase=phase, keep_latest=keep)
    print("Done.\n")


def clear_checkpoints(checkpoint_mgr: CheckpointManager, phase: Optional[str] = None):
    """Clear all checkpoints for a phase or all phases."""
    if phase:
        confirm = input(f"Clear ALL checkpoints for {phase}? (yes/no): ")
    else:
        confirm = input("Clear ALL checkpoints for ALL phases? (yes/no): ")
    
    if confirm.lower() != 'yes':
        print("Cancelled.")
        return
    
    print("\nClearing checkpoints...")
    checkpoint_mgr.clear_checkpoints(phase=phase, keep_latest=0)
    print("Done.\n")


def main():
    parser = argparse.ArgumentParser(
        description="Manage IDS Healthcare CIP pipeline checkpoints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "action",
        choices=["list", "info", "clean", "clear"],
        help="Action to perform"
    )
    parser.add_argument(
        "--phase",
        type=str,
        help="Specific phase (e.g., phase1, phase2)"
    )
    parser.add_argument(
        "--keep",
        type=int,
        default=2,
        help="Number of latest checkpoints to keep (for clean action)"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="results/checkpoints",
        help="Checkpoint directory"
    )
    
    args = parser.parse_args()
    
    checkpoint_mgr = CheckpointManager(args.checkpoint_dir)
    
    if args.action == "list":
        list_checkpoints(checkpoint_mgr, args.phase)
    elif args.action == "info":
        if not args.phase:
            print("Error: --phase required for info action")
            sys.exit(1)
        show_checkpoint_info(checkpoint_mgr, args.phase)
    elif args.action == "clean":
        clean_checkpoints(checkpoint_mgr, args.keep, args.phase)
    elif args.action == "clear":
        clear_checkpoints(checkpoint_mgr, args.phase)


if __name__ == "__main__":
    main()
