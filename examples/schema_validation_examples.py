"""
Example integration of pydantic schemas with Phase 1.

This file demonstrates how to use the data contract schemas to prevent
silent failures in inter-phase data passing.

Usage:
    # In Phase 1 run_phase1.py, instead of saving raw numpy arrays:
    from src.schemas import Phase1Output, DataContractValidator
    from src.utils.checkpoint_manager import CheckpointManager
    
    # Create schema object (validates all constraints)
    phase1_output = Phase1Output(
        X_train_normalized=X_train_norm,
        X_val_normalized=X_val_norm,
        X_test_normalized=X_test_norm,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        feature_names=feature_names,
        feature_count=len(feature_names),
        train_size=X_train_norm.shape[0],
        val_size=X_val_norm.shape[0],
        test_size=X_test_norm.shape[0],
        config_hash=checkpoint_mgr._compute_config_hash(config)
    )
    
    # Validate contract (optional but recommended)
    DataContractValidator.validate_phase1_to_phase2(phase1_output)
    
    # Save with validation
    checkpoint_mgr.validate_and_save_pydantic(
        "phase1",
        phase1_output,
        config=config
    )
"""

from typing import Dict, Any
import logging
import numpy as np
from pathlib import Path
from src.schemas import Phase1Output, DataContractValidator
from src.utils.checkpoint_manager import CheckpointManager

logger = logging.getLogger(__name__)


def example_save_phase1_output():
    """Example: Save Phase 1 output with schema validation."""
    
    # Simulate Phase 1 outputs
    X_train_norm = np.random.randn(1000, 52)  # 52 selected features
    X_val_norm = np.random.randn(250, 52)
    X_test_norm = np.random.randn(250, 52)
    
    y_train = np.random.randint(0, 7, 1000)
    y_val = np.random.randint(0, 7, 250)
    y_test = np.random.randint(0, 7, 250)
    
    feature_names = [f"feature_{i}" for i in range(52)]
    
    # Create schema object (validates all constraints)
    try:
        phase1_output = Phase1Output(
            X_train_normalized=X_train_norm,
            X_val_normalized=X_val_norm,
            X_test_normalized=X_test_norm,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
            feature_names=feature_names,
            feature_count=len(feature_names),
            train_size=X_train_norm.shape[0],
            val_size=X_val_norm.shape[0],
            test_size=X_test_norm.shape[0]
        )
        print("✓ Phase 1 output passed schema validation")
    except ValueError as e:
        print(f"✗ Phase 1 output failed validation: {e}")
        return
    
    # Validate contract with Phase 2
    try:
        DataContractValidator.validate_phase1_to_phase2(phase1_output)
        print("✓ Phase 1→Phase 2 contract validation passed")
    except ValueError as e:
        print(f"✗ Phase 1→Phase 2 contract validation failed: {e}")
        return
    
    # Save with checkpoint manager
    checkpoint_mgr = CheckpointManager("results/checkpoints")
    config = {"data": {"normalization": "standard"}}
    
    metadata = checkpoint_mgr.validate_and_save_pydantic(
        "phase1",
        phase1_output,
        artifact_name="output",
        config=config
    )
    print(f"✓ Saved Phase 1 output: {metadata}")


def example_load_phase1_output():
    """Example: Load Phase 1 output with schema validation."""
    from src.schemas import Phase1Output
    
    checkpoint_mgr = CheckpointManager("results/checkpoints")
    config = {"data": {"normalization": "standard"}}
    
    try:
        phase1_output, metadata = checkpoint_mgr.load_and_validate_pydantic(
            "phase1",
            Phase1Output,
            artifact_name="output",
            allow_stale=False,
            config=config
        )
        print("✓ Loaded and validated Phase 1 output")
        print(f"  Train shape: {phase1_output.X_train_normalized.shape}")
        print(f"  Features: {phase1_output.feature_count}")
        print(f"  Config hash: {metadata.get('config_hash')}")
    except ValueError as e:
        print(f"✗ Failed to load Phase 1 output: {e}")
    except FileNotFoundError as e:
        print(f"✗ Checkpoint not found: {e}")


def example_stale_checkpoint_detection():
    """Example: Detect and reject stale checkpoints."""
    from src.schemas import Phase1Output
    
    checkpoint_mgr = CheckpointManager("results/checkpoints")
    
    # Old config that was used to create checkpoint
    old_config = {"data": {"normalization": "standard"}}
    
    # New config with different parameters
    new_config = {"data": {"normalization": "robust"}}
    
    try:
        # This will fail because config hash doesn't match
        phase1_output, metadata = checkpoint_mgr.load_and_validate_pydantic(
            "phase1",
            Phase1Output,
            artifact_name="output",
            allow_stale=False,
            config=new_config  # Different from saved config
        )
    except ValueError as e:
        print(f"✓ Correctly detected stale checkpoint: {e}")
    except FileNotFoundError:
        print("Checkpoint not found (expected if running example without prior save)")


def example_data_contract_violations():
    """Example: Detect data contract violations."""
    
    # Example 1: Wrong number of features
    try:
        phase1_output = Phase1Output(
            X_train_normalized=np.random.randn(1000, 52),
            X_val_normalized=np.random.randn(250, 52),
            X_test_normalized=np.random.randn(250, 52),
            y_train=np.random.randint(0, 7, 1000),
            y_val=np.random.randint(0, 7, 250),
            y_test=np.random.randint(0, 7, 250),
            feature_names=[f"f_{i}" for i in range(50)],  # Mismatch!
            feature_count=52,  # Mismatch!
            train_size=1000,
            val_size=250,
            test_size=250
        )
    except ValueError as e:
        print(f"✓ Caught feature count mismatch: {e}")
    
    # Example 2: Mismatched data shapes
    try:
        phase1_output = Phase1Output(
            X_train_normalized=np.random.randn(1000, 52),
            X_val_normalized=np.random.randn(250, 52),
            X_test_normalized=np.random.randn(250, 52),
            y_train=np.random.randint(0, 7, 1000),
            y_val=np.random.randint(0, 7, 250),
            y_test=np.random.randint(0, 7, 100),  # Wrong size!
            feature_names=[f"f_{i}" for i in range(52)],
            feature_count=52,
            train_size=1000,
            val_size=250,
            test_size=250
        )
    except ValueError as e:
        print(f"✓ Caught data shape mismatch: {e}")
    
    # Example 3: Phase 2 contract validation
    try:
        # Phase 2 expects exactly 35 features (per CIC-IDS2018 dataset)
        DataContractValidator.validate_phase1_to_phase2(
            Phase1Output(
                X_train_normalized=np.random.randn(1000, 50),  # Wrong!
                X_val_normalized=np.random.randn(250, 50),
                X_test_normalized=np.random.randn(250, 50),
                y_train=np.random.randint(0, 7, 1000),
                y_val=np.random.randint(0, 7, 250),
                y_test=np.random.randint(0, 7, 250),
                feature_names=[f"f_{i}" for i in range(50)],
                feature_count=50,
                train_size=1000,
                val_size=250,
                test_size=250
            )
        )
    except ValueError as e:
        print(f"✓ Caught Phase 1→2 contract violation: {e}")


if __name__ == "__main__":
    print("=" * 80)
    print("Schema Validation Examples")
    print("=" * 80)
    
    print("\n1. Saving Phase 1 output with validation:")
    example_save_phase1_output()
    
    print("\n2. Loading Phase 1 output with validation:")
    example_load_phase1_output()
    
    print("\n3. Stale checkpoint detection:")
    example_stale_checkpoint_detection()
    
    print("\n4. Data contract violation detection:")
    example_data_contract_violations()
    
    print("\n" + "=" * 80)
    print("Schema validation helps prevent:")
    print("  - Silent data corruption between phases")
    print("  - Stale checkpoint reuse with new config")
    print("  - Mismatched data shapes and types")
    print("  - Downstream phase failures due to invalid inputs")
    print("=" * 80)
