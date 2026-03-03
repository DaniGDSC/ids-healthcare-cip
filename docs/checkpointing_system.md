# Checkpoint & Recovery System

## Overview

The IDS Healthcare CIP pipeline includes a comprehensive checkpoint and recovery system that enables:

- **Automatic checkpoint saving** at the end of each pipeline phase
- **Resume from last good phase** on pipeline restart or failure
- **Model and artifact versioning** with timestamp-based versions
- **Metadata tracking** for reproducibility (config hash, metrics, timestamps)
- **Legacy checkpoint compatibility** with existing phase marker files

## Architecture

### CheckpointManager

Central utility class for managing checkpoints across all phases:

**Location:** `src/utils/checkpoint_manager.py`

**Key Features:**
- Save/load DataFrames (feather or parquet format)
- Save/load models (pickle format)
- Version tracking with timestamps
- Metadata storage (JSON sidecar files)
- Config hashing for reproducibility
- List/query available checkpoints
- Cleanup old versions

### Directory Structure

```
results/checkpoints/
├── phase1/
│   ├── preprocessed_data_20260127_143052.feather
│   ├── preprocessed_data_20260127_143052.json
│   ├── preprocessed_data_20260127_120000.feather
│   └── preprocessed_data_20260127_120000.json
├── phase2/
│   ├── selected_features_20260127_144523.feather
│   └── selected_features_20260127_144523.json
├── phase3/
│   ├── encoder_20260127_150012.pkl
│   ├── encoder_20260127_150012.json
│   ├── latent_features_20260127_150012.feather
│   └── latent_features_20260127_150012.json
└── phase5/
    ├── ensemble_classifier_20260127_153045.pkl
    ├── ensemble_classifier_20260127_153045.json
    ├── ensemble_classifier_20260127_140000.pkl
    └── ensemble_classifier_20260127_140000.json
```

### Metadata Format

Each checkpoint includes a JSON metadata file:

```json
{
  "timestamp": "2026-01-27T14:30:52.123456",
  "version": "20260127_143052",
  "phase": "phase1",
  "artifact_type": "preprocessed_data",
  "format": "feather",
  "file_path": "phase1/preprocessed_data_20260127_143052.feather",
  "dataframe_shape": [1000000, 45],
  "columns": ["feature1", "feature2", ...],
  "memory_mb": 123.456,
  "config_hash": "a1b2c3d4",
  "stats": {
    "rows_processed": 1000000,
    "features_selected": 35
  }
}
```

For models:
```json
{
  "timestamp": "2026-01-27T15:30:45.789012",
  "version": "20260127_153045",
  "phase": "phase5",
  "model_name": "ensemble_classifier",
  "model_type": "EnsembleClassifier",
  "file_path": "phase5/ensemble_classifier_20260127_153045.pkl",
  "config_hash": "e5f6g7h8",
  "metrics": {
    "test_accuracy": 0.9876,
    "test_f1_weighted": 0.9823,
    "val_accuracy": 0.9834
  }
}
```

## Usage

### Configuration

Enable checkpointing in phase configs:

**config/phase1_config.yaml:**
```yaml
checkpointing:
  enabled: true                 # Enable checkpoint save/load
  allow_resume: true            # Resume from last checkpoint if available
  checkpoint_dir: "results/checkpoints"
  keep_latest: 2                # Keep N latest versions per phase
```

**config/orchestration.yaml:**
```yaml
orchestration:
  resume: true
  checkpoints_dir: "results/checkpoints"
  
  checkpointing:
    enabled: true
    keep_latest: 2
    format: "feather"  # or "parquet"
```

### Running with Checkpoints

#### Full Pipeline with Resume
```bash
# Normal run - resumes from last checkpoint
python scripts/prefect_pipeline.py

# Explicit resume
python scripts/prefect_pipeline.py --resume
```

#### Fresh Start
```bash
# Ignore existing checkpoints, restart from beginning
python scripts/prefect_pipeline.py --fresh
```

#### Individual Phase
```bash
# Phase 1 will auto-resume if checkpoint exists
python src/phase1_preprocessing/run_phase1.py

# To force fresh run, disable in config:
# checkpointing.allow_resume: false
```

### Managing Checkpoints

**List all checkpoints:**
```bash
python scripts/checkpoint_utils.py list
```

**List specific phase:**
```bash
python scripts/checkpoint_utils.py list --phase phase1
```

**Show detailed info:**
```bash
python scripts/checkpoint_utils.py info --phase phase5
```

**Clean old versions (keep 2 latest):**
```bash
python scripts/checkpoint_utils.py clean --keep 2
```

**Clear specific phase:**
```bash
python scripts/checkpoint_utils.py clear --phase phase3
```

**Clear all checkpoints:**
```bash
python scripts/checkpoint_utils.py clear
```

### Programmatic Usage

```python
from src.utils.checkpoint_manager import CheckpointManager

# Initialize
checkpoint_mgr = CheckpointManager("results/checkpoints")

# Save DataFrame
checkpoint_mgr.save_dataframe(
    df,
    phase="phase1",
    artifact_name="preprocessed_data",
    config=config_dict,
    stats={"rows": len(df), "cols": len(df.columns)}
)

# Load DataFrame (latest version)
df, metadata = checkpoint_mgr.load_dataframe("phase1", "preprocessed_data")

# Load specific version
df, metadata = checkpoint_mgr.load_dataframe(
    "phase1", 
    "preprocessed_data",
    version="20260127_143052"
)

# Save model
checkpoint_mgr.save_model(
    model_object,
    phase="phase5",
    model_name="ensemble_classifier",
    config=config_dict,
    metrics={"test_accuracy": 0.987}
)

# Load model (latest)
model, metadata = checkpoint_mgr.load_model("phase5", "ensemble_classifier")

# Check if checkpoint exists
if checkpoint_mgr.has_checkpoint("phase1"):
    latest = checkpoint_mgr.get_latest_checkpoint("phase1")
    print(f"Latest: {latest['version']}")

# List all checkpoints
all_checkpoints = checkpoint_mgr.list_checkpoints()
for phase, checkpoints in all_checkpoints.items():
    print(f"{phase}: {len(checkpoints)} versions")
```

## Recovery Scenarios

### Scenario 1: Phase Failure

Pipeline fails during Phase 3:

```
Phase 1: ✓ (checkpoint saved)
Phase 2: ✓ (checkpoint saved)
Phase 3: ✗ (failed)
Phase 4: (not started)
Phase 5: (not started)
```

**Recovery:**
```bash
# Fix the issue, then re-run
python scripts/prefect_pipeline.py

# Phases 1-2 will be skipped (checkpoints exist)
# Phase 3 will execute from scratch
# Phases 4-5 will execute normally
```

### Scenario 2: Data Corruption

Suspect Phase 2 output is corrupted:

```bash
# Clear Phase 2 checkpoint
python scripts/checkpoint_utils.py clear --phase phase2

# Re-run pipeline
python scripts/prefect_pipeline.py

# Phase 1 skipped (checkpoint exists)
# Phase 2 will re-execute
# Phases 3-5 will re-execute (depend on Phase 2)
```

### Scenario 3: Experiment Comparison

Compare different model versions:

```python
from src.utils.checkpoint_manager import CheckpointManager

mgr = CheckpointManager()

# List all phase5 model versions
checkpoints = mgr.list_checkpoints("phase5")
for ckpt in checkpoints["phase5"]:
    print(f"Version: {ckpt['version']}")
    print(f"Accuracy: {ckpt['metrics']['test_accuracy']}")
    print(f"F1: {ckpt['metrics']['test_f1_weighted']}")
    print("---")

# Load specific version for analysis
model_v1, meta_v1 = mgr.load_model("phase5", "ensemble_classifier", version="20260127_140000")
model_v2, meta_v2 = mgr.load_model("phase5", "ensemble_classifier", version="20260127_153045")
```

## Best Practices

1. **Keep checkpoints enabled** in production to enable fast recovery
2. **Set `keep_latest: 2-3`** to balance disk space and version history
3. **Use `--fresh` sparingly** - only when explicitly restarting experiments
4. **Review checkpoint metadata** before resuming from old checkpoints
5. **Clean old checkpoints periodically** to manage disk usage
6. **Version control your configs** along with checkpoint metadata for full reproducibility

## Troubleshooting

### Checkpoint not loading

Check if checkpoint exists:
```bash
python scripts/checkpoint_utils.py list --phase phase1
```

Enable debug logging:
```python
import logging
logging.getLogger('src.utils.checkpoint_manager').setLevel(logging.DEBUG)
```

### Disk space issues

Clean old versions:
```bash
python scripts/checkpoint_utils.py clean --keep 1
```

### Phase always re-executes

Check `allow_resume` setting in phase config:
```yaml
checkpointing:
  enabled: true
  allow_resume: true  # Must be true
```

## Implementation Details

### Phase 1 Integration

```python
class Phase1Pipeline:
    def __init__(self, config, logger, checkpoint_dir="results/checkpoints"):
        self.checkpoint_mgr = CheckpointManager(checkpoint_dir)
        self.enable_checkpoints = config.get('checkpointing', {}).get('enabled', True)
        self.allow_resume = config.get('checkpointing', {}).get('allow_resume', True)
    
    def run(self):
        # Try to resume
        df = self._try_load_checkpoint("preprocessed_data")
        
        if df is None:
            # Load and preprocess from scratch
            df = self._load_data()
            # ... preprocessing steps ...
            
            # Save checkpoint
            self._save_checkpoint(df, "preprocessed_data")
        
        # Continue with splitting, normalization, etc.
```

### Phase 5 Integration

```python
def save_outputs(..., checkpoint_mgr=None):
    # Legacy save
    ensemble.save(model_dir)
    
    # Versioned checkpoint
    if checkpoint_mgr:
        checkpoint_mgr.save_model(
            ensemble,
            phase="phase5",
            model_name="ensemble_classifier",
            metrics={"test_accuracy": acc, "test_f1": f1}
        )
```

### Prefect Pipeline Integration

```python
@task
def run_phase(phase_name, checkpoint_mgr, ...):
    # Check CheckpointManager
    if resume and checkpoint_mgr.has_checkpoint(phase_name):
        logger.info(f"Checkpoint found, skipping {phase_name}")
        return
    
    # Execute phase
    subprocess.run(...)
    
    # Verify checkpoint was created
    if checkpoint_mgr.has_checkpoint(phase_name):
        logger.info(f"Checkpoint verified for {phase_name}")
```

## Future Enhancements

- Cloud storage backend (S3, Azure Blob)
- Checkpoint compression options
- Incremental checkpointing for large datasets
- Checkpoint diff/comparison tools
- Automatic cleanup policies based on age/count
- Checkpoint encryption for sensitive data
