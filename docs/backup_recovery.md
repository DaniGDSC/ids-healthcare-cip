# Backup & Recovery - RPO/RTO Documentation

## Overview

The IDS Healthcare CIP pipeline implements a comprehensive backup and recovery strategy to ensure data durability, business continuity, and rapid recovery from failures.

## RPO/RTO Targets

### Recovery Point Objective (RPO)

**RPO: 1 hour** - Maximum acceptable data loss in case of failure

| Component | RPO Target | Backup Frequency | Implementation |
|-----------|------------|------------------|----------------|
| Raw Data | N/A (immutable) | One-time | Original dataset preserved |
| Processed Data | 1 hour | Per phase completion | Automatic checkpoint + backup |
| Models | 1 hour | Per training completion | Versioned model checkpoints + backup |
| Results | 1 hour | Per phase completion | Automatic result backup |
| Configuration | 15 minutes | Version controlled | Git commits |

### Recovery Time Objective (RTO)

**RTO: 30 minutes** - Maximum acceptable downtime for recovery

| Failure Scenario | RTO Target | Recovery Steps | Estimated Time |
|------------------|------------|----------------|----------------|
| Phase failure | 5 minutes | Resume from last checkpoint | < 5 min |
| Data corruption | 15 minutes | Restore from backup + verify | 10-15 min |
| Full system failure | 30 minutes | Restore environment + data + resume | 20-30 min |
| Model loss | 10 minutes | Load versioned model checkpoint | 5-10 min |
| Configuration loss | 5 minutes | Git checkout + reapply | < 5 min |

## Backup Architecture

### Backup Types

1. **Checkpoints** (Primary Recovery Mechanism)
   - Location: `results/checkpoints/`
   - Format: Feather/Parquet + JSON metadata
   - Frequency: After each phase step
   - Retention: 2 latest versions per phase
   - RPO: < 1 hour

2. **Full Backups** (Secondary Recovery)
   - Location: `backups/`
   - Format: Compressed tar.gz archives
   - Frequency: After each phase completion
   - Retention: 7 latest + 30 days
   - RPO: 1 hour

3. **Model Versioning**
   - Location: `backups/models/`
   - Format: Pickle + JSON metadata with metrics
   - Frequency: Per training run
   - Retention: 7 latest versions
   - RPO: Per training iteration

### Backup Components

```
backups/
├── checkpoints/
│   ├── checkpoints_phase1_20260127_143052.tar.gz
│   ├── checkpoints_phase1_20260127_143052.json
│   ├── checkpoints_phase2_20260127_144523.tar.gz
│   └── checkpoints_phase2_20260127_144523.json
├── models/
│   ├── models_phase3_20260127_150012.tar.gz
│   ├── models_phase3_20260127_150012.json
│   ├── models_phase5_20260127_153045.tar.gz
│   └── models_phase5_20260127_153045.json
└── results/
    ├── results_phase1_20260127_143100.tar.gz
    ├── results_phase1_20260127_143100.json
    ├── results_phase5_20260127_153200.tar.gz
    └── results_phase5_20260127_153200.json
```

## Configuration

### Orchestration Config (`config/orchestration.yaml`)

```yaml
orchestration:
  # Backup configuration
  backup:
    enabled: true              # Auto-backup after each phase
    backup_dir: "backups"      # Backup root directory
    retention_count: 7         # Keep 7 latest backups per type
    retention_days: 30         # Keep backups newer than 30 days
    compression: true          # Use gzip compression
  
  # Storage validation
  storage_validation:
    enabled: true              # Check disk space before phases
    fail_on_error: false       # Warn only (don't fail pipeline)
    min_free_gb: 10            # Minimum 10GB free space required
```

### Automated Backup Schedule

| Event | Backup Trigger | Components Backed Up |
|-------|---------------|---------------------|
| Phase 1 completion | Automatic | Checkpoints, processed data |
| Phase 2 completion | Automatic | Checkpoints, selected features |
| Phase 3 completion | Automatic | Checkpoints, encoder model, latent features |
| Phase 4 completion | Automatic | Checkpoints, clustering results |
| Phase 5 completion | Automatic | Checkpoints, final models, predictions, reports |
| Manual trigger | On-demand | User-specified components |

## Storage Space Management

### Pre-Flight Checks

Before pipeline execution, automatic storage validation:

```python
from src.utils.storage_validator import StorageValidator

validator = StorageValidator()
has_space, details = validator.check_space_for_pipeline(
    phases=["phase1", "phase2", "phase3", "phase4", "phase5"]
)
```

### Space Requirements

| Phase | Input | Working | Output | Total (with margin) |
|-------|-------|---------|--------|---------------------|
| Phase 1 | 5 GB | 10 GB | 3 GB | ~22 GB |
| Phase 2 | 3 GB | 2 GB | 500 MB | ~7 GB |
| Phase 3 | 500 MB | 2 GB | 1 GB | ~4 GB |
| Phase 4 | 1 GB | 1 GB | 500 MB | ~3 GB |
| Phase 5 | 1.5 GB | 1 GB | 500 MB | ~4 GB |
| **Total** | | | | **~40 GB** |

### Storage Monitoring

```bash
# Check current disk usage
python scripts/backup_utils.py storage

# Get cleanup suggestions
python scripts/backup_utils.py cleanup-suggestions

# Free space by removing old backups
python scripts/backup_utils.py cleanup --keep 3
```

## Backup Operations

### Creating Backups

**Automatic (via pipeline):**
```bash
# Backups created automatically during pipeline execution
python scripts/prefect_pipeline.py
```

**Manual:**
```python
from src.utils.backup_manager import BackupManager

mgr = BackupManager()

# Backup checkpoints
mgr.backup_directory(
    "results/checkpoints/phase1",
    backup_type="checkpoints",
    tag="phase1"
)

# Backup models
mgr.backup_directory(
    "models/phase5",
    backup_type="models",
    tag="phase5"
)

# Backup results
mgr.backup_directory(
    "results/phase5",
    backup_type="results",
    tag="phase5"
)
```

### Listing Backups

```python
from src.utils.backup_manager import BackupManager

mgr = BackupManager()

# List all backups
all_backups = mgr.list_backups()

# List specific type
model_backups = mgr.list_backups(backup_type="models")

# List specific tag
phase5_backups = mgr.list_backups(backup_type="models", tag="phase5")

# Get latest backup
latest = mgr.get_latest_backup("models", tag="phase5")
print(f"Latest model backup: {latest['version']}")
```

### Restoring Backups

```python
from src.utils.backup_manager import BackupManager

mgr = BackupManager()

# Restore checkpoint backup
mgr.restore_backup(
    "backups/checkpoints/checkpoints_phase1_20260127_143052.tar.gz",
    restore_dir="results/checkpoints",
    verify_checksum=True
)

# Restore model backup
mgr.restore_backup(
    "backups/models/models_phase5_20260127_153045.tar.gz",
    restore_dir="models",
    verify_checksum=True
)
```

### Backup Verification

```python
from src.utils.backup_manager import BackupManager

mgr = BackupManager()

# Verify backup integrity
is_valid, message = mgr.verify_backup(
    "backups/models/models_phase5_20260127_153045.tar.gz"
)

if is_valid:
    print(f"✓ Backup verified: {message}")
else:
    print(f"✗ Verification failed: {message}")
```

## Recovery Procedures

### Scenario 1: Phase Failure During Execution

**Symptom:** Pipeline fails partway through a phase

**RPO:** Last completed checkpoint (< 1 hour)  
**RTO:** < 5 minutes

**Recovery Steps:**
```bash
# 1. Resume pipeline from last checkpoint
python scripts/prefect_pipeline.py --resume

# Pipeline automatically:
# - Detects completed phases via checkpoints
# - Skips completed work
# - Resumes from failure point
```

### Scenario 2: Data Corruption in Checkpoint

**Symptom:** Checkpoint files corrupted or invalid

**RPO:** Last valid backup (< 1 hour)  
**RTO:** 10-15 minutes

**Recovery Steps:**
```python
# 1. Identify corrupted phase
from src.utils.checkpoint_manager import CheckpointManager
mgr = CheckpointManager()
mgr.list_checkpoints("phase3")

# 2. Clear corrupted checkpoint
from scripts.checkpoint_utils import clear_checkpoints
clear_checkpoints(mgr, phase="phase3")

# 3. Restore from backup
from src.utils.backup_manager import BackupManager
backup_mgr = BackupManager()
latest = backup_mgr.get_latest_backup("checkpoints", tag="phase3")
backup_mgr.restore_backup(
    latest['archive_path'],
    "results/checkpoints",
    verify_checksum=True
)

# 4. Resume pipeline
# python scripts/prefect_pipeline.py --resume
```

**Estimated Time:** 10-15 minutes

### Scenario 3: Model Loss or Corruption

**Symptom:** Trained model files missing or corrupted

**RPO:** Last model training (< 1 hour)  
**RTO:** 5-10 minutes

**Recovery Steps:**
```python
# 1. List available model backups
from src.utils.backup_manager import BackupManager
mgr = BackupManager()
backups = mgr.list_backups("models", tag="phase5")

# 2. Restore latest valid backup
latest = mgr.get_latest_backup("models", tag="phase5")
mgr.restore_backup(
    f"backups/{latest['archive_path']}",
    restore_dir="models",
    verify_checksum=True
)

# 3. Verify model loads correctly
from src.utils.checkpoint_manager import CheckpointManager
ckpt_mgr = CheckpointManager()
model, metadata = ckpt_mgr.load_model("phase5", "ensemble_classifier")
print(f"Restored model from {metadata['timestamp']}")
print(f"Test accuracy: {metadata['metrics']['test_accuracy']}")
```

**Estimated Time:** 5-10 minutes

### Scenario 4: Complete System Failure

**Symptom:** Total environment loss, all local files gone

**RPO:** Last backup sync (< 1 hour)  
**RTO:** 20-30 minutes

**Recovery Steps:**
```bash
# 1. Restore environment
git clone <repository>
cd ids-healthcare-cip
python -m venv ids
source ids/bin/activate
pip install -r requirements.txt

# 2. Restore raw data (if not backed up, re-download)
bash scripts/download_dataset.sh

# 3. Restore latest backups
python -c "
from src.utils.backup_manager import BackupManager
mgr = BackupManager()

# Restore all components
for backup_type in ['checkpoints', 'models', 'results']:
    backups = mgr.list_backups(backup_type)
    for phase in backups:
        for backup in phase:
            mgr.restore_backup(backup['archive_path'], 'restore_temp')
"

# 4. Resume pipeline from restored checkpoints
python scripts/prefect_pipeline.py --resume
```

**Estimated Time:** 20-30 minutes

### Scenario 5: Rollback to Previous Model Version

**Symptom:** New model performs worse, need to rollback

**RPO:** N/A (historical version)  
**RTO:** < 5 minutes

**Recovery Steps:**
```python
# 1. List model versions with metrics
from src.utils.checkpoint_manager import CheckpointManager
mgr = CheckpointManager()

checkpoints = mgr.list_checkpoints("phase5")
for ckpt in checkpoints["phase5"]:
    print(f"Version: {ckpt['version']}")
    print(f"  Accuracy: {ckpt['metrics']['test_accuracy']:.4f}")
    print(f"  F1: {ckpt['metrics']['test_f1_weighted']:.4f}")

# 2. Load specific version
model, metadata = mgr.load_model(
    "phase5",
    "ensemble_classifier",
    version="20260127_140000"  # Previous better version
)

# 3. Deploy reverted model
import pickle
with open("models/phase5/ensemble_current.pkl", "wb") as f:
    pickle.dump(model, f)
```

**Estimated Time:** < 5 minutes

## Retention & Cleanup

### Automatic Cleanup

Runs before each pipeline execution:

```python
# Configured in orchestration.yaml
retention_count: 7   # Keep 7 latest backups
retention_days: 30   # Keep backups < 30 days old
```

### Manual Cleanup

```python
from src.utils.backup_manager import BackupManager

mgr = BackupManager()

# Preview cleanup (dry run)
deleted = mgr.cleanup_old_backups(dry_run=True)
print(f"Would delete {len(deleted)} backups")

# Execute cleanup
deleted = mgr.cleanup_old_backups(dry_run=False)
print(f"Deleted {len(deleted)} old backups")

# Get backup sizes
sizes = mgr.get_backup_size()
print(f"Checkpoints: {sizes['checkpoints']}MB")
print(f"Models: {sizes['models']}MB")
print(f"Results: {sizes['results']}MB")
```

## Monitoring & Alerts

### Backup Health Checks

```python
from src.utils.backup_manager import BackupManager

mgr = BackupManager()

# Verify all recent backups
for backup_type in ["checkpoints", "models", "results"]:
    backups = mgr.list_backups(backup_type)
    for phase_backups in backups.values():
        for backup in phase_backups[:3]:  # Check 3 latest
            is_valid, msg = mgr.verify_backup(backup['archive_path'])
            if not is_valid:
                print(f"⚠️  Invalid backup: {backup['archive_path']}")
                print(f"   {msg}")
```

### Storage Monitoring

```python
from src.utils.storage_validator import StorageValidator

validator = StorageValidator()

# Check current usage
usage = validator.get_disk_usage(Path.cwd())
print(f"Free space: {usage['free_mb']:.0f}MB ({100-usage['percent_used']:.1f}% free)")

# Check if cleanup needed
needs_cleanup, needed_mb = validator.check_cleanup_needed(target_free_mb=10000)
if needs_cleanup:
    print(f"Need to free {needed_mb:.0f}MB")
    
    # Get suggestions
    suggestions = validator.suggest_cleanup_targets()
    for name, info in suggestions.items():
        print(f"  {name}: {info['size_mb']:.0f}MB")
```

## Best Practices

1. **Enable automatic backups** for production pipelines
2. **Verify backups periodically** using `verify_backup()`
3. **Test recovery procedures** monthly in non-production environment
4. **Monitor disk space** before long-running operations
5. **Keep retention policies balanced** - enough history vs disk usage
6. **Document configuration changes** affecting backup behavior
7. **Use version tags** for important model releases
8. **Maintain offsite backups** for critical deployments (cloud storage)

## Disaster Recovery Plan

### Critical Data Priority

1. **P0 - Critical (RPO: 15 min)**
   - Configuration files (version controlled)
   - Trained models (versioned checkpoints)

2. **P1 - High (RPO: 1 hour)**
   - Processed data (checkpoints + backups)
   - Phase results (automatic backups)

3. **P2 - Medium (RPO: 24 hours)**
   - Logs (retained locally)
   - Temporary intermediate files

4. **P3 - Low (RPO: N/A)**
   - Raw data (immutable, can re-download)
   - Cache files (can regenerate)

### Recovery Validation

After any recovery operation:

```python
# 1. Verify data integrity
from src.utils.checkpoint_manager import CheckpointManager
mgr = CheckpointManager()
df, metadata = mgr.load_dataframe("phase1", "preprocessed_data")
assert df.shape == metadata['dataframe_shape']

# 2. Verify model integrity
model, model_meta = mgr.load_model("phase5", "ensemble_classifier")
assert model_meta['model_type'] == 'EnsembleClassifier'

# 3. Run validation tests
pytest tests/test_phase1.py
pytest tests/test_phase5.py

# 4. Compare metrics with historical
assert model_meta['metrics']['test_accuracy'] > 0.95
```

## Cloud Backup Integration (Optional)

For production deployments, extend BackupManager for cloud storage:

```python
# Future enhancement example
class CloudBackupManager(BackupManager):
    def __init__(self, cloud_provider="s3", bucket_name=None, **kwargs):
        super().__init__(**kwargs)
        self.cloud_provider = cloud_provider
        self.bucket_name = bucket_name
    
    def sync_to_cloud(self, backup_type: str):
        """Upload backups to cloud storage."""
        # Implementation for S3, Azure Blob, GCS, etc.
        pass
    
    def restore_from_cloud(self, backup_key: str, restore_dir: str):
        """Restore backup from cloud storage."""
        pass
```

**Recommended Cloud Providers:**
- AWS S3 (with S3 Glacier for long-term retention)
- Azure Blob Storage
- Google Cloud Storage
- Wasabi (cost-effective alternative)

## Support & Troubleshooting

### Common Issues

**Issue:** "Insufficient disk space" error  
**Solution:** Run cleanup or free space manually
```bash
python scripts/backup_utils.py cleanup --keep 3
rm -rf logs/*
```

**Issue:** Backup verification fails  
**Solution:** Check for partial downloads or corruption
```python
# Re-create backup
mgr.backup_directory("results/checkpoints/phase1", "checkpoints", "phase1")
```

**Issue:** Restore fails with "Archive not found"  
**Solution:** Check backup location and permissions
```bash
ls -la backups/checkpoints/
```

---

**Document Version:** 1.0  
**Last Updated:** 2026-01-27  
**Next Review:** 2026-04-27 (Quarterly)
