# Dependency Injection for Prefect Tasks

## Overview

The Prefect orchestration pipeline now supports dependency injection for managers (CheckpointManager, BackupManager, StorageValidator), enabling easier testing and modular architecture.

## Architecture

### Factory Pattern

The `create_managers()` factory function centralizes manager instantiation:

```python
from scripts.prefect_pipeline import create_managers

config = {
    "checkpoints_dir": "results/checkpoints",
    "backup": {
        "backup_dir": "backups",
        "retention_count": 7,
        "retention_days": 30
    }
}

checkpoint_mgr, backup_mgr, storage_validator = create_managers(
    config, project_root=Path("/path/to/project")
)
```

**Benefits:**
- Single source of truth for manager configuration
- Easy to swap implementations for testing
- Consistent initialization across orchestration and tests

### Optional Injection into orchestrate()

The `orchestrate()` flow accepts optional manager instances:

```python
orchestrate(
    resume=True,
    checkpoint_mgr=custom_checkpoint_mgr,  # Optional
    backup_mgr=custom_backup_mgr,          # Optional
    storage_validator=custom_validator,    # Optional
    config=custom_config                   # Optional
)
```

If managers are not provided, `create_managers()` is called automatically using the config.

## Testing Patterns

### Pattern 1: Factory-Based Integration Tests

Use `create_managers()` with temporary directories for integration tests:

```python
import tempfile
from pathlib import Path
from scripts.prefect_pipeline import create_managers

def test_pipeline_integration():
    with tempfile.TemporaryDirectory() as tmpdir:
        config = {
            "checkpoints_dir": "checkpoints",
            "backup": {
                "backup_dir": "backups",
                "retention_count": 3,
                "retention_days": 7
            }
        }
        
        # Get real manager instances pointing to temp directory
        checkpoint_mgr, backup_mgr, storage_validator = create_managers(
            config, project_root=Path(tmpdir)
        )
        
        # Use in tests without polluting actual project directories
        # All filesystem operations happen in tmpdir
        assert checkpoint_mgr.checkpoints_dir.startswith(str(tmpdir))
```

**When to use:**
- Integration tests that need real filesystem behavior
- Testing manager interactions
- Validating configuration propagation

### Pattern 2: Mock run_phase Task Directly

Mock the Prefect task to test orchestration logic:

```python
from unittest.mock import Mock, patch
from scripts.prefect_pipeline import orchestrate

def test_orchestration_flow():
    with patch('scripts.prefect_pipeline.run_phase') as mock_task:
        # Configure mock task behavior
        mock_task.submit.return_value.result.return_value = None
        
        # Run orchestrate (will use mocked task)
        orchestrate(config={
            "phases": [{"name": "phase1", "enabled": True}],
            "backup": {"enabled": False},
            "storage_validation": {"enabled": False}
        })
        
        # Verify task was called with correct parameters
        assert mock_task.submit.called
        call_args = mock_task.submit.call_args
        assert call_args[1]['phase_name'] == 'phase1'
```

**When to use:**
- Testing orchestration logic (phase sequencing, resume logic)
- Avoiding actual Prefect runtime
- Fast unit tests

### Pattern 3: Test run_phase.fn Directly with Mocks

Call the underlying task function with mock managers:

```python
from unittest.mock import Mock
from scripts.prefect_pipeline import run_phase

def test_run_phase_with_mocks():
    # Create properly configured mocks
    mock_checkpoint = Mock()
    mock_checkpoint.checkpoint_exists.return_value = False
    mock_checkpoint.save_checkpoint.return_value = None
    
    mock_backup = Mock()
    mock_backup.create_backup.return_value = Path("/fake/backup.zip")
    mock_backup.verify_backup.return_value = True
    
    mock_storage = Mock()
    mock_storage.check_space_for_phase.return_value = True
    
    # Call task function directly (bypasses Prefect serialization)
    run_phase.fn(
        phase_name="phase1",
        phase_config={"enabled": True, "timeout_seconds": 300},
        checkpoint_mgr=mock_checkpoint,
        backup_mgr=mock_backup,
        storage_validator=mock_storage
    )
    
    # Verify interactions
    mock_checkpoint.save_checkpoint.assert_called()
    mock_backup.create_backup.assert_called()
```

**When to use:**
- Testing individual task logic
- Verifying manager interactions
- Testing error paths (e.g., low space, backup failures)

**Important:** Use `.fn` to access the underlying function without Prefect runtime. This avoids parameter serialization issues with Mock objects.

## Prefect Serialization Constraints

Prefect serializes flow parameters using JSON encoding, which means:

- **Mock objects cannot be passed directly** to `@flow` decorated functions
- Use one of the patterns above to work around this limitation
- The `create_managers()` factory creates real instances that can be serialized

## Configuration

### orchestration.yaml

The factory reads configuration from `orchestration.yaml`:

```yaml
orchestration:
  checkpoints_dir: "results/checkpoints"
  
  backup:
    enabled: true
    backup_dir: "backups"
    retention_count: 7
    retention_days: 30
  
  storage_validation:
    enabled: true
    min_free_space_mb: 1000
```

### Custom Config Override

Override config at runtime:

```python
custom_config = {
    "checkpoints_dir": "custom/checkpoints",
    "backup": {
        "backup_dir": "custom/backups",
        "retention_count": 3,
        "retention_days": 14
    }
}

orchestrate(config=custom_config)
```

## Migration Guide

### Before (Hard-Coded Managers)

```python
@flow
def orchestrate(resume: bool = True):
    checkpoint_mgr = CheckpointManager("results/checkpoints")
    backup_mgr = BackupManager("backups", 7, 30)
    storage_validator = StorageValidator(PROJECT_ROOT)
    
    # ... orchestration logic
```

**Problems:**
- Managers hard-coded inside flow
- Difficult to test without filesystem side effects
- Can't swap implementations

### After (Dependency Injection)

```python
@flow
def orchestrate(
    resume: bool = True,
    checkpoint_mgr: CheckpointManager | None = None,
    backup_mgr: BackupManager | None = None,
    storage_validator: StorageValidator | None = None,
    config: Dict[str, Any] | None = None
):
    # Initialize via factory if not injected
    if checkpoint_mgr is None or backup_mgr is None or storage_validator is None:
        _checkpoint_mgr, _backup_mgr, _storage_validator = create_managers(cfg)
        checkpoint_mgr = checkpoint_mgr or _checkpoint_mgr
        backup_mgr = backup_mgr or _backup_mgr
        storage_validator = storage_validator or _storage_validator
    
    # ... orchestration logic
```

**Benefits:**
- Managers can be injected for testing
- Factory centralizes configuration
- Backward compatible (existing code works unchanged)

## Examples

See [test_prefect_injection.py](../tests/test_prefect_injection.py) for complete examples:

- `TestManagerFactory`: Tests factory with custom config and defaults
- `TestPrefectDependencyInjection`: Tests factory integration with orchestrate
- `TestManagerMockingPattern`: Examples of common mocking patterns

## Best Practices

1. **Use factory for integration tests** - Real managers with temp directories
2. **Mock run_phase for orchestration tests** - Test phase sequencing without execution
3. **Use .fn for unit tests** - Test individual task logic with mock managers
4. **Configure mocks properly** - Ensure all methods called by code return expected types
5. **Avoid direct Mock injection into @flow** - Prefect can't serialize Mock objects

## Related Documentation

- [Checkpointing System](checkpointing_system.md) - CheckpointManager usage
- [Backup & Recovery](backup_recovery.md) - BackupManager and StorageValidator
- [Config Safety](../src/utils/config_safety.py) - Configuration validation
- [Prefect Documentation](https://docs.prefect.io/) - Official Prefect docs
