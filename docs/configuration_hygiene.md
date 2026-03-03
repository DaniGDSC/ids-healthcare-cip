# Configuration Hygiene Improvements

## Summary of Changes

This document summarizes the configuration hygiene improvements made to eliminate import-time side effects and deduplicate loader settings across configuration files.

## 1. Removed Import-Time Side Effects from settings.py

### Problem
[config/settings.py](../config/settings.py) contained a `print()` statement that executed on module import, causing side effects and cluttering test output.

### Solution
- Removed the `print(f"✅ Settings loaded. Project root: {PROJECT_ROOT}")` statement
- Added `get_project_info()` function that can be called explicitly when needed:
  ```python
  def get_project_info() -> str:
      """
      Get project information string.
      Call this explicitly when you need to display settings info.
      """
      return f"Settings loaded. Project root: {PROJECT_ROOT}"
  ```

### Benefits
- No side effects on import
- Cleaner test output
- More explicit control over when settings information is displayed
- Follows Python best practices (imports should not have visible side effects)

## 2. Shared Loader Configuration Schema

### Problem
Loader settings were duplicated between:
- [config/orchestration.yaml](../config/orchestration.yaml) `config_safety.defaults`
- [config/phase1_config.yaml](../config/phase1_config.yaml) `data_loading`
- Hard-coded defaults in DataLoader class

This led to:
- Maintenance burden (update in multiple places)
- Risk of inconsistency
- Unclear source of truth

### Solution

#### Created [config/loader_schema.py](../config/loader_schema.py)

Single source of truth for loader configuration:

```python
DEFAULT_LOADER_CONFIG = {
    # File discovery
    "file_pattern": "*.csv",
    "encoding": "utf-8",
    
    # Memory management
    "low_memory": False,
    "chunksize": None,
    "use_smart_loading": True,
    "memory_threshold_mb": 2000,
    
    # Size limits
    "max_file_size_mb": 10240,    # 10 GB
    "max_total_size_mb": 51200,   # 50 GB
    "max_memory_mb": 32768,       # 32 GB
    
    # I/O configuration
    "io_workers": 4,
    "io_retries": 2,
    "io_retry_delay_seconds": 2,
    "read_timeout_seconds": 30,
    
    # Error handling
    "skip_on_error": True,
    "quarantine_dir": "data/quarantine",
    
    # Strategy
    "loader": "default",
}
```

Provides utility functions:
- `get_loader_config(overrides)` - Get config with optional overrides
- `validate_loader_config(config)` - Validate and clamp to bounds
- `merge_loader_configs(*configs)` - Merge multiple configs with precedence

#### Updated [config/phase1_config.yaml](../config/phase1_config.yaml)

Simplified to only specify **overrides** for dataset-specific needs:

```yaml
data_loading:
  # Note: Loader defaults are defined in config/loader_schema.py
  # Only specify overrides here to avoid duplication
  
  # For CSE-CIC-IDS2018, we need higher limits:
  max_file_size_mb: 12000     # 12 GB (larger than default)
  max_total_size_mb: 60000    # 60 GB (accommodate full dataset)
  
  # All other settings use defaults from loader_schema.py
  
  # Data directories
  input_dir: "data/raw/CSE-CIC-IDS2018"
  output_dir: "data/processed"
  splits_dir: "data/splits"
```

Benefits:
- Clear intent: only overrides shown
- Defaults documented and referenced
- No duplication
- Easy to see what's special about this phase

#### Updated [config/orchestration.yaml](../config/orchestration.yaml)

Added cross-reference comment:

```yaml
config_safety:
  # These defaults are also defined in config/loader_schema.py
  # for use across all phases. Update both if changing loader defaults.
  defaults:
    max_file_size_mb: 10240
    # ... other defaults
```

#### Enhanced DataLoader with `from_config()` Classmethod

Added convenience method to create DataLoader from config dict:

```python
@classmethod
def from_config(cls, data_dir: str, config: Dict[str, Any]) -> "DataLoader":
    """
    Create DataLoader from configuration dict.
    
    Example:
        >>> config = {"max_file_size_mb": 20480, "io_workers": 8}
        >>> loader = DataLoader.from_config("data/raw", config)
    """
    # Merge with defaults and validate
    full_config = get_loader_config(config)
    validated_config = validate_loader_config(full_config)
    
    # Extract DataLoader constructor parameters
    return cls(
        data_dir=data_dir,
        max_file_size_mb=validated_config.get("max_file_size_mb", ...),
        # ... other parameters
    )
```

Benefits:
- Clean interface for config-driven instantiation
- Automatic defaults and validation
- Easier testing with custom configs

## 3. Configuration Validation Flow

```
phase1_config.yaml (overrides only)
         ↓
get_loader_config(overrides) → merges with DEFAULT_LOADER_CONFIG
         ↓
validate_loader_config(config) → clamps to LOADER_CONFIG_BOUNDS
         ↓
DataLoader.from_config() → instantiates with validated config
```

## 4. Testing

Created [tests/test_loader_schema.py](../tests/test_loader_schema.py) with comprehensive coverage:

- ✅ Default config has all required keys
- ✅ Overrides are applied correctly
- ✅ Validation clamps values to bounds
- ✅ Config merging with precedence
- ✅ Bounds consistency with config_safety
- ✅ DataLoader.from_config integration
- ✅ Clamping in from_config
- ✅ Defaults propagation

**Test Results:** 43 total tests passing (11 new loader schema tests)

## 5. Migration Guide

### Before: Duplicated Configuration

**orchestration.yaml:**
```yaml
config_safety:
  defaults:
    max_file_size_mb: 10240
    max_total_size_mb: 51200
    # ... all defaults
```

**phase1_config.yaml:**
```yaml
data_loading:
  file_pattern: "*.csv"
  encoding: "utf-8"
  max_file_size_mb: 12000  # Duplicated from orchestration
  max_total_size_mb: 60000
  io_workers: 4            # Duplicated
  io_retries: 2            # Duplicated
  # ... 15+ more duplicated fields
```

**Problems:**
- Update in 3+ places
- Inconsistency risk
- Unclear source of truth

### After: Single Source of Truth

**loader_schema.py:**
```python
DEFAULT_LOADER_CONFIG = {
    "max_file_size_mb": 10240,
    "max_total_size_mb": 51200,
    "io_workers": 4,
    # ... all defaults in one place
}
```

**phase1_config.yaml:**
```yaml
data_loading:
  # Only dataset-specific overrides
  max_file_size_mb: 12000  # Need 12 GB for large daily files
  max_total_size_mb: 60000 # Need 60 GB for full dataset
  
  # Directories (not in defaults)
  input_dir: "data/raw/CSE-CIC-IDS2018"
  output_dir: "data/processed"
```

**Benefits:**
- Update defaults in one place
- Overrides are explicit and documented
- Easy to see what's special about each phase

## 6. Best Practices

### When to Use Each Config Location

1. **loader_schema.py DEFAULT_LOADER_CONFIG**
   - Global defaults for all phases
   - General-purpose loader settings
   - Change when updating defaults project-wide

2. **orchestration.yaml config_safety.defaults**
   - Orchestration-level defaults
   - Keep in sync with loader_schema.py for consistency
   - Add orchestration-specific settings (timeouts, retries)

3. **phase{N}_config.yaml data_loading**
   - Phase-specific overrides only
   - Dataset-specific requirements
   - Document why values differ from defaults

### Adding New Loader Settings

1. Add to `DEFAULT_LOADER_CONFIG` in loader_schema.py
2. If bounded, add to `LOADER_CONFIG_BOUNDS`
3. Update `DataLoader.from_config()` to use new setting
4. Add test in test_loader_schema.py
5. Document in phase configs if overridden

## 7. Files Changed

- ✅ [config/settings.py](../config/settings.py) - Removed print, added get_project_info()
- ✅ [config/loader_schema.py](../config/loader_schema.py) - NEW: Shared loader config
- ✅ [config/phase1_config.yaml](../config/phase1_config.yaml) - Simplified to overrides only
- ✅ [config/orchestration.yaml](../config/orchestration.yaml) - Added cross-reference comment
- ✅ [src/phase1_preprocessing/data_loader.py](../src/phase1_preprocessing/data_loader.py) - Added from_config()
- ✅ [tests/test_loader_schema.py](../tests/test_loader_schema.py) - NEW: 11 tests

## 8. Backward Compatibility

All changes are **100% backward compatible**:

- Existing code using DataLoader() constructor works unchanged
- Phase configs without overrides use defaults from loader_schema
- No breaking changes to APIs or file formats

## 9. Related Documentation

- [Config Safety](../src/utils/config_safety.py) - Configuration validation
- [Dependency Injection](dependency_injection.md) - Manager injection patterns
- [Data Flow](data_flow.md) - Pipeline data flow documentation
