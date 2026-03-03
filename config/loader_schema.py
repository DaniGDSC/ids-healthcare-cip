"""
Shared data loader configuration schema.

Consolidates loader settings used across phases to avoid duplication
and ensure consistent behavior.
"""

from typing import Dict, Any


# Default loader configuration
# Referenced by orchestration.yaml config_safety.defaults and phase configs
DEFAULT_LOADER_CONFIG = {
    # File discovery
    "file_pattern": "*.csv",
    "encoding": "utf-8",
    
    # Memory management
    "low_memory": False,
    "chunksize": None,  # Load entire file at once
    "use_smart_loading": True,  # Auto-chunk large files
    "memory_threshold_mb": 2000,  # Chunk if files exceed this size
    
    # Size limits (enforced by LoadSafetyPolicy)
    "max_file_size_mb": 10240,    # 10 GB per file
    "max_total_size_mb": 51200,   # 50 GB total dataset
    "max_memory_mb": 32768,       # 32 GB in-memory DataFrame
    
    # I/O configuration
    "io_workers": 4,  # Parallel file reads
    "io_retries": 2,  # Retry count for transient errors
    "io_retry_delay_seconds": 2,  # Delay between retries
    "read_timeout_seconds": 30,  # Timeout per file read
    
    # Error handling
    "skip_on_error": True,  # Skip files that fail after retries
    "quarantine_dir": "data/quarantine",
    
    # Strategy
    "loader": "default",  # Pluggable loader strategy
}


# Loader config bounds (min/max values)
# These match the bounds in orchestration.yaml config_safety.bounds
LOADER_CONFIG_BOUNDS = {
    "max_file_size_mb": {"min": 10, "max": 204800},       # 10 MB - 200 GB
    "max_total_size_mb": {"min": 100, "max": 409600},     # 100 MB - 400 GB
    "max_memory_mb": {"min": 512, "max": 262144},         # 512 MB - 256 GB
    "memory_threshold_mb": {"min": 128, "max": 131072},   # 128 MB - 128 GB
    "io_workers": {"min": 1, "max": 32},
    "io_retries": {"min": 0, "max": 10},
    "io_retry_delay_seconds": {"min": 0, "max": 300},
    "read_timeout_seconds": {"min": 0, "max": 600},
}


def get_loader_config(overrides: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """
    Get loader configuration with optional overrides.
    
    Args:
        overrides: Dict of config values to override defaults
        
    Returns:
        Merged configuration dict
        
    Example:
        >>> config = get_loader_config({"max_file_size_mb": 20480})
        >>> config["max_file_size_mb"]
        20480
    """
    config = DEFAULT_LOADER_CONFIG.copy()
    if overrides:
        config.update(overrides)
    return config


def validate_loader_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and clamp loader configuration to safe bounds.
    
    Args:
        config: Loader configuration dict
        
    Returns:
        Validated configuration with values clamped to bounds
        
    Raises:
        ValueError: If required keys are missing
    """
    from src.utils.config_safety import validate_loader_limits
    
    # Start with a copy of the config
    validated_config = config.copy()
    
    # Clamp bounded values that config_safety doesn't handle
    for key, bounds in LOADER_CONFIG_BOUNDS.items():
        if key in validated_config:
            value = validated_config[key]
            if value is not None and isinstance(value, (int, float)):
                validated_config[key] = max(bounds["min"], min(bounds["max"], int(value)))
    
    # Extract loader-specific limits for validation by config_safety
    limits = {
        "max_file_size_mb": validated_config.get("max_file_size_mb"),
        "max_total_size_mb": validated_config.get("max_total_size_mb"),
        "max_memory_mb": validated_config.get("max_memory_mb"),
    }
    
    # Validate and clamp using config_safety (only updates size limits we extracted)
    validated_limits = validate_loader_limits(limits)
    
    # Update only the size-related limits (don't overwrite other fields from validate_loader_limits)
    validated_config["max_file_size_mb"] = validated_limits["max_file_size_mb"]
    validated_config["max_total_size_mb"] = validated_limits["max_total_size_mb"]
    validated_config["max_memory_mb"] = validated_limits["max_memory_mb"]
    # Also update memory_threshold_mb if it was adjusted by config_safety
    if "memory_threshold_mb" in validated_limits:
        validated_config["memory_threshold_mb"] = validated_limits["memory_threshold_mb"]
    
    return validated_config


def merge_loader_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple loader configs, with later configs taking precedence.
    
    Args:
        *configs: Variable number of config dicts to merge
        
    Returns:
        Merged configuration dict
        
    Example:
        >>> base = DEFAULT_LOADER_CONFIG
        >>> custom = {"max_file_size_mb": 20480, "io_workers": 8}
        >>> merged = merge_loader_configs(base, custom)
    """
    result = {}
    for config in configs:
        if config:
            result.update(config)
    return result
