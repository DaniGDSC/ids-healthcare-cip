"""
Test loader schema configuration and deduplication.
"""
import pytest
from config.loader_schema import (
    DEFAULT_LOADER_CONFIG,
    LOADER_CONFIG_BOUNDS,
    get_loader_config,
    validate_loader_config,
    merge_loader_configs,
)


class TestLoaderSchema:
    """Test shared loader configuration schema."""
    
    def test_default_loader_config_has_required_keys(self):
        """Verify default config contains all expected keys."""
        required_keys = {
            "file_pattern",
            "encoding",
            "max_file_size_mb",
            "max_total_size_mb",
            "max_memory_mb",
            "memory_threshold_mb",
            "io_workers",
            "io_retries",
            "io_retry_delay_seconds",
            "read_timeout_seconds",
            "skip_on_error",
            "quarantine_dir",
        }
        
        for key in required_keys:
            assert key in DEFAULT_LOADER_CONFIG, f"Missing required key: {key}"
    
    def test_get_loader_config_with_no_overrides(self):
        """Verify get_loader_config returns defaults when no overrides."""
        config = get_loader_config()
        assert config == DEFAULT_LOADER_CONFIG
        assert config["max_file_size_mb"] == 10240
        assert config["io_workers"] == 4
    
    def test_get_loader_config_with_overrides(self):
        """Verify overrides are applied correctly."""
        overrides = {
            "max_file_size_mb": 20480,
            "io_workers": 8,
            "custom_key": "custom_value"
        }
        
        config = get_loader_config(overrides)
        assert config["max_file_size_mb"] == 20480
        assert config["io_workers"] == 8
        assert config["custom_key"] == "custom_value"
        # Default values still present
        assert config["encoding"] == "utf-8"
    
    def test_validate_loader_config_clamps_to_bounds(self):
        """Verify validation clamps values to defined bounds."""
        config = {
            "max_file_size_mb": 500_000,  # Exceeds max (204800)
            "max_total_size_mb": 50,      # Below min (100)
            "max_memory_mb": 16384,       # Within bounds
            "io_workers": 100,            # Exceeds max (32)
        }
        
        validated = validate_loader_config(config)
        
        # Check clamping
        assert validated["max_file_size_mb"] == 204800  # Clamped to max
        assert validated["max_total_size_mb"] == 100    # Clamped to min
        assert validated["max_memory_mb"] == 16384      # Unchanged
        assert validated["io_workers"] == 32            # Clamped to max
    
    def test_validate_loader_config_handles_none_values(self):
        """Verify validation uses defaults for None values."""
        config = {
            "max_file_size_mb": None,  # Should use default
            "max_total_size_mb": 50000,
            "max_memory_mb": 32768,
        }
        
        validated = validate_loader_config(config)
        
        # None gets replaced with default by config_safety.validate_loader_limits
        assert validated["max_file_size_mb"] == 10240  # Default from DEFAULT_LIMITS
        assert validated["max_total_size_mb"] == 50000
    
    def test_merge_loader_configs_precedence(self):
        """Verify later configs override earlier ones."""
        base = {"max_file_size_mb": 10000, "io_workers": 4, "encoding": "utf-8"}
        override1 = {"max_file_size_mb": 20000, "skip_on_error": True}
        override2 = {"max_file_size_mb": 30000, "io_workers": 8}
        
        merged = merge_loader_configs(base, override1, override2)
        
        assert merged["max_file_size_mb"] == 30000  # Last override wins
        assert merged["io_workers"] == 8            # From override2
        assert merged["skip_on_error"] is True      # From override1
        assert merged["encoding"] == "utf-8"        # From base
    
    def test_merge_loader_configs_handles_none(self):
        """Verify merge skips None configs."""
        base = {"max_file_size_mb": 10000}
        
        merged = merge_loader_configs(base, None, {"io_workers": 8})
        
        assert merged["max_file_size_mb"] == 10000
        assert merged["io_workers"] == 8
    
    def test_loader_config_bounds_consistency(self):
        """Verify bounds match config_safety bounds."""
        # Import config_safety bounds for comparison
        from src.utils.config_safety import BOUNDS
        
        # Check critical bounds match
        assert LOADER_CONFIG_BOUNDS["max_file_size_mb"]["min"] == BOUNDS["max_file_size_mb"][0]
        assert LOADER_CONFIG_BOUNDS["max_file_size_mb"]["max"] == BOUNDS["max_file_size_mb"][1]
        
        assert LOADER_CONFIG_BOUNDS["max_total_size_mb"]["min"] == BOUNDS["max_total_size_mb"][0]
        assert LOADER_CONFIG_BOUNDS["max_total_size_mb"]["max"] == BOUNDS["max_total_size_mb"][1]
        
        assert LOADER_CONFIG_BOUNDS["max_memory_mb"]["min"] == BOUNDS["max_memory_mb"][0]
        assert LOADER_CONFIG_BOUNDS["max_memory_mb"]["max"] == BOUNDS["max_memory_mb"][1]


