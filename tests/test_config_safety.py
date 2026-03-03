import pytest

from src.utils.config_safety import validate_orchestration_config, validate_loader_limits


def test_orchestration_timeouts_clamped_to_bounds():
    cfg = {
        "phases": [
            {"name": "p1", "timeout_seconds": 10, "retry_delay_seconds": 5},
        ]
    }

    validated = validate_orchestration_config(cfg)
    phase = validated["phases"][0]

    assert phase["timeout_seconds"] == 60  # min bound applied
    assert phase["retry_delay_seconds"] == 5  # within bounds


def test_orchestration_custom_bounds_cap_values():
    cfg = {
        "config_safety": {
            "bounds": {
                "timeout_seconds": {"min": 30, "max": 100},
            },
            "defaults": {
                "phase_timeout_seconds": 50,
            },
        },
        "phases": [
            {"name": "p1", "timeout_seconds": 200},
        ],
    }

    validated = validate_orchestration_config(cfg)
    phase = validated["phases"][0]

    assert phase["timeout_seconds"] == 100  # capped by custom max
    assert validated["config_safety"]["bounds"]["timeout_seconds"]["max"] == 100


def test_loader_limits_clamp_and_defaults():
    loader_cfg = {
        "max_file_size_mb": 500_000,  # way above bound
        "io_retries": 0,              # below min
        "io_retry_delay_seconds": -1, # below min
        "read_timeout_seconds": 999,  # above max
    }
    safety_cfg = {
        "defaults": {
            "max_file_size_mb": 1_000,
            "io_retries": 2,
            "io_retry_delay_seconds": 2,
            "read_timeout_seconds": 30,
        }
    }

    sanitized = validate_loader_limits(loader_cfg, safety_cfg)

    assert sanitized["max_file_size_mb"] == 204_800  # upper bound applied
    assert sanitized["io_retries"] == 1  # min bound
    assert sanitized["io_retry_delay_seconds"] == 0  # min bound
    assert sanitized["read_timeout_seconds"] == 600  # upper bound
