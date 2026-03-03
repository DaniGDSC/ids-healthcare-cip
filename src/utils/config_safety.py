"""Configuration safety helpers for enforcing bounds and defaults."""

from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

# Default thresholds (safe fallbacks)
DEFAULT_LIMITS: Dict[str, int] = {
    "phase_timeout_seconds": 3600,
    "retry_delay_seconds": 60,
    "read_timeout_seconds": 30,
    "max_file_size_mb": 10_240,      # 10 GB per file
    "max_total_size_mb": 51_200,     # 50 GB total dataset budget
    "max_memory_mb": 32_768,         # 32 GB in-memory cap
    "memory_threshold_mb": 2_000,    # Trigger chunking above 2 GB
    "io_retries": 2,
    "io_retry_delay_seconds": 2,
}

# Hard bounds to prevent unbounded configs
BOUNDS: Dict[str, Tuple[int, int]] = {
    "timeout_seconds": (60, 86_400),          # 1 minute to 24 hours
    "retry_delay_seconds": (0, 3_600),
    "read_timeout_seconds": (0, 600),         # up to 10 minutes
    "max_file_size_mb": (10, 204_800),        # 10 MB to 200 GB
    "max_total_size_mb": (100, 409_600),      # 0.1 GB to 400 GB
    "max_memory_mb": (512, 262_144),          # 0.5 GB to 256 GB
    "memory_threshold_mb": (128, 131_072),
    "io_retries": (1, 10),
    "io_retry_delay_seconds": (0, 600),
}

_logger = logging.getLogger(__name__)


def _resolved_bounds(name: str, overrides: Dict[str, Dict[str, int]] | None) -> Tuple[int, int]:
    base_min, base_max = BOUNDS[name]
    if not overrides:
        return base_min, base_max
    custom = overrides.get(name, {})
    return max(base_min, custom.get("min", base_min)), min(base_max, custom.get("max", base_max))


def _clamp(name: str, value: Any, bounds: Tuple[int, int], default: int) -> int:
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        _logger.warning(f"Invalid value for {name} ({value!r}), using default {default}")
        return default
    low, high = bounds
    if numeric < low:
        _logger.warning(f"{name}={numeric} below minimum {low}, clamping to {low}")
        return low
    if numeric > high:
        _logger.warning(f"{name}={numeric} above maximum {high}, clamping to {high}")
        return high
    return numeric


def validate_orchestration_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate orchestration config with safe defaults and bounds."""
    safety_cfg = config.get("config_safety", {})
    defaults = safety_cfg.get("defaults", {})
    bounds_override = safety_cfg.get("bounds", {})

    timeout_bounds = _resolved_bounds("timeout_seconds", bounds_override)
    retry_delay_bounds = _resolved_bounds("retry_delay_seconds", bounds_override)

    phase_timeout_default = _clamp(
        "timeout_seconds",
        defaults.get("phase_timeout_seconds", DEFAULT_LIMITS["phase_timeout_seconds"]),
        timeout_bounds,
        DEFAULT_LIMITS["phase_timeout_seconds"],
    )
    retry_delay_default = _clamp(
        "retry_delay_seconds",
        defaults.get("retry_delay_seconds", DEFAULT_LIMITS["retry_delay_seconds"]),
        retry_delay_bounds,
        DEFAULT_LIMITS["retry_delay_seconds"],
    )

    # Data loader defaults (propagate to phases that may use them)
    read_timeout_bounds = _resolved_bounds("read_timeout_seconds", bounds_override)
    max_file_bounds = _resolved_bounds("max_file_size_mb", bounds_override)
    max_total_bounds = _resolved_bounds("max_total_size_mb", bounds_override)
    max_memory_bounds = _resolved_bounds("max_memory_mb", bounds_override)
    memory_threshold_bounds = _resolved_bounds("memory_threshold_mb", bounds_override)

    read_timeout_default = _clamp(
        "read_timeout_seconds",
        defaults.get("read_timeout_seconds", DEFAULT_LIMITS["read_timeout_seconds"]),
        read_timeout_bounds,
        DEFAULT_LIMITS["read_timeout_seconds"],
    )
    max_file_default = _clamp(
        "max_file_size_mb",
        defaults.get("max_file_size_mb", DEFAULT_LIMITS["max_file_size_mb"]),
        max_file_bounds,
        DEFAULT_LIMITS["max_file_size_mb"],
    )
    max_total_default = _clamp(
        "max_total_size_mb",
        defaults.get("max_total_size_mb", DEFAULT_LIMITS["max_total_size_mb"]),
        max_total_bounds,
        DEFAULT_LIMITS["max_total_size_mb"],
    )
    max_memory_default = _clamp(
        "max_memory_mb",
        defaults.get("max_memory_mb", DEFAULT_LIMITS["max_memory_mb"]),
        max_memory_bounds,
        DEFAULT_LIMITS["max_memory_mb"],
    )
    memory_threshold_default = _clamp(
        "memory_threshold_mb",
        defaults.get("memory_threshold_mb", DEFAULT_LIMITS["memory_threshold_mb"]),
        memory_threshold_bounds,
        DEFAULT_LIMITS["memory_threshold_mb"],
    )

    for phase in config.get("phases", []):
        phase["timeout_seconds"] = _clamp(
            "timeout_seconds",
            phase.get("timeout_seconds", phase_timeout_default),
            timeout_bounds,
            phase_timeout_default,
        )
        phase["retry_delay_seconds"] = _clamp(
            "retry_delay_seconds",
            phase.get("retry_delay_seconds", retry_delay_default),
            retry_delay_bounds,
            retry_delay_default,
        )

    config.setdefault("config_safety", {})
    config["config_safety"]["defaults"] = {
        "phase_timeout_seconds": phase_timeout_default,
        "retry_delay_seconds": retry_delay_default,
        "read_timeout_seconds": read_timeout_default,
        "max_file_size_mb": max_file_default,
        "max_total_size_mb": max_total_default,
        "max_memory_mb": max_memory_default,
        "memory_threshold_mb": memory_threshold_default,
    }
    config["config_safety"]["bounds"] = {
        "timeout_seconds": {"min": timeout_bounds[0], "max": timeout_bounds[1]},
        "retry_delay_seconds": {"min": retry_delay_bounds[0], "max": retry_delay_bounds[1]},
        "read_timeout_seconds": {"min": read_timeout_bounds[0], "max": read_timeout_bounds[1]},
        "max_file_size_mb": {"min": max_file_bounds[0], "max": max_file_bounds[1]},
        "max_total_size_mb": {"min": max_total_bounds[0], "max": max_total_bounds[1]},
        "max_memory_mb": {"min": max_memory_bounds[0], "max": max_memory_bounds[1]},
        "memory_threshold_mb": {"min": memory_threshold_bounds[0], "max": memory_threshold_bounds[1]},
    }

    return config


def validate_loader_limits(loader_cfg: Dict[str, Any], safety_cfg: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Validate data loader-related thresholds and inject defaults."""
    defaults = (safety_cfg or {}).get("defaults", {})
    bounds_override = (safety_cfg or {}).get("bounds", {})

    read_timeout_bounds = _resolved_bounds("read_timeout_seconds", bounds_override)
    max_file_bounds = _resolved_bounds("max_file_size_mb", bounds_override)
    max_total_bounds = _resolved_bounds("max_total_size_mb", bounds_override)
    max_memory_bounds = _resolved_bounds("max_memory_mb", bounds_override)
    memory_threshold_bounds = _resolved_bounds("memory_threshold_mb", bounds_override)
    io_retry_bounds = _resolved_bounds("io_retry_delay_seconds", bounds_override)
    io_retries_bounds = _resolved_bounds("io_retries", bounds_override)

    sanitized = {
        "io_retries": _clamp(
            "io_retries",
            loader_cfg.get("io_retries", defaults.get("io_retries", DEFAULT_LIMITS["io_retries"])),
            io_retries_bounds,
            DEFAULT_LIMITS["io_retries"],
        ),
        "io_retry_delay_seconds": _clamp(
            "io_retry_delay_seconds",
            loader_cfg.get(
                "io_retry_delay_seconds",
                defaults.get("io_retry_delay_seconds", DEFAULT_LIMITS["io_retry_delay_seconds"]),
            ),
            io_retry_bounds,
            DEFAULT_LIMITS["io_retry_delay_seconds"],
        ),
        "read_timeout_seconds": _clamp(
            "read_timeout_seconds",
            loader_cfg.get(
                "read_timeout_seconds",
                defaults.get("read_timeout_seconds", DEFAULT_LIMITS["read_timeout_seconds"]),
            ),
            read_timeout_bounds,
            DEFAULT_LIMITS["read_timeout_seconds"],
        ),
        "skip_on_error": bool(loader_cfg.get("skip_on_error", False)),
        "quarantine_dir": loader_cfg.get("quarantine_dir"),
        "use_smart_loading": bool(loader_cfg.get("use_smart_loading", True)),
        "io_workers": int(loader_cfg.get("io_workers", 1) or 1),
    }

    sanitized["max_file_size_mb"] = _clamp(
        "max_file_size_mb",
        loader_cfg.get("max_file_size_mb", defaults.get("max_file_size_mb", DEFAULT_LIMITS["max_file_size_mb"])),
        max_file_bounds,
        DEFAULT_LIMITS["max_file_size_mb"],
    )
    sanitized["max_total_size_mb"] = _clamp(
        "max_total_size_mb",
        loader_cfg.get(
            "max_total_size_mb",
            defaults.get("max_total_size_mb", DEFAULT_LIMITS["max_total_size_mb"]),
        ),
        max_total_bounds,
        DEFAULT_LIMITS["max_total_size_mb"],
    )
    sanitized["max_memory_mb"] = _clamp(
        "max_memory_mb",
        loader_cfg.get("max_memory_mb", defaults.get("max_memory_mb", DEFAULT_LIMITS["max_memory_mb"])),
        max_memory_bounds,
        DEFAULT_LIMITS["max_memory_mb"],
    )
    sanitized["memory_threshold_mb"] = _clamp(
        "memory_threshold_mb",
        loader_cfg.get(
            "memory_threshold_mb",
            defaults.get("memory_threshold_mb", DEFAULT_LIMITS["memory_threshold_mb"]),
        ),
        memory_threshold_bounds,
        DEFAULT_LIMITS["memory_threshold_mb"],
    )

    # Ensure chunking threshold does not exceed total budget
    if sanitized["memory_threshold_mb"] > sanitized["max_total_size_mb"]:
        _logger.warning(
            "memory_threshold_mb exceeds max_total_size_mb; lowering threshold to total size budget"
        )
        sanitized["memory_threshold_mb"] = sanitized["max_total_size_mb"]

    return sanitized
