"""Load and validate production configuration.

Provides a global singleton config loaded from production.yaml.
Supports dot-notation access for nested keys with fallback defaults.

Usage::

    from config.production_loader import cfg

    window = cfg("streaming.window_size", 20)
    fpr = cfg("calibration.target_fpr", 0.10)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)

_CONFIG: Dict[str, Any] = {}
_LOADED = False


def _load(path: Optional[Path] = None) -> Dict[str, Any]:
    """Load config from YAML file. Called once on first access."""
    global _CONFIG, _LOADED
    if _LOADED:
        return _CONFIG

    if path is None:
        path = Path(__file__).parent / "production.yaml"

    if path.exists():
        with open(path) as f:
            _CONFIG = yaml.safe_load(f) or {}
        logger.info("Production config loaded: %s (%d keys)", path, len(_CONFIG))
    else:
        logger.warning("Production config not found: %s — using defaults", path)
        _CONFIG = {}

    _LOADED = True
    return _CONFIG


def cfg(key: str, default: Any = None) -> Any:
    """Get config value by dot-notation key with fallback default.

    Args:
        key: Dot-separated path, e.g. 'calibration.target_fpr'.
        default: Value returned if key not found.

    Returns:
        Config value or default.
    """
    _load()
    parts = key.split(".")
    val: Any = _CONFIG
    for p in parts:
        if isinstance(val, dict):
            val = val.get(p)
        else:
            return default
        if val is None:
            return default
    return val


def reload(path: Optional[Path] = None) -> Dict[str, Any]:
    """Force reload config (for testing or hot-reload)."""
    global _LOADED
    _LOADED = False
    return _load(path)
