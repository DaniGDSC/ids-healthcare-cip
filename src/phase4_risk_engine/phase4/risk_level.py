"""Risk level enumeration for Phase 4 risk scoring.

Replaces string constants with a type-safe enum.
"""

from __future__ import annotations

from enum import Enum


class RiskLevel(str, Enum):
    """Five-level risk classification for IoMT traffic.

    Levels are ordered by increasing severity::

        NORMAL < LOW < MEDIUM < HIGH < CRITICAL
    """

    NORMAL = "NORMAL"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"
