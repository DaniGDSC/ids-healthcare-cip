"""Shared pytest fixtures.

Closes GAP-A13 — `tests/negative_tests.py` was previously runnable only
through `run_tests.py` because its functions take `system_logs`,
`system_actions`, and `outputs` parameters that pytest could not resolve.

This module provides those fixtures so `python -m pytest tests/` can run
the negative suite end-to-end. The default fixtures are empty lists; that
is the correct *positive verification* of the invariants — when there are
zero events, there are zero violations of any forbidden-pattern rule.

Tests that want to assert violations are detected on real data should
parametrise these fixtures with their own payloads.
"""
from __future__ import annotations

from typing import Any, List

import pytest


@pytest.fixture
def system_logs() -> List[dict[str, Any]]:
    """Empty default — no logs ⇒ no discovery violations."""
    return []


@pytest.fixture
def system_actions() -> List[dict[str, Any]]:
    """Empty default — no actions ⇒ no automated-blocking violations."""
    return []


@pytest.fixture
def outputs() -> List[dict[str, Any]]:
    """Empty default — no MVE outputs ⇒ no text-content violations."""
    return []
