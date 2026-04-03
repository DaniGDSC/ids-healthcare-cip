"""State change logger with circular buffer storage.

Stores ONLY state changes, not every heartbeat (~99% storage reduction).
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Deque, List

from .state_machine import StateChangeEvent

logger = logging.getLogger(__name__)


class StateChangeLogger:
    """Logs engine state transitions using a circular buffer.

    Storage optimization: only state CHANGES are persisted,
    not every heartbeat tick (~99% reduction vs full logging).
    """

    def __init__(self, buffer_size: int = 1000) -> None:
        self._buffer: Deque[StateChangeEvent] = deque(maxlen=buffer_size)

    def log(self, event: StateChangeEvent) -> None:
        """Record a state change event."""
        self._buffer.append(event)
        logger.info(
            "State: %s %s -> %s (%s)",
            event.engine_id,
            event.old_state,
            event.new_state,
            event.reason,
        )

    @property
    def all_events(self) -> List[StateChangeEvent]:
        """Return all logged state change events."""
        return list(self._buffer)

    def get_events_for(self, engine_id: str) -> List[StateChangeEvent]:
        """Return events for a specific engine."""
        return [e for e in self._buffer if e.engine_id == engine_id]
