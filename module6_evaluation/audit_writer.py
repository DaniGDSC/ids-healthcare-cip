"""Audit-trail writers — plain JSONL + hardened Module 5 logger.

Y3 + Y5 + Y8 follow-up: the hardened ECDSA-signed logger is lazily
constructed via :func:`get_hardened_audit` so importing the module
doesn't bootstrap signing keys at import time. The plain ``AuditTrailWriter``
remains for offline study mode but writes the same payload through the
hardened logger when available.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = PROJECT_ROOT / "results/reports"

_hardened_audit: Any = None  # lazy init


def get_hardened_audit():
    """Return (and lazily construct) the Module 5 hardened audit logger."""
    global _hardened_audit
    if _hardened_audit is None:
        from module5_responses.module5_pipeline import (
            AuditLogger as HardenedAuditLogger,
        )
        EVAL_DIR.mkdir(parents=True, exist_ok=True)
        _hardened_audit = HardenedAuditLogger(EVAL_DIR / "audit_log.jsonl")
    return _hardened_audit


class AuditTrailWriter:
    """Plain JSONL audit writer kept for offline study mode.

    Reviewer-attributed events should additionally go through
    :func:`get_hardened_audit` for the signed chain.
    """

    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path else EVAL_DIR / "audit_trail.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event: dict) -> None:
        event = {
            "timestamp_iso": datetime.now(timezone.utc).isoformat(),
            "epoch_sec": time.time(),
            **event,
        }
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")


def audit_log(
    event_type: str,
    *,
    participant_id: str | None = None,
    role: str | None = None,
    action: str | None = None,
    sign: bool = True,
    **kwargs,
) -> None:
    """Append an event to both the plain JSONL writer and the hardened chain.

    When ``sign=True`` (the default), the same payload is bound through the
    hardened logger with reviewer attribution so participant decisions are
    cryptographically attestable.
    """
    payload = {"event_type": event_type, **kwargs}
    if participant_id is not None:
        payload["participant_id"] = participant_id
    if role is not None:
        payload["role"] = role
    if action is not None:
        payload["action"] = action

    AuditTrailWriter().write(payload)

    if not sign:
        return
    try:
        get_hardened_audit().log(
            payload,
            reviewer_id=participant_id,
            reviewer_role=role,
            review_action=action,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "audit_log: hardened sign failed (%s) — plain JSONL still wrote.", exc,
        )


__all__ = [
    "AuditTrailWriter",
    "audit_log",
    "get_hardened_audit",
    "EVAL_DIR",
    "PROJECT_ROOT",
]
