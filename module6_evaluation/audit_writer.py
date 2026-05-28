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
        # Tier 2 F7: chmod 0640 on first write. open(..., "a") creates
        # the file with the process umask (typically 0022 → 0644); we
        # force the audit-trail JSONL to be group-readable only.
        import os as _os
        is_new = not self.path.exists() or self.path.stat().st_size == 0
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")
        if is_new:
            try:
                _os.chmod(self.path, 0o640)
            except OSError as exc:
                logger.warning(
                    "AuditTrailWriter: chmod 0640 on %s failed: %s",
                    self.path, exc,
                )


class HardenedAuditUnavailable(RuntimeError):
    """Raised by ``audit_log`` when ``sign=True`` was requested but the
    hardened signed chain could not record the event. Tier 2 F4: refuses
    to silently degrade to plain JSONL because that is indistinguishable
    from a successful signed write at the rendering layer.
    """


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

    When ``sign=True`` (the default), the same payload is bound through
    the hardened logger with reviewer attribution so participant
    decisions are cryptographically attestable. Tier 2 F4: a hardened-
    write failure raises :class:`HardenedAuditUnavailable` instead of
    silently degrading to plain JSONL only. Callers can decide whether
    to surface the failure to the operator or fall back to a degraded
    workflow (see ``_capture_dashboard_action`` in the module 6 app).
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
        logger.error(
            "audit_log: hardened sign failed (%s). Plain JSONL wrote, but "
            "the signed chain did NOT record this event. Raising so the "
            "caller can surface the failure.", exc,
        )
        raise HardenedAuditUnavailable(
            f"hardened audit sign failed: {exc}"
        ) from exc


__all__ = [
    "AuditTrailWriter",
    "audit_log",
    "get_hardened_audit",
    "HardenedAuditUnavailable",
    "EVAL_DIR",
    "PROJECT_ROOT",
]
