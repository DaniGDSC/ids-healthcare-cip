"""Alert fatigue manager — suppresses, aggregates, and rate-limits alerts.

In hospital environments, excessive alarms cause clinicians to ignore
real threats. This manager sits between risk scoring and alert dispatch,
reducing alert volume while preserving escalations.

Three mechanisms:
  1. Cooldown — suppress repeat alerts at same severity for a device
  2. Aggregation — collapse consecutive same-level alerts into one
  3. Rate limiting — cap alerts per device per time window

Escalations (severity increase) ALWAYS bypass suppression.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional

from .base import BaseDetector

logger = logging.getLogger(__name__)


class AlertRecord:
    """Tracks alert state for a single device."""

    __slots__ = ("device_id", "last_severity", "consecutive_count",
                 "suppressed_count", "alerts_in_window", "total_emitted")

    def __init__(self, device_id: str) -> None:
        self.device_id = device_id
        self.last_severity: Optional[int] = None
        self.consecutive_count: int = 0
        self.suppressed_count: int = 0
        self.alerts_in_window: int = 0
        self.total_emitted: int = 0


class AlertDecision:
    """Decision for a single sample's alert."""

    __slots__ = ("emit", "reason", "aggregated_count")

    def __init__(self, emit: bool, reason: str, aggregated_count: int = 0) -> None:
        self.emit = emit
        self.reason = reason
        self.aggregated_count = aggregated_count


class AlertFatigueManager(BaseDetector):
    """Reduce alert volume while preserving clinical safety.

    Rules (in order):
      1. ROUTINE (severity=1) alerts are NEVER emitted (log only).
      2. Escalations (severity increased from previous) ALWAYS emit.
      3. Consecutive same-severity alerts are aggregated — only the
         first and every Nth (aggregation_window) emit; intermediate
         ones are suppressed. When emitted, includes the count of
         suppressed alerts since last emission.
      4. Per-device rate limit: max alerts_per_window alerts per
         rate_window_size samples. Exceeded → suppress until window
         resets.

    De-escalations (severity decreased) also emit to signal recovery.

    Args:
        aggregation_window: Emit every Nth consecutive same-level alert.
        alerts_per_window: Max alerts per device per rate window.
        rate_window_size: Number of samples in one rate-limit window.
    """

    def __init__(
        self,
        aggregation_window: int = 10,
        alerts_per_window: int = 5,
        rate_window_size: int = 100,
    ) -> None:
        self._agg_window = aggregation_window
        self._max_alerts = alerts_per_window
        self._rate_window = rate_window_size
        self._devices: Dict[str, AlertRecord] = defaultdict(lambda: AlertRecord("unknown"))
        self._sample_counter = 0
        self._stats = {
            "total_samples": 0,
            "total_emitted": 0,
            "total_suppressed": 0,
            "suppressed_routine": 0,
            "suppressed_aggregation": 0,
            "suppressed_rate_limit": 0,
            "escalations": 0,
            "de_escalations": 0,
        }

    def process(
        self,
        risk_results: List[Dict[str, Any]],
        device_id: str = "generic_iomt_sensor",
    ) -> List[Dict[str, Any]]:
        """Process risk results and apply fatigue mitigation.

        Adds to each result dict:
          - alert_emit: bool (True = send alert, False = suppressed)
          - alert_reason: str (why emitted or suppressed)
          - alert_aggregated_count: int (suppressed alerts since last emission)

        Args:
            risk_results: List of per-sample risk dicts from RiskScorer.
            device_id: Device identifier for per-device tracking.

        Returns:
            Same list with alert fields added.
        """
        record = self._devices[device_id]
        record.device_id = device_id

        for result in risk_results:
            severity = result.get("clinical_severity", 1)
            decision = self._decide(severity, record)

            result["alert_emit"] = decision.emit
            result["alert_reason"] = decision.reason
            result["alert_aggregated_count"] = decision.aggregated_count

            self._stats["total_samples"] += 1
            if decision.emit:
                self._stats["total_emitted"] += 1
                record.total_emitted += 1
            else:
                self._stats["total_suppressed"] += 1

            # Rate window reset
            self._sample_counter += 1
            if self._sample_counter >= self._rate_window:
                self._sample_counter = 0
                for rec in self._devices.values():
                    rec.alerts_in_window = 0

        return risk_results

    def _decide(self, severity: int, record: AlertRecord) -> AlertDecision:
        """Decide whether to emit or suppress an alert."""

        # Rule 1: ROUTINE (severity=1) → always suppress
        if severity <= 1:
            record.last_severity = severity
            record.consecutive_count = 0
            self._stats["suppressed_routine"] += 1
            return AlertDecision(emit=False, reason="routine_suppressed")

        # Rule 2: Escalation → always emit
        if record.last_severity is not None and severity > record.last_severity:
            aggregated = record.suppressed_count
            record.last_severity = severity
            record.consecutive_count = 1
            record.suppressed_count = 0
            record.alerts_in_window += 1
            self._stats["escalations"] += 1
            return AlertDecision(emit=True, reason="escalation", aggregated_count=aggregated)

        # De-escalation → emit to signal recovery
        if record.last_severity is not None and severity < record.last_severity and severity > 1:
            aggregated = record.suppressed_count
            record.last_severity = severity
            record.consecutive_count = 1
            record.suppressed_count = 0
            record.alerts_in_window += 1
            self._stats["de_escalations"] += 1
            return AlertDecision(emit=True, reason="de_escalation", aggregated_count=aggregated)

        # Rule 4: Rate limit check
        if record.alerts_in_window >= self._max_alerts:
            record.consecutive_count += 1
            record.suppressed_count += 1
            record.last_severity = severity
            self._stats["suppressed_rate_limit"] += 1
            return AlertDecision(emit=False, reason="rate_limited")

        # Rule 3: Aggregation — same severity as last
        record.consecutive_count += 1
        record.last_severity = severity

        if record.consecutive_count == 1:
            # First alert at this level → emit
            record.suppressed_count = 0
            record.alerts_in_window += 1
            return AlertDecision(emit=True, reason="first_at_level")

        if record.consecutive_count % self._agg_window == 0:
            # Every Nth consecutive → emit with aggregated count
            aggregated = record.suppressed_count
            record.suppressed_count = 0
            record.alerts_in_window += 1
            return AlertDecision(emit=True, reason="aggregation_summary", aggregated_count=aggregated)

        # Suppress intermediate
        record.suppressed_count += 1
        self._stats["suppressed_aggregation"] += 1
        return AlertDecision(emit=False, reason="aggregation_suppressed")

    def get_summary(self) -> Dict[str, Any]:
        """Return fatigue mitigation summary statistics."""
        total = self._stats["total_samples"]
        emitted = self._stats["total_emitted"]
        suppressed = self._stats["total_suppressed"]
        return {
            "total_samples": total,
            "alerts_emitted": emitted,
            "alerts_suppressed": suppressed,
            "suppression_rate": round(suppressed / max(total, 1), 4),
            "breakdown": {
                "suppressed_routine": self._stats["suppressed_routine"],
                "suppressed_aggregation": self._stats["suppressed_aggregation"],
                "suppressed_rate_limit": self._stats["suppressed_rate_limit"],
            },
            "escalations_emitted": self._stats["escalations"],
            "de_escalations_emitted": self._stats["de_escalations"],
            "config": self.get_config(),
        }

    def get_config(self) -> Dict[str, Any]:
        return {
            "aggregation_window": self._agg_window,
            "alerts_per_window": self._max_alerts,
            "rate_window_size": self._rate_window,
        }
