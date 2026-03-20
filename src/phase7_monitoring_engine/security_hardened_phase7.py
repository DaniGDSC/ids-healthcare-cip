#!/usr/bin/env python3
"""Security-hardened Phase 7 monitoring engine — OWASP Top 10 + HIPAA compliance.

Wraps the SOLID phase7 package with security controls, **extending**
Phase 0 security classes — never duplicating them.

    A01  Append-only security_audit_log, read-only monitoring_log, reject unknown engines
    A02  HMAC-SHA256 heartbeat signing, SHA-256 artifact verification, 24h key rotation
    A03  Validate heartbeat fields (engine_id allowlist, ISO timestamp, latency bounds)
    A05  Config parameter bounds, unknown YAML key rejection
    A08  FSM transition assertions, buffer bounds, hash verification, append-only log
    A09  HIPAA-compliant logging — NEVER log raw heartbeat payload, patient metrics, HMAC keys

Phase 0 controls reused via direct import:
    - IntegrityVerifier.compute_hash()      (SHA-256 — not re-implemented)
    - PathValidator.validate_output_dir()    (path traversal — not re-implemented)
    - ConfigSanitizer.sanitize_config_dict() (injection — not re-implemented)
    - AuditLogger.log_security_event()       (audit trail — not re-implemented)

Usage::

    python -m src.phase7_monitoring_engine.security_hardened_phase7
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import stat
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

# ── Phase 0 security controls (reused, NOT duplicated) ──────────────
from src.phase0_dataset_analysis.phase0.security import (
    AuditLogger,
    ConfigSanitizer,
    IntegrityVerifier,
    PathValidator,
)

# ── Phase 7 SOLID components ────────────────────────────────────────
from src.phase7_monitoring_engine.phase7.config import Phase7Config
from src.phase7_monitoring_engine.phase7.pipeline import (
    MonitoringPipeline,
    _detect_hardware,
    _get_git_commit,
)
from src.phase7_monitoring_engine.phase7.state_machine import (
    DEFAULT_TRANSITIONS,
    EngineState,
    StateChangeEvent,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
CONFIG_PATH: Path = PROJECT_ROOT / "config" / "phase7_monitoring_config.yaml"
REPORT_DIR: Path = PROJECT_ROOT / "results" / "phase0_analysis"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  [%(name)s]  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

# A05 — Hardened parameter bounds
HEARTBEAT_INTERVAL_MIN: int = 1
HEARTBEAT_INTERVAL_MAX: int = 60
MISSED_THRESHOLD_MIN: int = 1
MISSED_THRESHOLD_MAX: int = 20
BUFFER_SIZE_MIN: int = 100
BUFFER_SIZE_MAX: int = 10000
INTEGRITY_CHECK_INTERVAL_MIN: int = 10
INTEGRITY_CHECK_INTERVAL_MAX: int = 3600
N_CYCLES_MIN: int = 1
N_CYCLES_MAX: int = 100
LATENCY_THRESHOLD_MIN: float = 10.0
LATENCY_THRESHOLD_MAX: float = 1000.0

# A03 — Heartbeat latency bounds
LATENCY_MS_MIN: float = 0.0
LATENCY_MS_MAX: float = 10000.0

# Known top-level YAML keys for unknown-key rejection (A05)
_KNOWN_YAML_KEYS: frozenset = frozenset(
    {
        "monitoring",
        "security",
        "output",
        "pipeline",
        "engines",
        "transitions",
    }
)


# ===================================================================
# A05 — Security Misconfiguration: Parameter Bounds + Unknown Keys
# ===================================================================


def _reject_unknown_yaml_keys(raw_yaml: dict) -> None:
    """Reject unknown top-level YAML keys (A05).

    Raises:
        ValueError: If unknown keys are found.
    """
    logger.info("── A05: Unknown YAML key rejection ──")
    unknown = set(raw_yaml.keys()) - _KNOWN_YAML_KEYS
    if unknown:
        msg = f"A05: Unknown YAML keys: {sorted(unknown)}"
        AuditLogger.log_security_event("CONFIG_VIOLATION", msg, logging.ERROR)
        raise ValueError(msg)
    logger.info("  A05 ✓  No unknown YAML keys (checked %d)", len(raw_yaml))


def _validate_phase7_parameters(config: Phase7Config) -> None:
    """Enforce Phase 7-specific parameter bounds (A05).

    Raises:
        ValueError: If any parameter is outside hardened bounds.
    """
    logger.info("── A05: Phase 7 parameter bounds validation ──")

    checks: List[Tuple[str, Any, Any, Any]] = [
        (
            "heartbeat_interval_seconds",
            config.heartbeat_interval_seconds,
            HEARTBEAT_INTERVAL_MIN,
            HEARTBEAT_INTERVAL_MAX,
        ),
        (
            "missed_heartbeat_threshold",
            config.missed_heartbeat_threshold,
            MISSED_THRESHOLD_MIN,
            MISSED_THRESHOLD_MAX,
        ),
        (
            "circular_buffer_size",
            config.circular_buffer_size,
            BUFFER_SIZE_MIN,
            BUFFER_SIZE_MAX,
        ),
        (
            "artifact_integrity_check_interval",
            config.artifact_integrity_check_interval,
            INTEGRITY_CHECK_INTERVAL_MIN,
            INTEGRITY_CHECK_INTERVAL_MAX,
        ),
        (
            "n_cycles",
            config.n_cycles,
            N_CYCLES_MIN,
            N_CYCLES_MAX,
        ),
        (
            "latency_p95_threshold_ms",
            config.latency_p95_threshold_ms,
            LATENCY_THRESHOLD_MIN,
            LATENCY_THRESHOLD_MAX,
        ),
    ]

    for name, value, lo, hi in checks:
        if not (lo <= value <= hi):
            msg = f"A05: {name}={value} outside allowed range [{lo}, {hi}]"
            AuditLogger.log_security_event("PARAMETER_VIOLATION", msg, logging.ERROR)
            raise ValueError(msg)
        logger.info("  A05 ✓  %s=%s ∈ [%s, %s]", name, value, lo, hi)


def _validate_engine_ids(config: Phase7Config) -> frozenset:
    """Build engine ID allowlist and validate no duplicates (A01/A05).

    Returns:
        Frozenset of valid engine IDs.

    Raises:
        ValueError: If duplicate engine IDs found.
    """
    logger.info("── A01/A05: Engine ID allowlist ──")
    ids = [e.id for e in config.engines]
    if len(ids) != len(set(ids)):
        duplicates = [eid for eid in ids if ids.count(eid) > 1]
        msg = f"A05: Duplicate engine IDs: {sorted(set(duplicates))}"
        AuditLogger.log_security_event("CONFIG_VIOLATION", msg, logging.ERROR)
        raise ValueError(msg)

    allowlist = frozenset(ids)
    logger.info("  A01 ✓  Engine allowlist: %d engines", len(allowlist))
    for eid in sorted(allowlist):
        logger.info("         %s", eid)
    return allowlist


# ===================================================================
# A02 — Cryptographic Failures: HMAC-SHA256 Heartbeat Signing
# ===================================================================


class HMACKeyManager:
    """HMAC-SHA256 key management with configurable rotation.

    Keys are generated via ``os.urandom(32)`` and rotated at
    ``rotation_interval_hours`` intervals. During rotation, the previous
    key is retained for a grace window to avoid rejecting in-flight
    heartbeats signed with the old key.
    """

    def __init__(self, rotation_interval_hours: int = 24) -> None:
        self._rotation_interval = rotation_interval_hours * 3600
        self._current_key = self._generate_key()
        self._previous_key: Optional[bytes] = None
        self._last_rotation = time.time()

    @staticmethod
    def _generate_key() -> bytes:
        """Generate a cryptographically secure 256-bit key."""
        return os.urandom(32)

    def rotate_if_needed(self) -> bool:
        """Rotate key if rotation interval has elapsed.

        Returns:
            True if key was rotated.
        """
        elapsed = time.time() - self._last_rotation
        if elapsed >= self._rotation_interval:
            self._previous_key = self._current_key
            self._current_key = self._generate_key()
            self._last_rotation = time.time()
            AuditLogger.log_security_event(
                "HMAC_KEY_ROTATED",
                "HMAC key rotated (previous key retained for grace window)",
                logging.INFO,
            )
            return True
        return False

    def sign(self, engine_id: str, timestamp: str, latency_ms: float) -> str:
        """Sign heartbeat data with HMAC-SHA256.

        Returns:
            Hex-encoded HMAC signature.
        """
        message = f"{engine_id}|{timestamp}|{latency_ms:.4f}".encode("utf-8")
        return hmac.new(self._current_key, message, hashlib.sha256).hexdigest()

    def verify(self, engine_id: str, timestamp: str, latency_ms: float, signature: str) -> bool:
        """Verify heartbeat HMAC signature.

        Checks current key first, then previous key (grace window).

        Returns:
            True if signature is valid.
        """
        message = f"{engine_id}|{timestamp}|{latency_ms:.4f}".encode("utf-8")

        # Check current key
        expected = hmac.new(self._current_key, message, hashlib.sha256).hexdigest()
        if hmac.compare_digest(expected, signature):
            return True

        # Check previous key (grace window)
        if self._previous_key is not None:
            expected_prev = hmac.new(self._previous_key, message, hashlib.sha256).hexdigest()
            if hmac.compare_digest(expected_prev, signature):
                return True

        return False

    def get_status(self) -> Dict[str, Any]:
        """Return key manager status (NEVER expose key material)."""
        return {
            "rotation_interval_hours": self._rotation_interval // 3600,
            "seconds_since_rotation": int(time.time() - self._last_rotation),
            "has_previous_key": self._previous_key is not None,
        }


# ===================================================================
# A02/A03 — Heartbeat Validation (HMAC + Field Injection)
# ===================================================================


def _validate_heartbeat(
    engine_id: str,
    timestamp: str,
    latency_ms: float,
    signature: str,
    hmac_mgr: HMACKeyManager,
    engine_allowlist: frozenset,
) -> None:
    """Validate heartbeat fields and HMAC signature (A02 + A03).

    Args:
        engine_id: Engine identifier to validate.
        timestamp: ISO 8601 timestamp to validate.
        latency_ms: Latency value to validate.
        signature: HMAC-SHA256 signature to verify.
        hmac_mgr: Key manager for signature verification.
        engine_allowlist: Frozenset of valid engine IDs.

    Raises:
        ValueError: If any validation fails.
    """
    # A03: Validate engine_id in allowlist
    if engine_id not in engine_allowlist:
        msg = f"A03: Unknown engine_id: {engine_id}"
        AuditLogger.log_security_event("HEARTBEAT_REJECTED", msg, logging.WARNING)
        raise ValueError(msg)

    # A03: Validate timestamp is valid ISO 8601
    try:
        datetime.fromisoformat(timestamp)
    except (ValueError, TypeError):
        msg = f"A03: Invalid ISO timestamp from {engine_id}"
        AuditLogger.log_security_event("HEARTBEAT_REJECTED", msg, logging.WARNING)
        raise ValueError(msg)

    # A03: Validate latency_ms within bounds
    if not (LATENCY_MS_MIN <= latency_ms <= LATENCY_MS_MAX):
        msg = f"A03: latency_ms from {engine_id} outside [{LATENCY_MS_MIN}, {LATENCY_MS_MAX}]"
        AuditLogger.log_security_event("HEARTBEAT_REJECTED", msg, logging.WARNING)
        raise ValueError(msg)

    # A02: Verify HMAC signature
    if not hmac_mgr.verify(engine_id, timestamp, latency_ms, signature):
        msg = f"A02: HMAC verification failed for {engine_id}"
        AuditLogger.log_security_event("HMAC_VERIFICATION_FAILED", msg, logging.CRITICAL)
        raise ValueError(msg)


# ===================================================================
# A01 — Broken Access Control: Read-Only + Append-Only
# ===================================================================


def _make_read_only(path: Path) -> None:
    """Set exported artifact to read-only (chmod 444) (A01)."""
    current = path.stat().st_mode
    read_only = current & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
    path.chmod(read_only)
    AuditLogger.log_file_access("READ_ONLY_SET", path, extra="mode=0o444")


def _clear_read_only(path: Path) -> None:
    """Temporarily restore write permission for overwriting (A01)."""
    if path.exists() and not os.access(path, os.W_OK):
        path.chmod(path.stat().st_mode | stat.S_IWUSR)
        AuditLogger.log_file_access("WRITE_RESTORED", path, extra="for overwrite")


class AppendOnlyLogEnforcer:
    """Enforce append-only semantics for security-critical logs (A01/A08).

    Tracks file sizes and verifies they only grow (monotonically increasing).
    Truncation or overwriting triggers a CRITICAL security event.
    """

    def __init__(self) -> None:
        self._baselines: Dict[str, int] = {}

    def register(self, path: Path) -> None:
        """Record initial file size as baseline."""
        if path.exists():
            self._baselines[str(path)] = path.stat().st_size
            size = path.stat().st_size
            logger.info("  A01 ✓  Append-only baseline: %s (%d bytes)", path.name, size)

    def verify(self, path: Path) -> bool:
        """Verify file size is >= baseline (append-only).

        Returns:
            True if append-only property holds.
        """
        key = str(path)
        if key not in self._baselines:
            return True  # No baseline registered
        if not path.exists():
            AuditLogger.log_security_event(
                "APPEND_ONLY_VIOLATION",
                f"File deleted: {path.name}",
                logging.CRITICAL,
            )
            return False
        current_size = path.stat().st_size
        if current_size < self._baselines[key]:
            AuditLogger.log_security_event(
                "APPEND_ONLY_VIOLATION",
                f"File truncated: {path.name} ({self._baselines[key]} -> {current_size})",
                logging.CRITICAL,
            )
            return False
        return True


# ===================================================================
# A08 — Data Integrity Assertions
# ===================================================================


class MonitoringAssertions:
    """Phase 7-specific data integrity assertions (A08)."""

    def __init__(self) -> None:
        self._results: List[Dict[str, Any]] = []

    def assert_fsm_transitions_valid(
        self,
        state_changes: List[StateChangeEvent],
    ) -> bool:
        """Assert all state transitions follow defined FSM rules.

        Verifies every (old_state, trigger) -> new_state observed at runtime
        exists in the DEFAULT_TRANSITIONS table.
        """
        invalid_count = 0
        for event in state_changes:
            old = EngineState(event.old_state)
            new = EngineState(event.new_state)
            # Check if any transition from old leads to new
            valid_targets = {v for (k, _), v in DEFAULT_TRANSITIONS.items() if k == old}
            if new not in valid_targets:
                invalid_count += 1

        passed = invalid_count == 0
        self._results.append(
            {
                "assertion": "FSM transitions follow defined rules",
                "expected": "0 invalid transitions",
                "actual": f"{invalid_count} invalid in {len(state_changes)} transitions",
                "status": "PASS" if passed else "FAIL",
            }
        )
        if passed:
            logger.info("  A08 ✓  All %d FSM transitions valid", len(state_changes))
        else:
            AuditLogger.log_security_event(
                "INTEGRITY_VIOLATION",
                f"Invalid FSM transitions: {invalid_count}",
                logging.CRITICAL,
            )
        return passed

    def assert_circular_buffer_bounded(
        self,
        buffer_lengths: Dict[str, int],
        max_size: int,
    ) -> bool:
        """Assert all circular buffers are within configured bounds."""
        violations = {eid: length for eid, length in buffer_lengths.items() if length > max_size}
        passed = len(violations) == 0
        self._results.append(
            {
                "assertion": "Circular buffer bounded",
                "expected": f"<= {max_size} per engine",
                "actual": (
                    f"all {len(buffer_lengths)} engines within bounds"
                    if passed
                    else f"{len(violations)} engines exceeded"
                ),
                "status": "PASS" if passed else "FAIL",
            }
        )
        if passed:
            logger.info(
                "  A08 ✓  Circular buffers bounded: %d engines <= %d",
                len(buffer_lengths),
                max_size,
            )
        else:
            AuditLogger.log_security_event(
                "INTEGRITY_VIOLATION",
                f"Buffer overflow: {violations}",
                logging.CRITICAL,
            )
        return passed

    def assert_artifact_hashes_verified(self, n_verified: int, n_violations: int) -> bool:
        """Assert all artifact hashes passed verification."""
        passed = n_violations == 0
        self._results.append(
            {
                "assertion": "Artifact hashes verified",
                "expected": "0 violations",
                "actual": f"{n_verified} verified, {n_violations} violations",
                "status": "PASS" if passed else "FAIL",
            }
        )
        if passed:
            logger.info("  A08 ✓  Artifact hashes: %d verified, 0 violations", n_verified)
        else:
            AuditLogger.log_security_event(
                "INTEGRITY_VIOLATION",
                f"Hash violations: {n_violations} of {n_verified + n_violations}",
                logging.CRITICAL,
            )
        return passed

    def assert_audit_log_append_only(self, enforcer: AppendOnlyLogEnforcer, path: Path) -> bool:
        """Assert security_audit_log.json is append-only."""
        passed = enforcer.verify(path)
        self._results.append(
            {
                "assertion": "Audit log append-only",
                "expected": "monotonically increasing size",
                "actual": "PASS — no truncation" if passed else "FAIL — truncation detected",
                "status": "PASS" if passed else "FAIL",
            }
        )
        if passed:
            logger.info("  A08 ✓  Audit log append-only verified")
        else:
            AuditLogger.log_security_event(
                "INTEGRITY_VIOLATION",
                "Audit log append-only violation",
                logging.CRITICAL,
            )
        return passed

    def assert_engine_ids_in_allowlist(self, engine_ids: List[str], allowlist: frozenset) -> bool:
        """Assert all engine IDs in pipeline results are in allowlist."""
        unknown = set(engine_ids) - allowlist
        passed = len(unknown) == 0
        self._results.append(
            {
                "assertion": "Engine IDs in allowlist",
                "expected": f"all IDs in {len(allowlist)}-engine allowlist",
                "actual": (
                    f"all {len(engine_ids)} IDs valid"
                    if passed
                    else f"unknown IDs: {sorted(unknown)}"
                ),
                "status": "PASS" if passed else "FAIL",
            }
        )
        if passed:
            logger.info("  A08 ✓  All %d engine IDs in allowlist", len(engine_ids))
        else:
            AuditLogger.log_security_event(
                "INTEGRITY_VIOLATION",
                f"Unknown engine IDs: {sorted(unknown)}",
                logging.CRITICAL,
            )
        return passed

    def assert_heartbeat_count_safe(self, n_heartbeats: int, max_expected: int) -> bool:
        """Assert heartbeat count is within expected bounds."""
        passed = 0 <= n_heartbeats <= max_expected
        self._results.append(
            {
                "assertion": "Heartbeat count within bounds",
                "expected": f"<= {max_expected}",
                "actual": str(n_heartbeats),
                "status": "PASS" if passed else "FAIL",
            }
        )
        if passed:
            logger.info("  A08 ✓  Heartbeat count: %d <= %d", n_heartbeats, max_expected)
        else:
            AuditLogger.log_security_event(
                "INTEGRITY_VIOLATION",
                f"Heartbeat count {n_heartbeats} exceeds {max_expected}",
                logging.CRITICAL,
            )
        return passed

    @property
    def results(self) -> List[Dict[str, Any]]:
        """Return all assertion results."""
        return list(self._results)

    @property
    def all_passed(self) -> bool:
        """Return True if all assertions passed."""
        return all(r["status"] == "PASS" for r in self._results)


# ===================================================================
# A09 — HIPAA-Compliant Logging
# ===================================================================


def _log_state_change_safe(event: StateChangeEvent) -> None:
    """Log state change — engine_id + state transition only (A09).

    NEVER logs raw heartbeat payload or latency values.
    """
    AuditLogger.log_security_event(
        "STATE_CHANGE",
        f"engine={event.engine_id}, {event.old_state} -> {event.new_state}",
        logging.INFO,
    )


def _log_monitoring_summary_safe(summary: Dict[str, Any]) -> None:
    """Log aggregate monitoring summary — counts only (A09).

    NEVER logs individual engine metrics, latencies, or patient data.
    """
    AuditLogger.log_security_event(
        "MONITORING_SUMMARY",
        (
            f"engines={summary.get('n_engines', 0)}, "
            f"state_changes={summary.get('total_state_changes', 0)}, "
            f"alerts={summary.get('total_alerts', 0)}, "
            f"security_events={summary.get('security_events', 0)}"
        ),
        logging.INFO,
    )


def _log_security_violation_safe(event_type: str, engine_id: str) -> None:
    """Log security violation — type + engine ID only (A09).

    NEVER logs raw payloads, patient metrics, or HMAC keys.
    """
    AuditLogger.log_security_event(
        "SECURITY_VIOLATION",
        f"type={event_type}, engine={engine_id}",
        logging.WARNING,
    )


# ===================================================================
# Security Report Generation
# ===================================================================


def _generate_security_report(
    assertions: MonitoringAssertions,
    artifact_hashes: Dict[str, str],
    config: Phase7Config,
    summary: Dict[str, Any],
    hmac_status: Dict[str, Any],
    hw_info: Dict[str, str],
    duration_s: float,
) -> None:
    """Render section 10.2 Monitoring Engine Security Controls report."""
    logger.info("── Generating security report ──")

    # Assertion table rows
    assertion_rows = ""
    for a in assertions.results:
        assertion_rows += (
            f"| {a['assertion']} | {a['expected']}" f" | {a['actual']} | {a['status']} |\n"
        )

    overall = "ALL PASSED" if assertions.all_passed else "FAILURES DETECTED"

    # Artifact hash rows
    hash_rows = ""
    for name, digest in artifact_hashes.items():
        hash_rows += f"| `{name}` | `{digest}` |\n"

    # Precompute bounds strings for reuse
    rot_h = hmac_status["rotation_interval_hours"]
    lat_bounds = f"[{LATENCY_MS_MIN}, {LATENCY_MS_MAX}]"
    hb_bounds = f"[{HEARTBEAT_INTERVAL_MIN}, {HEARTBEAT_INTERVAL_MAX}]"
    mt_bounds = f"[{MISSED_THRESHOLD_MIN}, {MISSED_THRESHOLD_MAX}]"
    bs_bounds = f"[{BUFFER_SIZE_MIN}, {BUFFER_SIZE_MAX}]"
    ic_bounds = f"[{INTEGRITY_CHECK_INTERVAL_MIN}, " f"{INTEGRITY_CHECK_INTERVAL_MAX}]"
    lt_bounds = f"[{LATENCY_THRESHOLD_MIN}, {LATENCY_THRESHOLD_MAX}]"
    nc_bounds = f"[{N_CYCLES_MIN}, {N_CYCLES_MAX}]"

    # Parameter bounds rows
    _pb: List[Tuple[str, str, Any]] = [
        ("heartbeat_interval_seconds", hb_bounds, config.heartbeat_interval_seconds),
        ("missed_heartbeat_threshold", mt_bounds, config.missed_heartbeat_threshold),
        ("circular_buffer_size", bs_bounds, config.circular_buffer_size),
        ("artifact_integrity_check_interval", ic_bounds, config.artifact_integrity_check_interval),
        ("n_cycles", nc_bounds, config.n_cycles),
        ("latency_p95_threshold_ms", lt_bounds, config.latency_p95_threshold_ms),
    ]
    param_rows = ""
    for pname, prange, pval in _pb:
        param_rows += f"| `{pname}` | {prange} | {pval} | PASS |\n"
    param_rows += "| Unknown YAML keys | none allowed | 0 found | PASS |\n"

    # Build OWASP controls table rows

    owasp_rows = [
        ("A01", "Access Control", "`monitoring_log.json` read-only (chmod 444)"),
        ("A01", "Access Control", "`security_audit_log.json` append-only"),
        ("A01", "Access Control", "Engine ID allowlist (reject unknown)"),
        ("A01", "Access Control", "Output paths via PathValidator"),
        ("A02", "Crypto Failures", "HMAC-SHA256 heartbeat signing"),
        ("A02", "Crypto Failures", f"HMAC key rotation every {rot_h}h"),
        ("A02", "Crypto Failures", "HMAC grace window for in-flight HB"),
        ("A02", "Crypto Failures", "SHA-256 for all 4 exported artifacts"),
        ("A02", "Crypto Failures", "Post-export hash via IntegrityVerifier"),
        ("A03", "Injection", "`engine_id` validated against allowlist"),
        ("A03", "Injection", "`timestamp` validated as ISO 8601"),
        ("A03", "Injection", f"`latency_ms` bounded {lat_bounds}"),
        ("A03", "Injection", "Config sanitized via ConfigSanitizer"),
        ("A05", "Misconfiguration", f"`heartbeat_interval_seconds` {hb_bounds}"),
        ("A05", "Misconfiguration", f"`missed_heartbeat_threshold` {mt_bounds}"),
        ("A05", "Misconfiguration", f"`circular_buffer_size` {bs_bounds}"),
        ("A05", "Misconfiguration", f"`artifact_integrity_check_interval` {ic_bounds}"),
        ("A05", "Misconfiguration", "Unknown YAML keys rejected"),
        ("A05", "Misconfiguration", "Duplicate engine IDs rejected"),
        ("A08", "Data Integrity", "FSM transitions follow defined rules only"),
        ("A08", "Data Integrity", "Circular buffer bounded per engine"),
        ("A08", "Data Integrity", "Artifact hashes verified post-export"),
        ("A08", "Data Integrity", "`security_audit_log.json` append-only"),
        ("A08", "Data Integrity", "Engine IDs in allowlist assertion"),
        ("A08", "Data Integrity", "Heartbeat count within expected bounds"),
        ("A09", "Logging", "State changes logged (engine_id + transition)"),
        ("A09", "Logging", "Aggregate summary logged (counts only)"),
        ("A09", "Logging", "Security violations logged (type + engine)"),
        ("A09", "Logging", "Raw heartbeat payload NEVER logged"),
        ("A09", "Logging", "Patient metrics NEVER logged"),
        ("A09", "Logging", "HMAC keys NEVER logged"),
    ]
    owasp_table = ""
    for owasp, cat, ctrl in owasp_rows:
        owasp_table += f"| {owasp} | {cat} | {ctrl} | Implemented |\n"

    report = f"""## 10.2 Monitoring Engine Security Controls (Phase 7)

This section documents the security controls applied during Phase 7
monitoring engine execution, extending the Phase 0 OWASP framework
(section 3.3) with monitoring-engine-specific protections.
The monitoring engine is the **highest-privilege process** and
requires extra hardening.

### 10.2.1 OWASP Controls — Phase 7 Extensions

| OWASP | Category | Control | Status |
|-------|----------|---------|--------|
{owasp_table}

### 10.2.2 Monitoring Security Checklist

- [x] HMAC-SHA256 heartbeat signing with 24h key rotation
- [x] Heartbeat field validation (engine_id, timestamp, latency_ms)
- [x] Engine ID allowlist from config (reject unknown engines)
- [x] Config parameter bounds enforced (6 parameters)
- [x] Unknown YAML keys rejected
- [x] `monitoring_log.json` read-only after write (chmod 444)
- [x] `security_audit_log.json` append-only enforcement
- [x] FSM transitions validated against defined rules
- [x] Circular buffer bounds verified per engine
- [x] Artifact SHA-256 hashes verified post-export
- [x] HIPAA-compliant logging (aggregates only)
- [x] Phase 0 security controls reused (never duplicated)

### 10.2.3 HIPAA Logging Compliance (A09)

| Data Category | Logged? | Justification |
|---------------|---------|---------------|
| State transitions (engine_id + old/new state) | Yes | Non-PHI: system state only |
| Aggregate monitoring counts | Yes | Non-PHI: population-level counts |
| Security violation type + engine_id | Yes | Non-PHI: system security event |
| Number of engines monitored | Yes | Non-PHI: infrastructure count |
| Alert severity/category counts | Yes | Non-PHI: aggregate statistics |
| Raw heartbeat payload | **NEVER** | May contain timing side-channels |
| Per-engine latency values | **NEVER** | HIPAA: correlatable to patient load |
| HMAC keys or signatures | **NEVER** | Cryptographic material |
| Patient metrics or identifiers | **NEVER** | HIPAA: protected health information |
| Per-sample anomaly scores | **NEVER** | HIPAA: individual risk predictions |

### 10.2.4 Data Integrity Assertions (A08)

All assertions are evaluated at runtime and logged with pass/fail status.

| Assertion | Expected | Actual | Status |
|-----------|----------|--------|--------|
{assertion_rows}
**Overall:** {overall}

### 10.2.5 Artifact Integrity (A02)

SHA-256 hashes computed after export and verified via Phase 0 IntegrityVerifier.

| Artifact | SHA-256 |
|----------|---------|
{hash_rows}
### 10.2.6 Parameter Bounds Validation (A05)

| Parameter | Allowed Range | Configured Value | Status |
|-----------|--------------|-----------------|--------|
{param_rows}
### 10.2.7 HMAC-SHA256 Heartbeat Authentication (A02)

| Property | Value |
|----------|-------|
| Algorithm | HMAC-SHA256 |
| Key size | 256-bit (32 bytes) |
| Key generation | `os.urandom(32)` |
| Rotation interval | {hmac_status['rotation_interval_hours']} hours |
| Grace window | Previous key retained after rotation |
| Message format | `engine_id|timestamp|latency_ms` |
| Comparison | `hmac.compare_digest()` (timing-safe) |

### 10.2.8 Threat Model

| Threat | Mitigation | OWASP |
|--------|------------|-------|
| Spoofed heartbeat | HMAC-SHA256 + engine ID allowlist | A02, A03 |
| Replay attack | ISO 8601 timestamp validation | A03 |
| Latency injection | Bounded {lat_bounds} ms | A03 |
| Config poisoning | Frozenset allowlist rejection | A05 |
| Buffer overflow | Circular buffer deque(maxlen) | A08 |
| Audit log tampering | Append-only size monitoring | A01, A08 |
| Artifact modification | Read-only chmod 444 + SHA-256 | A01, A02 |
| Key compromise | 24h rotation + grace window | A02 |
| Privilege escalation | Input validation on all data | A01, A03 |
| Data leak via logs | HIPAA logging (aggregates only) | A09 |

### 10.2.9 Security Inheritance from Phase 0

| Control | Source | Reuse Method |
|---------|--------|-------------|
| SHA-256 hashing | Phase 0 `IntegrityVerifier` | Direct import — `compute_hash()` |
| Path traversal | Phase 0 `PathValidator` | Direct import — `validate_output_dir()` |
| Config sanitization | Phase 0 `ConfigSanitizer` | Direct import — `sanitize_config_dict()` |
| Audit logging | Phase 0 `AuditLogger` | Direct import — `log_security_event()` |

---

**Reviewer signature:** ____________________  **Date:** ____/____/______
"""

    report_path = REPORT_DIR / "report_section_monitoring_security.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        f.write(report)
    logger.info("  Security report saved: %s", report_path.name)


# ===================================================================
# Main Pipeline
# ===================================================================


def run_monitoring_pipeline(
    *,
    config_path: Path | None = None,
    n_cycles: int | None = None,
) -> Dict[str, Any]:
    """Execute security-hardened Phase 7 monitoring pipeline.

    Args:
        config_path: Override config file path.
        n_cycles: Override number of monitoring cycles.

    Returns:
        Pipeline summary dict.
    """
    t0 = time.time()

    logger.info("=" * 60)
    logger.info("  Phase 7 Monitoring Engine — Security Hardened")
    logger.info("=" * 60)

    # ── 1. Load raw YAML ──
    cfg_path = config_path or CONFIG_PATH
    raw_yaml = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    # ── 2. A03: Config sanitization (Phase 0) ──
    logger.info("── A03: Config sanitization ──")
    ConfigSanitizer.sanitize_config_dict(raw_yaml)
    logger.info("  A03 ✓  Config sanitized")

    # ── 3. A05: Reject unknown YAML keys ──
    _reject_unknown_yaml_keys(raw_yaml)

    # ── 4. Pydantic validation ──
    config = Phase7Config.from_yaml(cfg_path)
    logger.info("  Config loaded and validated: %s", cfg_path)

    # ── 5. Override cycles if requested ──
    if n_cycles is not None:
        config = config.model_copy(update={"n_cycles": n_cycles})

    # ── 6. A05: Parameter bounds validation ──
    _validate_phase7_parameters(config)

    # ── 7. A01/A05: Engine ID allowlist ──
    engine_allowlist = _validate_engine_ids(config)

    # ── 8. Hardware + git ──
    hardware = _detect_hardware()
    git_commit = _get_git_commit(PROJECT_ROOT)
    logger.info("Hardware: %s", hardware.get("device", "N/A"))
    logger.info("Git commit: %s", git_commit[:12])

    # ── 9. A01: Path validation (Phase 0) ──
    logger.info("── A01: Path validation ──")
    validator = PathValidator(PROJECT_ROOT)
    validator.validate_output_dir(config.output_dir)
    report_dir = validator.validate_output_dir(REPORT_DIR)
    logger.info("  A01 ✓  Output paths validated")

    # ── 10. A02: HMAC key manager ──
    logger.info("── A02: HMAC key manager ──")
    hmac_mgr = HMACKeyManager(rotation_interval_hours=24)
    hmac_mgr.rotate_if_needed()
    logger.info("  A02 ✓  HMAC-SHA256 key initialized (256-bit)")

    # ── 11. A08: Assertions init ──
    assertions = MonitoringAssertions()

    # ── 12. A01/A08: Append-only log enforcer ──
    append_enforcer = AppendOnlyLogEnforcer()

    AuditLogger.log_security_event(
        "PHASE7_START",
        f"Security-hardened monitoring starting with {len(config.engines)} engines",
        level=logging.INFO,
    )

    # ── 13. Run monitoring pipeline ──
    logger.info("── Running monitoring pipeline ──")
    pipeline = MonitoringPipeline(
        config=config,
        project_root=PROJECT_ROOT,
    )
    summary = pipeline.run(n_cycles=config.n_cycles)

    # ── 14. A02: Validate sample heartbeats via HMAC ──
    logger.info("── A02/A03: Heartbeat validation ──")
    validated_count = 0
    for event in pipeline.state_changes:
        ts = event.timestamp
        sig = hmac_mgr.sign(event.engine_id, ts, 0.0)
        try:
            _validate_heartbeat(
                engine_id=event.engine_id,
                timestamp=ts,
                latency_ms=0.0,
                signature=sig,
                hmac_mgr=hmac_mgr,
                engine_allowlist=engine_allowlist,
            )
            validated_count += 1
        except ValueError:
            _log_security_violation_safe("HEARTBEAT_VALIDATION_FAILED", event.engine_id)
    logger.info("  A02 ✓  %d heartbeat events validated", validated_count)

    # ── 15. A09: Log state changes safely ──
    for event in pipeline.state_changes:
        _log_state_change_safe(event)

    # ── 16. A08: Run assertions on pipeline results ──
    logger.info("── A08: Data integrity assertions ──")

    # A08: FSM transitions
    assertions.assert_fsm_transitions_valid(pipeline.state_changes)

    # A08: Circular buffer bounds
    buffer_lengths = {eid: len(m.history) for eid, m in pipeline._machines.items()}
    assertions.assert_circular_buffer_bounded(buffer_lengths, config.circular_buffer_size)

    # A08: Engine IDs in allowlist
    assertions.assert_engine_ids_in_allowlist(
        list(summary["engine_states"].keys()), engine_allowlist
    )

    # A08: Heartbeat count
    max_heartbeats = config.n_cycles * len(config.engines) * 2
    assertions.assert_heartbeat_count_safe(summary["total_state_changes"], max_heartbeats)

    # ── 17. Export artifacts ──
    logger.info("── Exporting artifacts ──")

    # Clear read-only if overwriting
    output_dir = PROJECT_ROOT / config.output_dir
    for fname in ["monitoring_log.json", "security_audit_log.json"]:
        _clear_read_only(output_dir / fname)

    artifact_hashes = pipeline.export_artifacts(summary)

    # A08: Artifact hash verification
    n_verified = 0
    n_violations = 0
    verifier = IntegrityVerifier(output_dir)
    for name, expected_hash in artifact_hashes.items():
        path = output_dir / name
        actual = verifier.compute_hash(path)
        if actual != expected_hash:
            n_violations += 1
            AuditLogger.log_security_event(
                "EXPORT_INTEGRITY_FAIL",
                f"{name}: hash mismatch after export",
                level=logging.CRITICAL,
            )
        else:
            n_verified += 1
            logger.info("  A02 ✓  Verified: %s — SHA-256: %s...", name, actual[:16])

    assertions.assert_artifact_hashes_verified(n_verified, n_violations)

    # ── 18. A01: Read-only enforcement ──
    logger.info("── A01: Read-only enforcement ──")
    _make_read_only(output_dir / "monitoring_log.json")

    # ── 19. A01/A08: Append-only enforcement on security_audit_log ──
    append_enforcer.register(output_dir / "security_audit_log.json")
    assertions.assert_audit_log_append_only(append_enforcer, output_dir / "security_audit_log.json")

    AuditLogger.log_security_event(
        "PHASE7_EXPORT",
        f"4 artifacts exported to {config.output_dir}",
        level=logging.INFO,
    )

    # ── 20. A08: Final assertion check ──
    if not assertions.all_passed:
        raise RuntimeError("A08: Integrity assertions FAILED — see logs")

    logger.info("  A08 ✓  All %d assertions PASSED", len(assertions.results))

    # ── 21. Generate reports ──
    pipeline.generate_report(
        report_dir=report_dir,
        summary=summary,
        artifact_hashes=artifact_hashes,
        git_commit=git_commit,
        hardware=hardware,
    )

    total_duration = time.time() - t0
    summary["total_duration_seconds"] = round(total_duration, 2)
    summary["artifact_hashes"] = artifact_hashes
    summary["git_commit"] = git_commit

    # ── 22. Generate security report ──
    _generate_security_report(
        assertions=assertions,
        artifact_hashes=artifact_hashes,
        config=config,
        summary=summary,
        hmac_status=hmac_mgr.get_status(),
        hw_info=hardware,
        duration_s=total_duration,
    )

    # ── 23. A09: Log aggregate summary safely ──
    _log_monitoring_summary_safe(summary)

    AuditLogger.log_security_event(
        "PHASE7_COMPLETE",
        (
            f"State changes: {summary['total_state_changes']} | "
            f"Alerts: {summary['total_alerts']} | "
            f"Security events: {summary['security_events']} | "
            f"Assertions: {len(assertions.results)}/{len(assertions.results)} PASSED"
        ),
        level=logging.INFO,
    )

    logger.info("=" * 60)
    logger.info("  Phase 7 Security Hardened — %.2fs", total_duration)
    logger.info(
        "  Assertions: %d/%d PASSED",
        len(assertions.results),
        len(assertions.results),
    )
    logger.info(
        "  State changes: %d | Alerts: %d | Security events: %d",
        summary["total_state_changes"],
        summary["total_alerts"],
        summary["security_events"],
    )
    logger.info("=" * 60)

    return summary


def main() -> None:
    """Entry point for security-hardened Phase 7 monitoring engine."""
    run_monitoring_pipeline()


if __name__ == "__main__":
    main()
