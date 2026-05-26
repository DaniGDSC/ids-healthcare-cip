"""log_phase0_event + ColumnAllowlist tests — A03/A09 controls."""
from __future__ import annotations

import logging

import pytest

from module0_analysis import ColumnAllowlist, log_phase0_event
from common.phi import BIOMETRIC_COLUMNS


def test_column_allowlist_passes_when_all_present():
    actual = {"Label", "Attack Category", "Temp"}
    result = ColumnAllowlist.validate(["Label", "Attack Category"], actual)
    assert result == ["Label", "Attack Category"]


def test_column_allowlist_raises_on_missing():
    actual = {"Label"}
    with pytest.raises(ValueError, match="unknown columns:.*Attack Category"):
        ColumnAllowlist.validate(["Label", "Attack Category"], actual)


def test_log_phase0_event_redacts_biometric_keys(caplog):
    """Audit payload must not leak PHI even when caller forgets."""
    # Pick a biometric col we know is in BIOMETRIC_COLUMNS
    bio_key = next(iter(BIOMETRIC_COLUMNS))
    caplog.set_level(logging.INFO, logger="phase0.security.audit")
    log_phase0_event("TEST", {bio_key: 98.6, "non_phi": "ok"})
    # The local logger receives the redacted payload as a stringified dict
    record_msgs = [r.message for r in caplog.records]
    full_log = " ".join(record_msgs)
    assert "[REDACTED-PHI]" in full_log
    assert "98.6" not in full_log, "Raw biometric value leaked into log"


def test_log_phase0_event_non_phi_keys_pass_through(caplog):
    caplog.set_level(logging.INFO, logger="phase0.security.audit")
    log_phase0_event("TEST", {"file": "ok.csv", "rows": 100})
    full_log = " ".join(r.message for r in caplog.records)
    assert "ok.csv" in full_log
    assert "100" in full_log
    assert "REDACTED" not in full_log


def test_log_phase0_event_no_payload(caplog):
    caplog.set_level(logging.INFO, logger="phase0.security.audit")
    log_phase0_event("NO_PAYLOAD_EVENT")
    full_log = " ".join(r.message for r in caplog.records)
    assert "NO_PAYLOAD_EVENT" in full_log


def test_log_phase0_event_level_respected(caplog):
    caplog.set_level(logging.WARNING, logger="phase0.security.audit")
    log_phase0_event("LOW", level=logging.INFO)
    log_phase0_event("HIGH", level=logging.ERROR)
    levels = {r.levelno for r in caplog.records}
    assert logging.ERROR in levels
    assert logging.INFO not in levels  # filtered by caplog level
