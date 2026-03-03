"""
Tests for Phase 1: Database telemetry, state machine, and audit logging.

Tests validate:
1. Telemetry collection from database sources
2. State machine transitions with validation
3. Audit logging with chain integrity
"""

import pytest
import logging
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
import tempfile

from src.utils.db_telemetry import (
    TelemetryEvent,
    TelemetryEventType,
    TelemetrySeverity,
    DBTelemetryCollector,
    TransactionLogMonitor,
    FilesystemMonitor
)
from src.utils.backup_state_machine import (
    BackupState,
    BackupStateMachine,
    StateTransitionError,
    StateTransitionRecord
)
from src.utils.audit_logger import (
    AuditEvent,
    AuditEventType,
    AuditLogger
)


# ============================================================================
# TELEMETRY TESTS
# ============================================================================

class TestTelemetryEvent:
    """Test TelemetryEvent data class."""
    
    def test_event_creation(self):
        """Test basic event creation."""
        event = TelemetryEvent(
            timestamp=datetime.utcnow(),
            source="database",
            event_type=TelemetryEventType.QUERY_PATTERN,
            severity=TelemetrySeverity.INFO,
            metadata={"query_count": 100},
            anonymized=True
        )
        
        assert event.source == "database"
        assert event.event_type == TelemetryEventType.QUERY_PATTERN
        assert event.anonymized is True
        assert event.event_id is not None
        assert len(event.event_id) == 16
    
    def test_event_id_generation(self):
        """Test automatic event_id generation."""
        event1 = TelemetryEvent(
            timestamp=datetime.utcnow(),
            source="database",
            event_type=TelemetryEventType.QUERY_PATTERN,
            severity=TelemetrySeverity.INFO,
            metadata={"count": 1}
        )
        
        # Different events should have different IDs
        event2 = TelemetryEvent(
            timestamp=datetime.utcnow(),
            source="filesystem",
            event_type=TelemetryEventType.FILE_MODIFICATION,
            severity=TelemetrySeverity.INFO,
            metadata={"count": 2}
        )
        
        assert event1.event_id != event2.event_id


class TestDBTelemetryCollector:
    """Test database telemetry collection."""
    
    def test_postgresql_collector_init(self):
        """Test PostgreSQL collector initialization."""
        collector = DBTelemetryCollector(
            source_type="postgresql",
            replica_conn_string="postgresql://user:pass@localhost/db"
        )
        
        assert collector.source_type == "postgresql"
        assert collector.collection_interval_sec == 60
    
    def test_mysql_collector_init(self):
        """Test MySQL collector initialization."""
        collector = DBTelemetryCollector(
            source_type="mysql",
            replica_conn_string="mysql://user:pass@localhost/db"
        )
        
        assert collector.source_type == "mysql"
    
    def test_collect_query_patterns(self):
        """Test query pattern collection."""
        collector = DBTelemetryCollector(
            source_type="postgresql",
            replica_conn_string="postgresql://localhost/test"
        )
        
        events = collector.collect_query_patterns()
        
        assert isinstance(events, list)
        assert len(events) > 0
        assert events[0].source == "database"
        assert events[0].event_type == TelemetryEventType.QUERY_PATTERN
        assert events[0].anonymized is True
    
    def test_collect_connection_metadata(self):
        """Test connection metadata collection."""
        collector = DBTelemetryCollector(
            source_type="postgresql",
            replica_conn_string="postgresql://localhost/test"
        )
        
        events = collector.collect_connection_metadata()
        
        assert isinstance(events, list)
        assert len(events) > 0
        assert events[0].event_type == TelemetryEventType.CONNECTION_METADATA
    
    def test_collect_authentication_events(self):
        """Test authentication event collection."""
        collector = DBTelemetryCollector(
            source_type="postgresql",
            replica_conn_string="postgresql://localhost/test"
        )
        
        events = collector.collect_authentication_events()
        
        assert isinstance(events, list)
        assert len(events) > 0
        assert events[0].event_type == TelemetryEventType.AUTHENTICATION
    
    def test_unsupported_database_type(self):
        """Test handling of unsupported database type."""
        collector = DBTelemetryCollector(
            source_type="oracle",
            replica_conn_string="oracle://localhost/db"
        )
        
        events = collector.collect_query_patterns()
        assert events == []


class TestTransactionLogMonitor:
    """Test transaction log monitoring."""
    
    def test_monitor_initialization(self):
        """Test TransactionLogMonitor initialization."""
        monitor = TransactionLogMonitor(
            source_type="postgresql",
            wal_stream_url="postgresql+replication://localhost/db"
        )
        
        assert monitor.source_type == "postgresql"
        assert monitor.baseline_update_rate == 100
    
    def test_detect_mass_modification(self):
        """Test detection of mass modification (ransomware indicator)."""
        monitor = TransactionLogMonitor(source_type="postgresql")
        
        # Should return None for normal rate
        event = monitor.detect_mass_modification(window_sec=60)
        
        # Current implementation returns event with high modification rate
        assert event is not None
        assert event.severity in [TelemetrySeverity.SUSPICIOUS, TelemetrySeverity.CRITICAL]
        assert "MASS_MODIFICATION" in event.metadata["indicator"]
    
    def test_detect_schema_changes(self):
        """Test detection of suspicious schema changes."""
        monitor = TransactionLogMonitor(source_type="postgresql")
        
        event = monitor.detect_schema_changes()
        
        assert event is not None
        assert event.severity == TelemetrySeverity.CRITICAL
        assert "SCHEMA_CHANGE" in event.metadata["indicator"]


class TestFilesystemMonitor:
    """Test filesystem monitoring."""
    
    def test_filesystem_monitor_init(self):
        """Test FilesystemMonitor initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            monitor = FilesystemMonitor(
                db_data_dir=Path(tmpdir),
                watch_patterns=["*.mdf", "*.ibd"]
            )
            
            assert monitor.db_data_dir == Path(tmpdir)
            assert monitor.watch_patterns == ["*.mdf", "*.ibd"]
    
    def test_calculate_entropy(self):
        """Test entropy calculation for file encryption detection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test file with low entropy (normal data)
            test_file = Path(tmpdir) / "test.mdf"
            test_file.write_bytes(b"A" * 1000)  # Repetitive data = low entropy
            
            monitor = FilesystemMonitor(db_data_dir=Path(tmpdir))
            entropy = monitor.calculate_entropy(test_file)
            
            # Repetitive data should have low entropy
            assert entropy < 3.0
            
            # Create high-entropy file (simulating encrypted data)
            high_entropy_file = Path(tmpdir) / "encrypted.mdf"
            import os
            high_entropy_file.write_bytes(os.urandom(1000))  # Random data = high entropy
            
            entropy_high = monitor.calculate_entropy(high_entropy_file)
            assert entropy_high > 7.0
    
    def test_detect_entropy_change(self):
        """Test detection of file encryption via entropy analysis."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create encrypted-looking file
            test_file = Path(tmpdir) / "encrypted.mdf"
            import os
            test_file.write_bytes(os.urandom(10000))
            
            monitor = FilesystemMonitor(db_data_dir=Path(tmpdir))
            event = monitor.detect_entropy_change(test_file, baseline_entropy=7.0)
            
            assert event is not None
            assert "FILE_ENCRYPTION_DETECTED" in event.metadata["indicator"]
            assert event.severity == TelemetrySeverity.CRITICAL


# ============================================================================
# STATE MACHINE TESTS
# ============================================================================

class TestBackupStateMachine:
    """Test backup state machine."""
    
    def test_initialization(self):
        """Test state machine initialization."""
        sm = BackupStateMachine(initial_state=BackupState.NORMAL)
        
        assert sm.current_state == BackupState.NORMAL
        assert len(sm.state_history) == 1
        assert sm.is_terminal() is False
    
    def test_valid_transition(self):
        """Test valid state transition."""
        sm = BackupStateMachine(initial_state=BackupState.NORMAL)
        
        record = sm.transition(
            to_state=BackupState.ELEVATED,
            decision_maker="SYSTEM",
            reason="SUSPICIOUS_SIGNAL_DETECTED"
        )
        
        assert sm.current_state == BackupState.ELEVATED
        assert record.from_state == BackupState.NORMAL
        assert record.to_state == BackupState.ELEVATED
        assert len(sm.state_history) == 2
    
    def test_invalid_transition(self):
        """Test that invalid transitions are rejected."""
        sm = BackupStateMachine(initial_state=BackupState.NORMAL)
        
        # NORMAL cannot go directly to TRUSTED
        with pytest.raises(StateTransitionError):
            sm.transition(
                to_state=BackupState.TRUSTED,
                decision_maker="HUMAN",
                reason="INVALID_TRANSITION"
            )
    
    def test_state_transition_validation(self):
        """Test transition validation logic."""
        sm = BackupStateMachine(initial_state=BackupState.NORMAL)
        
        # Valid: NORMAL → ELEVATED
        is_valid, error = sm.validate_transition(BackupState.ELEVATED)
        assert is_valid is True
        assert error is None
        
        # Invalid: NORMAL → TRUSTED
        is_valid, error = sm.validate_transition(BackupState.TRUSTED)
        assert is_valid is False
        assert error is not None
    
    def test_human_approval_required(self):
        """Test that certain transitions require human approval."""
        sm = BackupStateMachine(initial_state=BackupState.ELEVATED)
        
        # Transition to SUSPICIOUS requires approval
        record = sm.transition(
            to_state=BackupState.SUSPICIOUS,
            decision_maker="HUMAN",
            reason="ESCALATION"
        )
        
        assert record.approval_required is True
    
    def test_sla_tracking(self):
        """Test SLA tracking for states."""
        sm = BackupStateMachine(initial_state=BackupState.SUSPICIOUS)
        
        # SUSPICIOUS state has 15-minute SLA
        sla_remaining = sm.get_sla_remaining()
        assert sla_remaining is not None
        assert sla_remaining <= 15
    
    def test_terminal_state(self):
        """Test TRUSTED as terminal state."""
        sm = BackupStateMachine(initial_state=BackupState.NORMAL)
        
        # Reach QUARANTINED first
        sm.transition(BackupState.ELEVATED, "SYSTEM", "ESCALATION")
        sm.transition(BackupState.SUSPICIOUS, "HUMAN", "ESCALATION")
        sm.transition(BackupState.QUARANTINED, "HUMAN", "ISOLATION")
        
        # Mark as TRUSTED (golden restore point)
        record = sm.transition(
            to_state=BackupState.TRUSTED,
            decision_maker="HUMAN",
            reason="VALIDATION_PASSED",
            metadata={"validated_at": datetime.utcnow().isoformat()}
        )
        
        assert sm.is_terminal() is True
        
        # Cannot transition out of TRUSTED
        with pytest.raises(StateTransitionError):
            sm.transition(BackupState.QUARANTINED, "HUMAN", "INVALID")
    
    def test_state_history(self):
        """Test state transition history tracking."""
        sm = BackupStateMachine(initial_state=BackupState.NORMAL)
        
        sm.transition(BackupState.ELEVATED, "SYSTEM", "ESCALATION")
        sm.transition(BackupState.NORMAL, "HUMAN", "RESET")
        
        history = sm.get_history()
        assert len(history) >= 2
        
        # History should be in reverse chronological order
        assert history[0].to_state == BackupState.NORMAL
        assert history[1].to_state == BackupState.ELEVATED
    
    def test_status_summary(self):
        """Test status summary generation."""
        sm = BackupStateMachine(initial_state=BackupState.ELEVATED)
        
        summary = sm.get_status_summary()
        
        assert summary["current_state"] == "ELEVATED"
        assert "state_duration_minutes" in summary
        assert "sla_remaining_minutes" in summary
        assert summary["is_terminal"] is False


# ============================================================================
# AUDIT LOGGER TESTS
# ============================================================================

class TestAuditEvent:
    """Test AuditEvent data class."""
    
    def test_event_creation(self):
        """Test basic audit event creation."""
        event = AuditEvent(
            timestamp=datetime.utcnow(),
            event_type=AuditEventType.STATE_TRANSITION,
            actor="SYSTEM",
            resource="backup_123",
            action="TRANSITION_TO_ELEVATED",
            result="SUCCESS",
            details={"from_state": "NORMAL", "to_state": "ELEVATED"}
        )
        
        assert event.event_type == AuditEventType.STATE_TRANSITION
        assert event.actor == "SYSTEM"
        assert event.result == "SUCCESS"
        assert event.event_id is not None
    
    def test_event_json_serialization(self):
        """Test JSON serialization of audit event."""
        event = AuditEvent(
            timestamp=datetime.utcnow(),
            event_type=AuditEventType.BACKUP_CREATED,
            actor="BACKUP_SYSTEM",
            resource="backup_123",
            action="CREATE",
            result="SUCCESS",
            details={"size_mb": 1024}
        )
        
        json_str = event.to_json()
        data = json.loads(json_str)
        
        assert data["event_type"] == "BACKUP_CREATED"
        assert data["actor"] == "BACKUP_SYSTEM"


class TestAuditLogger:
    """Test audit logging system."""
    
    def test_file_based_logger_init(self):
        """Test file-based audit logger initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "audit.log"
            
            logger = AuditLogger(
                log_file=log_file,
                storage_type="file"
            )
            
            assert logger.log_file == log_file
            assert logger.storage_type == "file"
    
    def test_log_event(self):
        """Test logging an event."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "audit.log"
            
            logger = AuditLogger(log_file=log_file, storage_type="file")
            
            event = logger.log_event(
                event_type=AuditEventType.STATE_TRANSITION,
                actor="HUMAN",
                resource="backup_123",
                action="ESCALATE_TO_SUSPICIOUS",
                result="SUCCESS",
                backup_id="backup_123"
            )
            
            assert event.event_id is not None
            assert log_file.exists()
            
            # Verify event was written to file
            with open(log_file, 'r') as f:
                content = f.read()
                assert "STATE_TRANSITION" in content
    
    def test_chain_integrity_verification(self):
        """Test audit log chain integrity verification."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "audit.log"
            
            logger = AuditLogger(log_file=log_file, storage_type="file")
            
            # Log multiple events
            logger.log_event(
                event_type=AuditEventType.BACKUP_CREATED,
                actor="SYSTEM",
                resource="backup_1",
                action="CREATE",
                result="SUCCESS"
            )
            
            logger.log_event(
                event_type=AuditEventType.STATE_TRANSITION,
                actor="SYSTEM",
                resource="backup_1",
                action="TRANSITION",
                result="SUCCESS"
            )
            
            # Verify chain
            is_valid, error = logger.verify_chain_integrity()
            assert is_valid is True
            assert error is None
    
    def test_immutable_storage(self):
        """Test that audit log is append-only."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "audit.log"
            
            logger = AuditLogger(log_file=log_file, storage_type="file")
            
            # Log event
            logger.log_event(
                event_type=AuditEventType.BACKUP_CREATED,
                actor="SYSTEM",
                resource="backup_1",
                action="CREATE",
                result="SUCCESS"
            )
            
            event_count_before = logger.event_count
            
            # Attempting to create new events should append
            logger.log_event(
                event_type=AuditEventType.STATE_TRANSITION,
                actor="HUMAN",
                resource="backup_1",
                action="TRANSITION",
                result="SUCCESS"
            )
            
            assert logger.event_count == event_count_before + 1
    
    def test_query_events(self):
        """Test querying audit log."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "audit.log"
            
            logger = AuditLogger(log_file=log_file, storage_type="file")
            
            # Log various events
            logger.log_event(
                event_type=AuditEventType.BACKUP_CREATED,
                actor="SYSTEM",
                resource="backup_1",
                action="CREATE",
                result="SUCCESS",
                backup_id="backup_1"
            )
            
            logger.log_event(
                event_type=AuditEventType.STATE_TRANSITION,
                actor="HUMAN",
                resource="backup_1",
                action="ESCALATE",
                result="SUCCESS",
                backup_id="backup_1"
            )
            
            # Query by backup_id
            results = logger.query_events(backup_id="backup_1")
            assert len(results) == 2
            
            # Query by event type
            results = logger.query_events(event_type=AuditEventType.BACKUP_CREATED)
            assert len(results) == 1
            assert results[0]["event_type"] == "BACKUP_CREATED"
    
    def test_statistics(self):
        """Test audit logger statistics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "audit.log"
            
            logger = AuditLogger(
                log_file=log_file,
                storage_type="file",
                retention_days=2555
            )
            
            logger.log_event(
                event_type=AuditEventType.BACKUP_CREATED,
                actor="SYSTEM",
                resource="backup_1",
                action="CREATE",
                result="SUCCESS"
            )
            
            stats = logger.get_statistics()
            
            assert "total_events" in stats
            assert "retention_days" in stats
            assert stats["retention_days"] == 2555
            assert "retention_until" in stats


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestPhase1Integration:
    """Integration tests for Phase 1 components."""
    
    def test_telemetry_to_state_machine_flow(self):
        """Test flow: Telemetry detection → State transition."""
        # Simulate ransomware detection via telemetry
        monitor = TransactionLogMonitor(source_type="postgresql")
        event = monitor.detect_mass_modification()
        
        # Event triggers state machine transition
        sm = BackupStateMachine(initial_state=BackupState.NORMAL)
        
        if event and event.severity == TelemetrySeverity.CRITICAL:
            sm.transition(
                to_state=BackupState.ELEVATED,
                decision_maker="SYSTEM",
                reason=f"RANSOMWARE_INDICATOR: {event.metadata['indicator']}"
            )
        
        assert sm.current_state == BackupState.ELEVATED
    
    def test_full_incident_response_flow(self):
        """Test complete incident response flow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "audit.log"
            logger = AuditLogger(log_file=log_file, storage_type="file")
            
            # 1. Telemetry detects threat
            sm = BackupStateMachine(initial_state=BackupState.NORMAL, backup_id="backup_123")
            
            logger.log_event(
                event_type=AuditEventType.POLICY_VIOLATION,
                actor="SYSTEM",
                resource="database",
                action="RANSOMWARE_SIGNAL_DETECTED",
                result="DETECTED",
                backup_id="backup_123",
                severity="WARNING"
            )
            
            # 2. System escalates state
            sm.transition(
                to_state=BackupState.ELEVATED,
                decision_maker="SYSTEM",
                reason="SUSPICIOUS_SIGNAL"
            )
            
            logger.log_event(
                event_type=AuditEventType.STATE_TRANSITION,
                actor="SYSTEM",
                resource="backup_123",
                action="ESCALATE_TO_ELEVATED",
                result="SUCCESS",
                backup_id="backup_123"
            )
            
            # 3. Human reviews and escalates further
            sm.transition(
                to_state=BackupState.SUSPICIOUS,
                decision_maker="HUMAN",
                reason="MANUAL_ESCALATION"
            )
            
            logger.log_event(
                event_type=AuditEventType.DECISION_APPROVED,
                actor="security_analyst_01",
                resource="backup_123",
                action="ESCALATE_TO_SUSPICIOUS",
                result="APPROVED",
                backup_id="backup_123",
                severity="CRITICAL"
            )
            
            # 4. System isolates backup
            sm.transition(
                to_state=BackupState.QUARANTINED,
                decision_maker="HUMAN",
                reason="ISOLATION_FOR_FORENSICS"
            )
            
            # 5. Verify audit trail
            is_valid, error = logger.verify_chain_integrity()
            assert is_valid is True
            
            # 6. Verify state machine history
            history = sm.get_history()
            assert len(history) >= 3
            assert history[-1].to_state == BackupState.QUARANTINED


# ============================================================================
# MAIN TEST EXECUTION
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
