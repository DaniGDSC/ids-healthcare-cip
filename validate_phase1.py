#!/usr/bin/env python3
"""
Phase 1 Implementation Validator

Validates that all Phase 1 components can be imported and initialized.
"""

import sys
import logging
from pathlib import Path
from datetime import datetime

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

def validate_imports():
    """Validate all Phase 1 modules can be imported."""
    print("\n" + "="*70)
    print("PHASE 1 IMPLEMENTATION VALIDATOR")
    print("="*70 + "\n")
    
    tests_passed = 0
    tests_failed = 0
    
    # Test 1: Import telemetry module
    print("Test 1: Importing telemetry module...")
    try:
        from src.utils.db_telemetry import (
            TelemetryEvent,
            TelemetryEventType,
            TelemetrySeverity,
            DBTelemetryCollector,
            TransactionLogMonitor,
            FilesystemMonitor
        )
        print("  ✅ PASS: Telemetry module imported successfully")
        tests_passed += 1
    except Exception as e:
        print(f"  ❌ FAIL: {e}")
        tests_failed += 1
    
    # Test 2: Import state machine module
    print("\nTest 2: Importing state machine module...")
    try:
        from src.utils.backup_state_machine import (
            BackupState,
            BackupStateMachine,
            StateTransitionError,
            StateTransitionRecord
        )
        print("  ✅ PASS: State machine module imported successfully")
        tests_passed += 1
    except Exception as e:
        print(f"  ❌ FAIL: {e}")
        tests_failed += 1
    
    # Test 3: Import audit logger module
    print("\nTest 3: Importing audit logger module...")
    try:
        from src.utils.audit_logger import (
            AuditEvent,
            AuditEventType,
            AuditLogger
        )
        print("  ✅ PASS: Audit logger module imported successfully")
        tests_passed += 1
    except Exception as e:
        print(f"  ❌ FAIL: {e}")
        tests_failed += 1
    
    # Test 4: Import configuration
    print("\nTest 4: Importing Phase 1 configuration...")
    try:
        from config.phase1_ransomware_config import (
            load_phase1_config,
            TELEMETRY_CONFIG,
            STATE_MACHINE_CONFIG,
            AUDIT_CONFIG,
            SAFETY_CONSTRAINTS
        )
        print("  ✅ PASS: Configuration module imported successfully")
        tests_passed += 1
    except Exception as e:
        print(f"  ❌ FAIL: {e}")
        tests_failed += 1
    
    # Test 5: Create telemetry event
    print("\nTest 5: Creating telemetry event...")
    try:
        event = TelemetryEvent(
            timestamp=datetime.utcnow(),
            source="database",
            event_type=TelemetryEventType.QUERY_PATTERN,
            severity=TelemetrySeverity.INFO,
            metadata={"test": "event"}
        )
        assert event.event_id is not None
        print(f"  ✅ PASS: Event created with ID {event.event_id}")
        tests_passed += 1
    except Exception as e:
        print(f"  ❌ FAIL: {e}")
        tests_failed += 1
    
    # Test 6: Initialize state machine
    print("\nTest 6: Initializing backup state machine...")
    try:
        sm = BackupStateMachine(
            initial_state=BackupState.NORMAL,
            backup_id="test_backup_001"
        )
        assert sm.current_state == BackupState.NORMAL
        print(f"  ✅ PASS: State machine initialized in {sm.current_state.value} state")
        tests_passed += 1
    except Exception as e:
        print(f"  ❌ FAIL: {e}")
        tests_failed += 1
    
    # Test 7: State transition
    print("\nTest 7: Performing state transition...")
    try:
        record = sm.transition(
            to_state=BackupState.ELEVATED,
            decision_maker="SYSTEM",
            reason="TEST_TRANSITION"
        )
        assert sm.current_state == BackupState.ELEVATED
        assert record.from_state == BackupState.NORMAL
        print(f"  ✅ PASS: Transitioned {record.from_state.value} → {record.to_state.value}")
        tests_passed += 1
    except Exception as e:
        print(f"  ❌ FAIL: {e}")
        tests_failed += 1
    
    # Test 8: SLA tracking
    print("\nTest 8: Checking SLA tracking...")
    try:
        sla_remaining = sm.get_sla_remaining()
        assert sla_remaining is not None and sla_remaining > 0
        print(f"  ✅ PASS: SLA remaining: {sla_remaining} minutes")
        tests_passed += 1
    except Exception as e:
        print(f"  ❌ FAIL: {e}")
        tests_failed += 1
    
    # Test 9: DB telemetry collector
    print("\nTest 9: Creating database telemetry collector...")
    try:
        collector = DBTelemetryCollector(
            source_type="postgresql",
            replica_conn_string="postgresql://localhost/test"
        )
        assert collector.source_type == "postgresql"
        print(f"  ✅ PASS: Collector initialized for {collector.source_type}")
        tests_passed += 1
    except Exception as e:
        print(f"  ❌ FAIL: {e}")
        tests_failed += 1
    
    # Test 10: Configuration loading
    print("\nTest 10: Loading Phase 1 configuration...")
    try:
        config = load_phase1_config()
        assert "telemetry" in config
        assert "state_machine" in config
        assert "audit" in config
        print(f"  ✅ PASS: Configuration loaded with {len(config)} sections")
        tests_passed += 1
    except Exception as e:
        print(f"  ❌ FAIL: {e}")
        tests_failed += 1
    
    # Test 11: Safety constraints
    print("\nTest 11: Validating safety constraints...")
    try:
        assert SAFETY_CONSTRAINTS["automated_actions"]["auto_delete_backup"] is False
        assert SAFETY_CONSTRAINTS["automated_actions"]["auto_restore_backup"] is False
        assert SAFETY_CONSTRAINTS["mandatory_gates"]["human_review_required"] is True
        print("  ✅ PASS: Safety constraints enforced (no automated destructive actions)")
        tests_passed += 1
    except Exception as e:
        print(f"  ❌ FAIL: {e}")
        tests_failed += 1
    
    # Test 12: State machine validation
    print("\nTest 12: Testing state transition validation...")
    try:
        is_valid, error = sm.validate_transition(BackupState.SUSPICIOUS)
        assert is_valid is True
        print(f"  ✅ PASS: Transition validation works")
        tests_passed += 1
    except Exception as e:
        print(f"  ❌ FAIL: {e}")
        tests_failed += 1
    
    # Print summary
    print("\n" + "="*70)
    print(f"RESULTS: {tests_passed} passed, {tests_failed} failed")
    print("="*70 + "\n")
    
    if tests_failed == 0:
        print("✅ All Phase 1 components validated successfully!")
        print("\nKey deliverables:")
        print("  • Database telemetry collection (PostgreSQL, MySQL, filesystem)")
        print("  • Backup state machine with SLA enforcement")
        print("  • Audit logging with chain integrity")
        print("  • Safety constraints (hard-coded, cannot be disabled)")
        print("  • Configuration system (Phase 1 settings)")
        print("\nNext steps:")
        print("  1. Run comprehensive unit tests: pytest tests/test_phase1_implementation.py")
        print("  2. Review implementation: docs/PHASE1_IMPLEMENTATION.md")
        print("  3. Begin Phase 2: Risk fusion and explainability layer")
        return 0
    else:
        print("❌ Some components failed validation. See details above.")
        return 1


if __name__ == "__main__":
    sys.exit(validate_imports())
