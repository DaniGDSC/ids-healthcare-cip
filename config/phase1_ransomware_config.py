"""Phase 1 configuration: Database telemetry and state machine.

Settings for:
- Database telemetry collection (PostgreSQL/MySQL)
- Backup state machine
- Audit logging
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional


# Database telemetry configuration
TELEMETRY_CONFIG = {
    # PostgreSQL read-only replica
    "postgresql": {
        "enabled": True,
        "replica_host": "db-replica.internal",
        "replica_port": 5432,
        "replica_database": "clinical_db",
        "replica_user": "readonly_user",  # Read-only role
        "collection_interval_sec": 60,     # Collect every 60 seconds
        "queries_per_interval": 100,       # Sample first 100 queries
        "timeout_sec": 10,
    },
    
    # MySQL read-only replica
    "mysql": {
        "enabled": False,  # Enable only if MySQL is used
        "replica_host": "mysql-replica.internal",
        "replica_port": 3306,
        "replica_database": "clinical_db",
        "replica_user": "readonly_user",
        "collection_interval_sec": 60,
        "queries_per_interval": 100,
        "timeout_sec": 10,
    }
}


# Transaction log monitoring configuration
TRANSACTION_LOG_CONFIG = {
    "enabled": True,
    
    # PostgreSQL WAL streaming
    "postgresql": {
        "enabled": True,
        "stream_url": "postgresql+replication://localhost/clinical_db",
        "baseline_update_rate": 100,      # Updates per minute (normal)
        "suspicious_threshold": 1000,      # 10x baseline = SUSPICIOUS
        "critical_threshold": 5000,        # 50x baseline = CRITICAL
        "window_sec": 60,                  # Analyze per 60-second window
    },
    
    # MySQL binlog streaming
    "mysql": {
        "enabled": False,
        "binlog_stream_url": "mysql+replication://localhost/clinical_db",
        "baseline_update_rate": 80,
        "suspicious_threshold": 800,
        "critical_threshold": 4000,
        "window_sec": 60,
    }
}


# Filesystem monitoring configuration
FILESYSTEM_CONFIG = {
    "enabled": True,
    
    # Database data directories to monitor
    "directories": [
        {
            "path": "/var/lib/postgresql/main",
            "database": "postgresql",
            "patterns": ["*.mdf", "*.ldf", "*"]
        },
        {
            "path": "/var/lib/mysql",
            "database": "mysql",
            "patterns": ["*.ibd", "*.frm", "*"]
        }
    ],
    
    # Entropy thresholds for encryption detection
    "entropy": {
        "baseline": 7.0,                   # Normal database file entropy
        "suspicious_threshold": 7.8,       # Suspicious encryption
        "critical_threshold": 7.95,        # Definite encryption
        "sample_size_bytes": 65536,        # Read first 64KB for analysis
    },
    
    # File modification monitoring
    "modification": {
        "check_interval_sec": 300,         # Check every 5 minutes
        "rapid_modification_threshold": 100,  # % change in 5 min = suspicious
    }
}


# Backup state machine configuration
STATE_MACHINE_CONFIG = {
    # Backup state definitions
    "states": {
        "NORMAL": {
            "description": "Standard operations, normal risk level",
            "backup_frequency": "hourly",
            "retention_days": 30,
        },
        "ELEVATED": {
            "description": "Suspicious signal detected, increased monitoring",
            "backup_frequency": "every_15_minutes",
            "retention_days": 90,
        },
        "SUSPICIOUS": {
            "description": "High-confidence ransomware, pending human review",
            "backup_frequency": "every_5_minutes",
            "retention_days": 180,
        },
        "QUARANTINED": {
            "description": "Isolated for forensics and investigation",
            "backup_frequency": "hourly_immutable",
            "retention_days": 180,
        },
        "TRUSTED": {
            "description": "Validated safe, golden restore point",
            "backup_frequency": "never",
            "retention_days": 365,  # Keep for 1 year
        }
    },
    
    # State transition SLAs (minutes until escalation)
    "escalation_slas": {
        "NORMAL": 1440,           # 24 hours (informational escalation)
        "ELEVATED": 240,          # 4 hours (wait for next risk window)
        "SUSPICIOUS": 15,         # 15 minutes (urgent decision needed)
        "QUARANTINED": 60,        # 1 hour (for investigation)
    },
    
    # Transitions requiring human approval
    "human_approval_required": [
        ("ELEVATED", "SUSPICIOUS"),
        ("SUSPICIOUS", "QUARANTINED"),
        ("QUARANTINED", "TRUSTED"),
        ("QUARANTINED", "NORMAL"),
    ]
}


# Audit logging configuration
AUDIT_CONFIG = {
    "enabled": True,
    "storage_type": "file",  # "file" or "database"
    
    # File-based storage
    "file": {
        "log_file": "logs/audit/ransomware_audit.log",
        "max_file_size_mb": 1024,
        "rotation_count": 10,  # Keep 10 rotated files
    },
    
    # Database storage (PostgreSQL)
    "database": {
        "enabled": False,
        "host": "localhost",
        "port": 5432,
        "database": "audit_db",
        "table": "audit_log",
        "user": "audit_logger",
    },
    
    # Retention policy (HIPAA requirement)
    "retention": {
        "days": 2555,  # 7 years for HIPAA compliance
        "archive_after_days": 365,  # Archive to cold storage after 1 year
    },
    
    # Event categories to log
    "event_types": [
        "STATE_TRANSITION",
        "BACKUP_CREATED",
        "BACKUP_DELETED",
        "BACKUP_RESTORED",
        "BACKUP_VALIDATED",
        "DECISION_APPROVED",
        "DECISION_REJECTED",
        "ALERT_ESCALATED",
        "POLICY_VIOLATION",
        "SYSTEM_ERROR",
        "ACCESS_GRANTED",
        "ACCESS_DENIED",
    ],
    
    # Immutability enforcement
    "immutability": {
        "method": "database_trigger",  # "database_trigger" or "filesystem_permissions"
        "verify_chain_interval_hours": 24,  # Verify chain daily
    }
}


# Performance and resource configuration
PERFORMANCE_CONFIG = {
    # Telemetry collection
    "telemetry_collection": {
        "batch_size": 1000,        # Batch events
        "max_memory_mb": 512,      # Max memory for buffered events
        "flush_interval_sec": 300, # Flush buffer every 5 minutes
    },
    
    # State machine
    "state_machine": {
        "max_history_entries": 10000,  # Keep last 10K transitions
        "cleanup_interval_hours": 24,   # Cleanup old history daily
    },
    
    # Audit logging
    "audit_logging": {
        "async_enabled": True,          # Use async logging
        "batch_write_count": 100,       # Batch writes
        "batch_write_timeout_sec": 60,  # Or timeout
    }
}


# Safety constraints (hard-coded, not configurable)
SAFETY_CONSTRAINTS = {
    "automated_actions": {
        "auto_delete_backup": False,      # ✅ NEVER automate
        "auto_restore_backup": False,     # ✅ NEVER automate
        "auto_overwrite_backup": False,   # ✅ NEVER automate
        "auto_modify_backup": False,      # ✅ NEVER automate
        "auto_bypass_approval": False,    # ✅ NEVER automate
    },
    
    "mandatory_gates": {
        "human_review_required": True,                # Mandatory
        "approval_for_critical": True,                # Mandatory
        "patient_care_override": True,                # Clinical priority
        "incident_response_timeout": 15,              # Minutes, mandatory
    },
    
    "immutability": {
        "audit_log_deletable": False,     # ✅ Audit cannot be deleted
        "audit_log_modifiable": False,    # ✅ Audit cannot be modified
        "trusted_backups_deletable": False,  # ✅ Golden points permanent
        "state_history_deletable": False,    # ✅ History permanent
    }
}


def load_phase1_config() -> Dict[str, Any]:
    """Load Phase 1 configuration."""
    return {
        "telemetry": TELEMETRY_CONFIG,
        "transaction_log": TRANSACTION_LOG_CONFIG,
        "filesystem": FILESYSTEM_CONFIG,
        "state_machine": STATE_MACHINE_CONFIG,
        "audit": AUDIT_CONFIG,
        "performance": PERFORMANCE_CONFIG,
        "safety_constraints": SAFETY_CONSTRAINTS,
    }


def get_config_summary() -> str:
    """Get human-readable configuration summary."""
    return f"""
Phase 1 Configuration Summary
=============================

1. Database Telemetry:
   - PostgreSQL replica: {TELEMETRY_CONFIG['postgresql']['enabled']}
   - MySQL replica: {TELEMETRY_CONFIG['mysql']['enabled']}
   - Collection interval: {TELEMETRY_CONFIG['postgresql']['collection_interval_sec']}s

2. Transaction Log Monitoring:
   - Enabled: {TRANSACTION_LOG_CONFIG['enabled']}
   - Baseline update rate: {TRANSACTION_LOG_CONFIG['postgresql']['baseline_update_rate']} updates/min
   - Suspicious threshold: {TRANSACTION_LOG_CONFIG['postgresql']['suspicious_threshold']} updates/min
   - Critical threshold: {TRANSACTION_LOG_CONFIG['postgresql']['critical_threshold']} updates/min

3. Filesystem Monitoring:
   - Enabled: {FILESYSTEM_CONFIG['enabled']}
   - Monitored directories: {len(FILESYSTEM_CONFIG['directories'])}
   - Entropy baseline: {FILESYSTEM_CONFIG['entropy']['baseline']}
   - Suspicious entropy threshold: {FILESYSTEM_CONFIG['entropy']['suspicious_threshold']}

4. State Machine:
   - States: {list(STATE_MACHINE_CONFIG['states'].keys())}
   - SLA for SUSPICIOUS: {STATE_MACHINE_CONFIG['escalation_slas']['SUSPICIOUS']} minutes
   - SLA for QUARANTINED: {STATE_MACHINE_CONFIG['escalation_slas']['QUARANTINED']} minutes

5. Audit Logging:
   - Storage type: {AUDIT_CONFIG['storage_type']}
   - Retention: {AUDIT_CONFIG['retention']['days']} days (7 years HIPAA)
   - Immutability: Database trigger

6. Safety Constraints:
   - Automated delete backup: {SAFETY_CONSTRAINTS['automated_actions']['auto_delete_backup']}
   - Automated restore backup: {SAFETY_CONSTRAINTS['automated_actions']['auto_restore_backup']}
   - Human review required: {SAFETY_CONSTRAINTS['mandatory_gates']['human_review_required']}
   - Patient care override: {SAFETY_CONSTRAINTS['mandatory_gates']['patient_care_override']}
"""


if __name__ == "__main__":
    print(get_config_summary())
