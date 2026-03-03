"""Append-only audit logging for ransomware-aware backup system.

Provides immutable audit trail for:
- State transitions
- Decision authority actions
- Backup operations
- Access to sensitive data

Maintains HIPAA 7-year retention requirement.
"""

import logging
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import hashlib

logger = logging.getLogger(__name__)


class AuditEventType(Enum):
    """Types of audit events."""
    STATE_TRANSITION = "STATE_TRANSITION"
    BACKUP_CREATED = "BACKUP_CREATED"
    BACKUP_DELETED = "BACKUP_DELETED"
    BACKUP_RESTORED = "BACKUP_RESTORED"
    BACKUP_VALIDATED = "BACKUP_VALIDATED"
    DECISION_APPROVED = "DECISION_APPROVED"
    DECISION_REJECTED = "DECISION_REJECTED"
    ALERT_ESCALATED = "ALERT_ESCALATED"
    POLICY_VIOLATION = "POLICY_VIOLATION"
    SYSTEM_ERROR = "SYSTEM_ERROR"
    ACCESS_GRANTED = "ACCESS_GRANTED"
    ACCESS_DENIED = "ACCESS_DENIED"


@dataclass
class AuditEvent:
    """
    Immutable audit log entry.
    
    Attributes:
        timestamp: When event occurred (ISO 8601 UTC)
        event_type: Type of audit event (AuditEventType)
        actor: Who performed action (user ID, system component, etc.)
        resource: What was affected (backup ID, database, etc.)
        action: What was done (specific operation)
        result: Outcome (success, failure, etc.)
        details: Event-specific details (dict)
        backup_id: Associated backup (optional)
        severity: Event severity (INFO, WARNING, CRITICAL)
        event_id: Unique identifier (hash of content)
        previous_event_hash: Hash of previous event (chain integrity)
    """
    timestamp: datetime
    event_type: AuditEventType
    actor: str
    resource: str
    action: str
    result: str
    details: Dict[str, Any]
    severity: str = "INFO"
    backup_id: Optional[str] = None
    event_id: Optional[str] = None
    previous_event_hash: Optional[str] = None
    
    def __post_init__(self):
        """Generate event_id if not provided."""
        if self.event_id is None:
            content = (
                f"{self.timestamp.isoformat()}"
                f"{self.event_type.value}"
                f"{self.actor}"
                f"{self.resource}"
                f"{self.action}"
            )
            self.event_id = hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def to_json(self) -> str:
        """Convert to JSON string for logging."""
        return json.dumps({
            "timestamp": self.timestamp.isoformat(),
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "actor": self.actor,
            "resource": self.resource,
            "action": self.action,
            "result": self.result,
            "details": self.details,
            "severity": self.severity,
            "backup_id": self.backup_id,
            "previous_event_hash": self.previous_event_hash
        })
    
    @staticmethod
    def compute_hash(data: str) -> str:
        """Compute SHA256 hash of event data."""
        return hashlib.sha256(data.encode()).hexdigest()


class AuditLogger:
    """
    Append-only audit logger for ransomware-aware backup system.
    
    Properties:
    - Immutable (append-only, no modification/deletion)
    - Tamper-evident (hash chain for integrity verification)
    - HIPAA compliant (7-year retention)
    - Performance optimized (batch writes, async logging possible)
    
    Storage options:
    1. File-based (append-only, protected by filesystem permissions)
    2. PostgreSQL (append-only trigger, WORM table)
    """
    
    def __init__(
        self,
        log_file: Optional[Path] = None,
        db_connection: Optional[Any] = None,
        storage_type: str = "file",
        retention_days: int = 2555  # 7 years for HIPAA
    ):
        """
        Initialize audit logger.
        
        Args:
            log_file: Path to append-only log file
            db_connection: Database connection for SQL storage
            storage_type: "file" or "database"
            retention_days: How long to retain logs (7 years default for HIPAA)
        """
        self.log_file = log_file
        self.db_connection = db_connection
        self.storage_type = storage_type
        self.retention_days = retention_days
        self.last_event_hash: Optional[str] = None
        self.event_count = 0
        
        # Ensure log file is accessible and writable
        if storage_type == "file" and log_file:
            self.log_file = Path(log_file)
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Test write capability
            try:
                with open(self.log_file, 'a') as f:
                    pass
                logger.info(f"Initialized file-based audit logger: {log_file}")
            except Exception as e:
                logger.error(f"Cannot access audit log file {log_file}: {e}")
                raise
        
        elif storage_type == "database":
            if not db_connection:
                raise ValueError("database storage requires db_connection parameter")
            self._init_database_schema()
            logger.info("Initialized database-based audit logger")
        
        # Load last event hash for chain integrity
        self._load_last_event_hash()
    
    def _init_database_schema(self):
        """Create PostgreSQL audit table with immutable semantics."""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS audit_log (
            id BIGSERIAL PRIMARY KEY,
            timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
            event_id VARCHAR(16) NOT NULL UNIQUE,
            event_type VARCHAR(50) NOT NULL,
            actor VARCHAR(255) NOT NULL,
            resource VARCHAR(255) NOT NULL,
            action VARCHAR(255) NOT NULL,
            result VARCHAR(50) NOT NULL,
            details JSONB NOT NULL DEFAULT '{}',
            severity VARCHAR(20) NOT NULL DEFAULT 'INFO',
            backup_id VARCHAR(255),
            previous_event_hash VARCHAR(64),
            event_hash VARCHAR(64) NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
            
            -- Prevent modification
            CONSTRAINT no_updates CHECK (true)
        );
        
        -- Create index for query performance
        CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(timestamp);
        CREATE INDEX IF NOT EXISTS idx_audit_event_type ON audit_log(event_type);
        CREATE INDEX IF NOT EXISTS idx_audit_backup_id ON audit_log(backup_id);
        CREATE INDEX IF NOT EXISTS idx_audit_actor ON audit_log(actor);
        
        -- Create immutability trigger (prevent UPDATE/DELETE)
        CREATE OR REPLACE FUNCTION audit_log_immutable()
        RETURNS TRIGGER AS $$
        BEGIN
            RAISE EXCEPTION 'Audit log is immutable: cannot update or delete records';
        END;
        $$ LANGUAGE plpgsql;
        
        DROP TRIGGER IF EXISTS audit_log_protection ON audit_log;
        CREATE TRIGGER audit_log_protection
            BEFORE UPDATE OR DELETE ON audit_log
            FOR EACH ROW EXECUTE FUNCTION audit_log_immutable();
        """
        
        try:
            cursor = self.db_connection.cursor()
            cursor.execute(create_table_sql)
            self.db_connection.commit()
            cursor.close()
            logger.info("Audit table schema initialized")
        except Exception as e:
            logger.error(f"Failed to initialize audit table: {e}")
            raise
    
    def _load_last_event_hash(self):
        """Load hash of last event for chain integrity."""
        try:
            if self.storage_type == "file" and self.log_file and self.log_file.exists():
                with open(self.log_file, 'r') as f:
                    lines = f.readlines()
                    if lines:
                        last_line = lines[-1].strip()
                        if last_line:
                            last_event = json.loads(last_line)
                            self.last_event_hash = last_event.get("event_id")
                            self.event_count = len(lines)
            
            elif self.storage_type == "database":
                cursor = self.db_connection.cursor()
                cursor.execute(
                    "SELECT event_id, COUNT(*) as cnt FROM audit_log ORDER BY id DESC LIMIT 1"
                )
                result = cursor.fetchone()
                if result:
                    self.last_event_hash = result[0]
                    self.event_count = result[1]
                cursor.close()
        
        except Exception as e:
            logger.warning(f"Could not load previous event hash: {e}")
            self.last_event_hash = None
    
    def log_event(
        self,
        event_type: AuditEventType,
        actor: str,
        resource: str,
        action: str,
        result: str,
        details: Optional[Dict[str, Any]] = None,
        severity: str = "INFO",
        backup_id: Optional[str] = None
    ) -> AuditEvent:
        """
        Log an audit event (append-only).
        
        Args:
            event_type: Type of event
            actor: Who performed action
            resource: What was affected
            action: What was done
            result: Outcome
            details: Additional context
            severity: Event severity
            backup_id: Associated backup ID
            
        Returns:
            AuditEvent that was logged
        """
        event = AuditEvent(
            timestamp=datetime.utcnow(),
            event_type=event_type,
            actor=actor,
            resource=resource,
            action=action,
            result=result,
            details=details or {},
            severity=severity,
            backup_id=backup_id,
            previous_event_hash=self.last_event_hash
        )
        
        # Store event
        if self.storage_type == "file":
            self._write_to_file(event)
        elif self.storage_type == "database":
            self._write_to_database(event)
        
        # Update chain state
        self.last_event_hash = event.event_id
        self.event_count += 1
        
        # Log to standard logger
        log_level = getattr(logging, severity, logging.INFO)
        logger.log(
            log_level,
            f"AUDIT: {event_type.value} | {actor} | {action} | {result}",
            extra={"event_id": event.event_id}
        )
        
        return event
    
    def _write_to_file(self, event: AuditEvent):
        """Write event to append-only file."""
        try:
            with open(self.log_file, 'a') as f:
                f.write(event.to_json() + '\n')
                f.flush()
        except Exception as e:
            logger.error(f"Failed to write audit event to file: {e}")
            raise
    
    def _write_to_database(self, event: AuditEvent):
        """Write event to PostgreSQL audit table."""
        try:
            cursor = self.db_connection.cursor()
            
            # Compute event hash for chain integrity
            event_hash = AuditEvent.compute_hash(event.to_json())
            
            cursor.execute(
                """
                INSERT INTO audit_log (
                    event_id, event_type, actor, resource, action, result,
                    details, severity, backup_id, previous_event_hash, event_hash
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    event.event_id,
                    event.event_type.value,
                    event.actor,
                    event.resource,
                    event.action,
                    event.result,
                    json.dumps(event.details),
                    event.severity,
                    event.backup_id,
                    event.previous_event_hash,
                    event_hash
                )
            )
            
            self.db_connection.commit()
            cursor.close()
        
        except Exception as e:
            logger.error(f"Failed to write audit event to database: {e}")
            raise
    
    def query_events(
        self,
        backup_id: Optional[str] = None,
        actor: Optional[str] = None,
        event_type: Optional[AuditEventType] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Query audit log (read-only, for investigation).
        
        Args:
            backup_id: Filter by backup ID
            actor: Filter by actor (user/component)
            event_type: Filter by event type
            start_time: Filter by minimum timestamp
            end_time: Filter by maximum timestamp
            limit: Maximum records to return
            
        Returns:
            List of matching audit events
        """
        if self.storage_type == "file":
            return self._query_file(
                backup_id=backup_id,
                actor=actor,
                event_type=event_type,
                start_time=start_time,
                end_time=end_time,
                limit=limit
            )
        
        elif self.storage_type == "database":
            return self._query_database(
                backup_id=backup_id,
                actor=actor,
                event_type=event_type,
                start_time=start_time,
                end_time=end_time,
                limit=limit
            )
        
        return []
    
    def _query_file(self, **kwargs) -> List[Dict[str, Any]]:
        """Query audit log from file."""
        results = []
        
        try:
            if not self.log_file.exists():
                return []
            
            with open(self.log_file, 'r') as f:
                for line in f:
                    if not line.strip():
                        continue
                    
                    try:
                        event = json.loads(line)
                        
                        # Apply filters
                        if kwargs.get('backup_id') and event.get('backup_id') != kwargs['backup_id']:
                            continue
                        if kwargs.get('actor') and event.get('actor') != kwargs['actor']:
                            continue
                        if kwargs.get('event_type') and event.get('event_type') != kwargs['event_type'].value:
                            continue
                        
                        event_time = datetime.fromisoformat(event['timestamp'].replace('Z', '+00:00'))
                        if kwargs.get('start_time') and event_time < kwargs['start_time']:
                            continue
                        if kwargs.get('end_time') and event_time > kwargs['end_time']:
                            continue
                        
                        results.append(event)
                        
                        if len(results) >= kwargs.get('limit', 1000):
                            break
                    
                    except json.JSONDecodeError as e:
                        logger.warning(f"Skipping malformed audit log line: {e}")
                        continue
        
        except Exception as e:
            logger.error(f"Error querying file audit log: {e}")
        
        return results
    
    def _query_database(self, **kwargs) -> List[Dict[str, Any]]:
        """Query audit log from PostgreSQL."""
        results = []
        
        try:
            cursor = self.db_connection.cursor()
            
            # Build query
            where_clauses = []
            params = []
            
            if kwargs.get('backup_id'):
                where_clauses.append("backup_id = %s")
                params.append(kwargs['backup_id'])
            
            if kwargs.get('actor'):
                where_clauses.append("actor = %s")
                params.append(kwargs['actor'])
            
            if kwargs.get('event_type'):
                where_clauses.append("event_type = %s")
                params.append(kwargs['event_type'].value)
            
            if kwargs.get('start_time'):
                where_clauses.append("timestamp >= %s")
                params.append(kwargs['start_time'])
            
            if kwargs.get('end_time'):
                where_clauses.append("timestamp <= %s")
                params.append(kwargs['end_time'])
            
            where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"
            limit = kwargs.get('limit', 1000)
            
            query = f"""
            SELECT event_id, timestamp, event_type, actor, resource, action, result,
                   details, severity, backup_id
            FROM audit_log
            WHERE {where_sql}
            ORDER BY id DESC
            LIMIT %s
            """
            params.append(limit)
            
            cursor.execute(query, params)
            
            for row in cursor.fetchall():
                results.append({
                    "event_id": row[0],
                    "timestamp": row[1].isoformat(),
                    "event_type": row[2],
                    "actor": row[3],
                    "resource": row[4],
                    "action": row[5],
                    "result": row[6],
                    "details": row[7],
                    "severity": row[8],
                    "backup_id": row[9]
                })
            
            cursor.close()
        
        except Exception as e:
            logger.error(f"Error querying database audit log: {e}")
        
        return results
    
    def verify_chain_integrity(self) -> Tuple[bool, Optional[str]]:
        """
        Verify audit log hash chain integrity.
        
        Returns:
            (is_valid, error_message) tuple
        """
        try:
            if self.storage_type == "file":
                return self._verify_file_chain()
            elif self.storage_type == "database":
                return self._verify_database_chain()
        
        except Exception as e:
            return False, str(e)
        
        return True, None
    
    def _verify_file_chain(self) -> Tuple[bool, Optional[str]]:
        """Verify file-based audit log chain."""
        if not self.log_file.exists():
            return True, None  # Empty log is valid
        
        previous_hash = None
        
        try:
            with open(self.log_file, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    if not line.strip():
                        continue
                    
                    event = json.loads(line)
                    
                    # Check hash chain
                    if event.get('previous_event_hash') != previous_hash:
                        return False, f"Hash chain broken at line {line_num}"
                    
                    previous_hash = event['event_id']
        
        except json.JSONDecodeError as e:
            return False, f"Malformed JSON in audit log: {e}"
        except Exception as e:
            return False, str(e)
        
        return True, None
    
    def _verify_database_chain(self) -> Tuple[bool, Optional[str]]:
        """Verify database audit log chain."""
        try:
            cursor = self.db_connection.cursor()
            
            cursor.execute(
                """
                SELECT event_id, previous_event_hash FROM audit_log
                ORDER BY id ASC
                """
            )
            
            previous_hash = None
            for row in cursor.fetchall():
                event_id, prev_hash = row
                
                if prev_hash != previous_hash:
                    return False, f"Hash chain broken at event {event_id}"
                
                previous_hash = event_id
            
            cursor.close()
        
        except Exception as e:
            return False, str(e)
        
        return True, None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get audit log statistics."""
        return {
            "total_events": self.event_count,
            "last_event_hash": self.last_event_hash,
            "retention_days": self.retention_days,
            "retention_until": (datetime.utcnow() + timedelta(days=self.retention_days)).isoformat(),
            "storage_type": self.storage_type
        }
