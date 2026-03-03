"""Database telemetry collection for ransomware-aware backup system.

Collects database metrics from read-only replicas:
- Query patterns and execution statistics
- Connection metadata and authentication events
- Transaction log activity (WAL/binlog)
- File system events on database directories
"""

import logging
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import json

logger = logging.getLogger(__name__)


class TelemetryEventType(Enum):
    """Types of telemetry events."""
    QUERY_PATTERN = "QUERY_PATTERN"
    CONNECTION_METADATA = "CONNECTION_METADATA"
    TRANSACTION_LOG = "TRANSACTION_LOG"
    FILE_MODIFICATION = "FILE_MODIFICATION"
    BACKUP_JOB = "BACKUP_JOB"
    AUTHENTICATION = "AUTHENTICATION"


class TelemetrySeverity(Enum):
    """Severity levels for telemetry events."""
    INFO = "INFO"
    WARNING = "WARNING"
    SUSPICIOUS = "SUSPICIOUS"
    CRITICAL = "CRITICAL"


@dataclass
class TelemetryEvent:
    """Standard telemetry event format.
    
    Attributes:
        timestamp: When event occurred
        source: Source of telemetry ('database', 'filesystem', 'backup', 'network')
        event_type: Type of event (TelemetryEventType)
        severity: Severity level (TelemetrySeverity)
        metadata: Source-specific details (dict)
        anonymized: Whether PHI has been removed (bool)
        event_id: Unique identifier for this event (str, optional)
    """
    timestamp: datetime
    source: str
    event_type: TelemetryEventType
    severity: TelemetrySeverity
    metadata: Dict[str, Any]
    anonymized: bool = True
    event_id: Optional[str] = None
    
    def __post_init__(self):
        """Generate event_id if not provided."""
        if self.event_id is None:
            content = f"{self.timestamp}{self.source}{self.event_type}{self.metadata}"
            self.event_id = hashlib.sha256(content.encode()).hexdigest()[:16]


class DBTelemetryCollector:
    """
    Collect database telemetry from read-only replica.
    
    Prevents performance impact on production DB.
    Collects anonymized query patterns (no PHI exposure).
    
    Attributes:
        source_type: Type of database ('postgresql' or 'mysql')
        replica_conn_string: Connection string for read-only replica
        collection_interval_sec: How often to collect telemetry (seconds)
        anonymizer: HIPAACompliance instance for anonymization
    """
    
    def __init__(
        self,
        source_type: str,
        replica_conn_string: str,
        collection_interval_sec: int = 60,
        anonymizer: Optional[Any] = None
    ):
        """
        Initialize database telemetry collector.
        
        Args:
            source_type: 'postgresql' or 'mysql'
            replica_conn_string: Connection to read-only replica
            collection_interval_sec: Telemetry collection interval
            anonymizer: HIPAACompliance instance (optional)
        """
        self.source_type = source_type
        self.replica_conn_string = replica_conn_string
        self.collection_interval_sec = collection_interval_sec
        self.anonymizer = anonymizer
        self.last_collection = datetime.utcnow()
        
        logger.info(f"Initialized {source_type} telemetry collector")
    
    def collect_query_patterns(self) -> List[TelemetryEvent]:
        """
        Collect aggregated query patterns (anonymized).
        
        Returns:
            List of TelemetryEvent objects with query statistics
        """
        events = []
        
        try:
            if self.source_type == 'postgresql':
                events = self._collect_postgresql_stats()
            elif self.source_type == 'mysql':
                events = self._collect_mysql_stats()
            else:
                logger.warning(f"Unsupported database type: {self.source_type}")
                return []
            
            # Log collection
            logger.debug(f"Collected {len(events)} query pattern events")
            return events
            
        except Exception as e:
            logger.error(f"Failed to collect query patterns: {e}")
            return []
    
    def collect_connection_metadata(self) -> List[TelemetryEvent]:
        """
        Collect connection patterns (no PHI).
        
        Returns:
            List of TelemetryEvent objects with connection metadata
        """
        events = []
        
        try:
            if self.source_type == 'postgresql':
                events = self._collect_postgresql_connections()
            elif self.source_type == 'mysql':
                events = self._collect_mysql_connections()
            else:
                logger.warning(f"Unsupported database type: {self.source_type}")
                return []
            
            logger.debug(f"Collected {len(events)} connection metadata events")
            return events
            
        except Exception as e:
            logger.error(f"Failed to collect connection metadata: {e}")
            return []
    
    def collect_authentication_events(self) -> List[TelemetryEvent]:
        """
        Collect authentication events (failed logins, privilege escalation).
        
        Returns:
            List of TelemetryEvent objects with authentication data
        """
        events = []
        
        try:
            if self.source_type == 'postgresql':
                events = self._collect_postgresql_auth()
            elif self.source_type == 'mysql':
                events = self._collect_mysql_auth()
            else:
                logger.warning(f"Unsupported database type: {self.source_type}")
                return []
            
            logger.debug(f"Collected {len(events)} authentication events")
            return events
            
        except Exception as e:
            logger.error(f"Failed to collect authentication events: {e}")
            return []
    
    def _collect_postgresql_stats(self) -> List[TelemetryEvent]:
        """Collect PostgreSQL query statistics from pg_stat_statements."""
        events = []
        
        # Note: Actual implementation requires psycopg2 and database connection
        # Placeholder returns example event structure
        
        event = TelemetryEvent(
            timestamp=datetime.utcnow(),
            source="database",
            event_type=TelemetryEventType.QUERY_PATTERN,
            severity=TelemetrySeverity.INFO,
            metadata={
                "database": "postgresql",
                "query_hash": "hash_value",  # Anonymized
                "execution_count": 150,
                "avg_duration_ms": 45.2,
                "error_count": 2,
                "affected_tables": ["table1", "table2"],  # No column-level details
            },
            anonymized=True
        )
        
        events.append(event)
        return events
    
    def _collect_postgresql_connections(self) -> List[TelemetryEvent]:
        """Collect PostgreSQL connection metadata."""
        events = []
        
        event = TelemetryEvent(
            timestamp=datetime.utcnow(),
            source="database",
            event_type=TelemetryEventType.CONNECTION_METADATA,
            severity=TelemetrySeverity.INFO,
            metadata={
                "database": "postgresql",
                "source_ip_hash": "hash_of_ip",  # Anonymized
                "user_role": "app_user",  # Role, not specific user
                "connection_time": datetime.utcnow().isoformat(),
                "authentication_method": "md5",
                "database_name": "clinical_db"  # Schema-level only
            },
            anonymized=True
        )
        
        events.append(event)
        return events
    
    def _collect_postgresql_auth(self) -> List[TelemetryEvent]:
        """Collect PostgreSQL authentication events."""
        events = []
        
        # Example: Failed authentication attempt
        event = TelemetryEvent(
            timestamp=datetime.utcnow(),
            source="database",
            event_type=TelemetryEventType.AUTHENTICATION,
            severity=TelemetrySeverity.WARNING,
            metadata={
                "database": "postgresql",
                "auth_status": "FAILED",
                "failure_reason": "INVALID_CREDENTIALS",
                "source_ip_hash": "hash_of_ip",
                "attempt_count": 3,  # 3 failed attempts
                "user_role": "dba",  # Anonymous role
                "timestamp": datetime.utcnow().isoformat()
            },
            anonymized=True
        )
        
        events.append(event)
        return events
    
    def _collect_mysql_stats(self) -> List[TelemetryEvent]:
        """Collect MySQL query statistics from performance_schema."""
        events = []
        
        event = TelemetryEvent(
            timestamp=datetime.utcnow(),
            source="database",
            event_type=TelemetryEventType.QUERY_PATTERN,
            severity=TelemetrySeverity.INFO,
            metadata={
                "database": "mysql",
                "query_hash": "hash_value",  # Anonymized
                "execution_count": 120,
                "avg_duration_ms": 32.1,
                "rows_examined": 50000,
                "affected_tables": ["patients", "encounters"]
            },
            anonymized=True
        )
        
        events.append(event)
        return events
    
    def _collect_mysql_connections(self) -> List[TelemetryEvent]:
        """Collect MySQL connection metadata."""
        events = []
        
        event = TelemetryEvent(
            timestamp=datetime.utcnow(),
            source="database",
            event_type=TelemetryEventType.CONNECTION_METADATA,
            severity=TelemetrySeverity.INFO,
            metadata={
                "database": "mysql",
                "source_ip_hash": "hash_of_ip",
                "user_role": "app_user",
                "connection_time": datetime.utcnow().isoformat(),
                "authentication_method": "native",
                "database_name": "clinical_db"
            },
            anonymized=True
        )
        
        events.append(event)
        return events
    
    def _collect_mysql_auth(self) -> List[TelemetryEvent]:
        """Collect MySQL authentication events."""
        events = []
        
        event = TelemetryEvent(
            timestamp=datetime.utcnow(),
            source="database",
            event_type=TelemetryEventType.AUTHENTICATION,
            severity=TelemetrySeverity.INFO,
            metadata={
                "database": "mysql",
                "auth_status": "SUCCESS",
                "source_ip_hash": "hash_of_ip",
                "user_role": "app_user",
                "timestamp": datetime.utcnow().isoformat()
            },
            anonymized=True
        )
        
        events.append(event)
        return events


class TransactionLogMonitor:
    """
    Monitor database transaction logs for anomalous activity.
    
    Detects ransomware indicators:
    - Rapid mass UPDATE/DELETE operations
    - Suspicious schema changes (DROP TABLE, TRUNCATE)
    - Backup table manipulation
    """
    
    def __init__(self, source_type: str, wal_stream_url: Optional[str] = None):
        """
        Initialize transaction log monitor.
        
        Args:
            source_type: 'postgresql' or 'mysql'
            wal_stream_url: URL for WAL/binlog streaming (optional)
        """
        self.source_type = source_type
        self.wal_stream_url = wal_stream_url
        self.baseline_update_rate = 100  # Updates per minute (configurable)
        
        logger.info(f"Initialized transaction log monitor for {source_type}")
    
    def detect_mass_modification(
        self,
        window_sec: int = 60
    ) -> Optional[TelemetryEvent]:
        """
        Detect unusually high modification rate (ransomware indicator).
        
        Args:
            window_sec: Time window for analysis (seconds)
            
        Returns:
            TelemetryEvent if anomaly detected, None otherwise
        """
        # Placeholder: In production, would read from actual transaction log
        modification_rate = 1200  # Updates per minute in current window
        baseline = self.baseline_update_rate
        
        if modification_rate > baseline * 10:
            severity = TelemetrySeverity.CRITICAL if modification_rate > baseline * 50 else TelemetrySeverity.SUSPICIOUS
            
            event = TelemetryEvent(
                timestamp=datetime.utcnow(),
                source="database",
                event_type=TelemetryEventType.TRANSACTION_LOG,
                severity=severity,
                metadata={
                    "indicator": "MASS_MODIFICATION",
                    "update_count": modification_rate,
                    "baseline": baseline,
                    "ratio": modification_rate / baseline,
                    "affected_tables": ["patient_records", "clinical_observations"],
                    "window_sec": window_sec
                },
                anonymized=True
            )
            
            logger.warning(f"Mass modification detected: {modification_rate} updates/min (baseline: {baseline})")
            return event
        
        return None
    
    def detect_schema_changes(self) -> Optional[TelemetryEvent]:
        """
        Detect suspicious schema changes (DROP TABLE, TRUNCATE).
        
        Returns:
            TelemetryEvent if suspicious schema change detected
        """
        # Placeholder: In production, would read from actual transaction log
        # Example: DROP TABLE statement detected
        
        suspicious_changes = ["DROP TABLE", "TRUNCATE"]
        
        for change in suspicious_changes:
            event = TelemetryEvent(
                timestamp=datetime.utcnow(),
                source="database",
                event_type=TelemetryEventType.TRANSACTION_LOG,
                severity=TelemetrySeverity.CRITICAL,
                metadata={
                    "indicator": "SCHEMA_CHANGE",
                    "change_type": change,
                    "affected_table": "unknown_table",
                    "timestamp": datetime.utcnow().isoformat()
                },
                anonymized=True
            )
            
            logger.critical(f"Suspicious schema change detected: {change}")
            return event
        
        return None


class FilesystemMonitor:
    """
    Monitor database data directory for ransomware activity.
    
    Detects:
    - File extension changes (.mdf → .encrypted)
    - Rapid file modification (entropy increase)
    - Suspicious file creation (ransom notes)
    """
    
    def __init__(self, db_data_dir: Path, watch_patterns: Optional[List[str]] = None):
        """
        Initialize filesystem monitor.
        
        Args:
            db_data_dir: Path to database data directory
            watch_patterns: File patterns to monitor (e.g., ['*.mdf', '*.ibd'])
        """
        self.db_data_dir = Path(db_data_dir)
        self.watch_patterns = watch_patterns or ["*.mdf", "*.ibd", "*.ibdata*"]
        
        logger.info(f"Initialized filesystem monitor for {db_data_dir}")
    
    def calculate_entropy(self, file_path: Path) -> float:
        """
        Calculate Shannon entropy of file (indication of encryption).
        
        Encrypted files have entropy close to 8.0 (maximum).
        Normal database files: entropy ~6.5-7.5
        
        Args:
            file_path: Path to file
            
        Returns:
            Entropy value (0.0-8.0)
        """
        try:
            with open(file_path, 'rb') as f:
                data = f.read(65536)  # Read first 64KB
            
            if not data:
                return 0.0
            
            # Calculate frequency of each byte
            byte_counts = [0] * 256
            for byte in data:
                byte_counts[byte] += 1
            
            # Calculate Shannon entropy
            entropy = 0.0
            data_len = len(data)
            for count in byte_counts:
                if count > 0:
                    probability = count / data_len
                    entropy -= probability * (probability.bit_length() - 1)  # log2(p) approximation
            
            return entropy
            
        except Exception as e:
            logger.error(f"Failed to calculate entropy for {file_path}: {e}")
            return 0.0
    
    def detect_entropy_change(
        self,
        file_path: Path,
        baseline_entropy: float = 7.0
    ) -> Optional[TelemetryEvent]:
        """
        Detect encryption via entropy analysis.
        
        Args:
            file_path: Path to file to analyze
            baseline_entropy: Expected entropy for normal database file
            
        Returns:
            TelemetryEvent if encryption detected
        """
        current_entropy = self.calculate_entropy(file_path)
        delta = current_entropy - baseline_entropy
        
        # Encryption indicator: entropy > 7.8 and increase > 0.5
        if current_entropy > 7.8 and delta > 0.5:
            severity = TelemetrySeverity.CRITICAL if current_entropy > 7.95 else TelemetrySeverity.SUSPICIOUS
            
            event = TelemetryEvent(
                timestamp=datetime.utcnow(),
                source="filesystem",
                event_type=TelemetryEventType.FILE_MODIFICATION,
                severity=severity,
                metadata={
                    "indicator": "FILE_ENCRYPTION_DETECTED",
                    "file": str(file_path.relative_to(self.db_data_dir)),
                    "current_entropy": round(current_entropy, 3),
                    "baseline_entropy": round(baseline_entropy, 3),
                    "delta": round(delta, 3),
                    "file_size_mb": file_path.stat().st_size / (1024 * 1024)
                },
                anonymized=True
            )
            
            logger.critical(f"File encryption detected: {file_path} (entropy: {current_entropy:.2f})")
            return event
        
        return None
