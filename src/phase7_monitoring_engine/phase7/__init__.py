"""Phase 7 monitoring engine package -- SOLID-architected pipeline.

Public API
----------
BaseMonitor                -- abstract monitor interface
MonitorStatus              -- monitor status dataclass
Phase7Config               -- pydantic-validated configuration
EngineEntry                -- engine registration model
EngineState                -- 5-state health enum
AlertSeverity              -- alert severity levels
AlertCategory              -- alert source categories
StateChangeEvent           -- state transition record
PerformanceSnapshot        -- point-in-time metrics
SecurityEvent              -- security event record
MonitoringAlert            -- generated alert
StateMachine               -- async state machine with Lock
AlertDispatcher            -- unified alert routing
StateChangeLogger          -- circular buffer logger
HeartbeatReceiver          -- async heartbeat processing
PerformanceCollector       -- rolling metric collection
ArtifactIntegrityChecker   -- SHA-256 verification
SecurityMonitor            -- security alert coordination
DashboardReporter          -- simulated WebSocket push
MonitoringExporter         -- 4-file artifact export
MonitoringPipeline         -- asyncio.gather orchestrator
render_monitoring_report   -- section 10.1 Markdown generator
"""

from .alert_dispatcher import AlertDispatcher
from .base import BaseMonitor, MonitorStatus
from .config import EngineEntry, Phase7Config
from .dashboard_reporter import DashboardReporter
from .exporter import MonitoringExporter
from .heartbeat_receiver import HeartbeatReceiver
from .integrity_checker import ArtifactIntegrityChecker
from .performance_collector import PerformanceCollector
from .pipeline import MonitoringPipeline, _detect_hardware, _get_git_commit
from .report import render_monitoring_report
from .security_monitor import SecurityMonitor
from .state_change_logger import StateChangeLogger
from .state_machine import (
    AlertCategory,
    AlertSeverity,
    EngineState,
    MonitoringAlert,
    PerformanceSnapshot,
    SecurityEvent,
    StateChangeEvent,
    StateMachine,
)

__all__ = [
    "BaseMonitor",
    "MonitorStatus",
    "Phase7Config",
    "EngineEntry",
    "EngineState",
    "AlertSeverity",
    "AlertCategory",
    "StateChangeEvent",
    "PerformanceSnapshot",
    "SecurityEvent",
    "MonitoringAlert",
    "StateMachine",
    "AlertDispatcher",
    "StateChangeLogger",
    "HeartbeatReceiver",
    "PerformanceCollector",
    "ArtifactIntegrityChecker",
    "SecurityMonitor",
    "DashboardReporter",
    "MonitoringExporter",
    "MonitoringPipeline",
    "render_monitoring_report",
    "_detect_hardware",
    "_get_git_commit",
]
