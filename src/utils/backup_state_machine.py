"""Backup state machine for ransomware-aware backup system.

Implements state transitions:
NORMAL → ELEVATED → SUSPICIOUS → QUARANTINED → TRUSTED
SUSPICIOUS ↔ QUARANTINED (investigation phase)
QUARANTINED → NORMAL (cleared)
QUARANTINED → TRUSTED (immutable restore point)
"""

import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)


class BackupState(Enum):
    """Backup states in ransomware-aware system."""
    NORMAL = "NORMAL"           # Standard operations
    ELEVATED = "ELEVATED"       # Increased monitoring (suspicious signal detected)
    SUSPICIOUS = "SUSPICIOUS"   # High-confidence ransomware (pending human review)
    QUARANTINED = "QUARANTINED" # Isolated for forensics
    TRUSTED = "TRUSTED"         # Validated safe (golden restore point)


class StateTransitionError(Exception):
    """Raised when invalid state transition is attempted."""
    pass


@dataclass
class StateTransitionRecord:
    """Record of a single state transition.
    
    Attributes:
        from_state: Previous state
        to_state: New state
        timestamp: When transition occurred
        decision_maker: Who/what triggered transition (human, AI, system)
        reason: Why transition occurred
        confidence: Confidence level (0.0-1.0) for AI-triggered transitions
        approval_required: Whether this transition requires human approval
        metadata: Additional context (dict)
    """
    from_state: BackupState
    to_state: BackupState
    timestamp: datetime
    decision_maker: str
    reason: str
    confidence: Optional[float] = None
    approval_required: bool = False
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for logging."""
        return {
            "from_state": self.from_state.value,
            "to_state": self.to_state.value,
            "timestamp": self.timestamp.isoformat(),
            "decision_maker": self.decision_maker,
            "reason": self.reason,
            "confidence": self.confidence,
            "approval_required": self.approval_required,
            "metadata": self.metadata
        }


class BackupStateMachine:
    """
    Manages backup state transitions with validation and audit trail.
    
    State diagram:
    ```
    NORMAL
      ↓
    ELEVATED (suspicious signal detected)
      ↓
    SUSPICIOUS (high-confidence ransomware)
      ↓ ↔ QUARANTINED (forensics/investigation)
      ↓
    TRUSTED (validated safe, golden restore point)
    ```
    
    Hard constraints:
    1. No automatic transitions to QUARANTINED (requires human decision)
    2. No automatic transitions from QUARANTINED (requires manual review)
    3. TRUSTED is permanent (cannot transition out)
    4. All transitions logged and immutable
    """
    
    # Define allowed state transitions
    ALLOWED_TRANSITIONS: Dict[BackupState, Set[BackupState]] = {
        BackupState.NORMAL: {BackupState.ELEVATED},
        BackupState.ELEVATED: {BackupState.NORMAL, BackupState.SUSPICIOUS},
        BackupState.SUSPICIOUS: {BackupState.QUARANTINED, BackupState.ELEVATED},
        BackupState.QUARANTINED: {BackupState.SUSPICIOUS, BackupState.TRUSTED, BackupState.NORMAL},
        BackupState.TRUSTED: set(),  # TRUSTED is terminal state
    }
    
    # Which transitions require human approval
    HUMAN_APPROVAL_REQUIRED = {
        (BackupState.ELEVATED, BackupState.SUSPICIOUS),
        (BackupState.SUSPICIOUS, BackupState.QUARANTINED),
        (BackupState.QUARANTINED, BackupState.TRUSTED),
        (BackupState.QUARANTINED, BackupState.NORMAL),
    }
    
    # SLA targets (minutes until escalation)
    ESCALATION_SLAS: Dict[BackupState, int] = {
        BackupState.NORMAL: 1440,        # 24 hours (informational)
        BackupState.ELEVATED: 240,       # 4 hours (wait for next risk window)
        BackupState.SUSPICIOUS: 15,      # 15 minutes (urgent decision needed)
        BackupState.QUARANTINED: 60,     # 1 hour (for investigation)
    }
    
    def __init__(
        self,
        initial_state: BackupState = BackupState.NORMAL,
        backup_id: Optional[str] = None
    ):
        """
        Initialize state machine.
        
        Args:
            initial_state: Starting state (default: NORMAL)
            backup_id: Identifier for this backup (for audit trail)
        """
        self.current_state = initial_state
        self.backup_id = backup_id or f"backup_{datetime.utcnow().timestamp()}"
        self.state_history: List[StateTransitionRecord] = []
        self.state_enter_time: Dict[BackupState, datetime] = {}
        
        # Log initial state
        initial_record = StateTransitionRecord(
            from_state=initial_state,
            to_state=initial_state,
            timestamp=datetime.utcnow(),
            decision_maker="SYSTEM",
            reason="INITIALIZATION",
            approval_required=False
        )
        self.state_history.append(initial_record)
        self.state_enter_time[initial_state] = datetime.utcnow()
        
        logger.info(f"Initialized state machine for {backup_id} in state {initial_state.value}")
    
    def validate_transition(
        self,
        to_state: BackupState
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate whether transition is allowed.
        
        Args:
            to_state: Desired target state
            
        Returns:
            (is_valid, error_reason) tuple
        """
        # Check if target state is reachable from current state
        allowed_targets = self.ALLOWED_TRANSITIONS.get(self.current_state, set())
        
        if to_state not in allowed_targets:
            error = f"Invalid transition: {self.current_state.value} → {to_state.value}"
            return False, error
        
        # TRUSTED is terminal - cannot transition out
        if self.current_state == BackupState.TRUSTED:
            return False, "Cannot transition out of TRUSTED state (terminal)"
        
        # Check SLA expiration
        sla_minutes = self.ESCALATION_SLAS.get(self.current_state, 1440)
        time_in_state = datetime.utcnow() - self.state_enter_time.get(self.current_state, datetime.utcnow())
        
        if time_in_state > timedelta(minutes=sla_minutes):
            logger.warning(
                f"SLA expired for state {self.current_state.value} "
                f"({int(time_in_state.total_seconds() / 60)} minutes, max {sla_minutes})"
            )
        
        return True, None
    
    def transition(
        self,
        to_state: BackupState,
        decision_maker: str,
        reason: str,
        confidence: Optional[float] = None,
        metadata: Optional[Dict] = None,
        force: bool = False
    ) -> StateTransitionRecord:
        """
        Perform state transition with validation and audit logging.
        
        Args:
            to_state: Target state
            decision_maker: Who triggered transition ("HUMAN", "AI", "SYSTEM")
            reason: Why transition is needed
            confidence: AI confidence (0.0-1.0) if AI decision
            metadata: Additional context
            force: Force transition even if validation fails (UNSAFE - use with caution)
            
        Returns:
            StateTransitionRecord documenting the transition
            
        Raises:
            StateTransitionError: If transition is invalid and force=False
        """
        # Validate transition
        is_valid, error_reason = self.validate_transition(to_state)
        
        if not is_valid:
            if force:
                logger.warning(f"FORCED transition (validation disabled): {error_reason}")
            else:
                logger.error(f"Rejecting invalid transition: {error_reason}")
                raise StateTransitionError(error_reason)
        
        # Check if human approval is required
        transition_key = (self.current_state, to_state)
        requires_approval = transition_key in self.HUMAN_APPROVAL_REQUIRED
        
        # Log transition
        record = StateTransitionRecord(
            from_state=self.current_state,
            to_state=to_state,
            timestamp=datetime.utcnow(),
            decision_maker=decision_maker,
            reason=reason,
            confidence=confidence,
            approval_required=requires_approval,
            metadata=metadata or {}
        )
        
        # Perform transition
        self.current_state = to_state
        self.state_history.append(record)
        self.state_enter_time[to_state] = datetime.utcnow()
        
        log_level = logging.WARNING if requires_approval else logging.INFO
        logger.log(
            log_level,
            f"State transition: {record.from_state.value} → {record.to_state.value} "
            f"(by {decision_maker}, reason: {reason})"
        )
        
        if requires_approval:
            logger.warning(f"⚠️  HUMAN APPROVAL REQUIRED for this transition")
        
        return record
    
    def get_current_state(self) -> BackupState:
        """Get current state."""
        return self.current_state
    
    def get_state_duration(self) -> timedelta:
        """Get how long system has been in current state."""
        enter_time = self.state_enter_time.get(self.current_state, datetime.utcnow())
        return datetime.utcnow() - enter_time
    
    def get_sla_remaining(self) -> Optional[int]:
        """
        Get remaining time before SLA escalation (minutes).
        
        Returns:
            Minutes remaining, or None if TRUSTED state
        """
        if self.current_state == BackupState.TRUSTED:
            return None  # No escalation for TRUSTED
        
        sla_minutes = self.ESCALATION_SLAS.get(self.current_state, 1440)
        time_in_state = self.get_state_duration()
        remaining = sla_minutes - int(time_in_state.total_seconds() / 60)
        
        return max(0, remaining)
    
    def is_sla_exceeded(self) -> bool:
        """Check if SLA has been exceeded."""
        remaining = self.get_sla_remaining()
        return remaining == 0 if remaining is not None else False
    
    def get_history(self, limit: Optional[int] = None) -> List[StateTransitionRecord]:
        """
        Get transition history.
        
        Args:
            limit: Maximum number of records to return (most recent first)
            
        Returns:
            List of StateTransitionRecord objects
        """
        history = list(reversed(self.state_history))
        if limit:
            history = history[:limit]
        return history
    
    def get_history_json(self, limit: Optional[int] = None) -> List[dict]:
        """Get transition history as JSON-serializable dicts."""
        return [record.to_dict() for record in self.get_history(limit)]
    
    def is_terminal(self) -> bool:
        """Check if in terminal state (TRUSTED)."""
        return self.current_state == BackupState.TRUSTED
    
    def reset_to_normal(
        self,
        decision_maker: str,
        reason: str,
        metadata: Optional[Dict] = None
    ) -> StateTransitionRecord:
        """
        Reset to NORMAL state (post-incident recovery).
        
        Only possible from QUARANTINED state.
        Requires explicit human decision.
        
        Args:
            decision_maker: Who is resetting (should be human)
            reason: Why it's safe to reset
            metadata: Investigation results, clearance notes
            
        Returns:
            StateTransitionRecord
            
        Raises:
            StateTransitionError: If not in QUARANTINED state
        """
        if self.current_state != BackupState.QUARANTINED:
            raise StateTransitionError(
                f"Can only reset to NORMAL from QUARANTINED state, "
                f"currently in {self.current_state.value}"
            )
        
        return self.transition(
            to_state=BackupState.NORMAL,
            decision_maker=decision_maker,
            reason=reason,
            metadata=metadata or {}
        )
    
    def mark_as_trusted(
        self,
        decision_maker: str,
        validation_results: Dict,
        metadata: Optional[Dict] = None
    ) -> StateTransitionRecord:
        """
        Mark backup as TRUSTED (golden restore point).
        
        Only possible from QUARANTINED state after thorough validation.
        Creates immutable reference point for recovery.
        
        Args:
            decision_maker: Who is approving (should be human)
            validation_results: Results of backup validation
            metadata: Validation details, golden point metadata
            
        Returns:
            StateTransitionRecord
            
        Raises:
            StateTransitionError: If not in QUARANTINED state
        """
        if self.current_state != BackupState.QUARANTINED:
            raise StateTransitionError(
                f"Can only mark as TRUSTED from QUARANTINED state, "
                f"currently in {self.current_state.value}"
            )
        
        full_metadata = metadata or {}
        full_metadata["validation_results"] = validation_results
        
        return self.transition(
            to_state=BackupState.TRUSTED,
            decision_maker=decision_maker,
            reason="GOLDEN_RESTORE_POINT_VALIDATION_PASSED",
            metadata=full_metadata
        )
    
    def get_status_summary(self) -> Dict[str, any]:
        """Get comprehensive status summary."""
        return {
            "backup_id": self.backup_id,
            "current_state": self.current_state.value,
            "state_duration_minutes": int(self.get_state_duration().total_seconds() / 60),
            "sla_remaining_minutes": self.get_sla_remaining(),
            "sla_exceeded": self.is_sla_exceeded(),
            "is_terminal": self.is_terminal(),
            "transitions_count": len(self.state_history),
            "last_transition": self.state_history[-1].to_dict() if self.state_history else None
        }
