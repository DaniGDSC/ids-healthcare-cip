"""Simulation scenario definitions for MedSec-25 streaming.

Four scenarios covering FPR characterization, gradual attack,
abrupt attack, and mixed attack-recovery cycles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List


class ScenarioID(str, Enum):
    """Simulation scenario identifiers."""

    A = "A"  # Benign only — FPR characterization
    B = "B"  # Gradual attack — reconnaissance simulation
    C = "C"  # Abrupt attack — DoS/DDoS simulation
    D = "D"  # Mixed cycle — attack-recovery resilience
    E = "E"  # Novelty attacks — zero-day / drift
    RANDOM = "RANDOM"  # Random scenario cycling


class SimMode(str, Enum):
    """Simulation timing modes."""

    REALTIME = "REALTIME"         # 1x actual flow duration
    ACCELERATED = "ACCELERATED"   # 10x speed (demo default)
    STRESS = "STRESS"             # Maximum throughput (load testing)


@dataclass
class ScenarioPhase:
    """A single phase within a scenario."""

    label_filter: str  # "Benign", "attack", or "all"
    duration_flows: int  # Number of flows in this phase
    description: str


@dataclass
class ScenarioConfig:
    """Full scenario configuration."""

    id: ScenarioID
    name: str
    description: str
    phases: List[ScenarioPhase] = field(default_factory=list)


SCENARIOS: dict[ScenarioID, ScenarioConfig] = {
    ScenarioID.A: ScenarioConfig(
        id=ScenarioID.A,
        name="Benign Only",
        description="FPR characterization — benign traffic only",
        phases=[
            ScenarioPhase("Benign", 500, "Sustained benign traffic"),
        ],
    ),
    ScenarioID.B: ScenarioConfig(
        id=ScenarioID.B,
        name="Gradual Attack",
        description="Reconnaissance simulation — gradual attack onset",
        phases=[
            ScenarioPhase("Benign", 200, "Normal baseline establishment"),
            ScenarioPhase("Reconnaissance", 150,
                          "Low-rate reconnaissance probing"),
            ScenarioPhase("attack", 150,
                          "Escalation to active exploitation"),
        ],
    ),
    ScenarioID.C: ScenarioConfig(
        id=ScenarioID.C,
        name="Abrupt Attack",
        description="DoS/DDoS simulation — sudden volumetric attack",
        phases=[
            ScenarioPhase("Benign", 100, "Normal baseline"),
            ScenarioPhase("attack", 300, "Abrupt volumetric attack"),
            ScenarioPhase("Benign", 100, "Post-attack recovery"),
        ],
    ),
    ScenarioID.D: ScenarioConfig(
        id=ScenarioID.D,
        name="Mixed Cycle",
        description="Attack-recovery resilience — alternating phases",
        phases=[
            ScenarioPhase("Benign", 100, "Normal Phase 1"),
            ScenarioPhase("attack", 80, "Attack Phase 1"),
            ScenarioPhase("Benign", 60, "Recovery Phase 1"),
            ScenarioPhase("attack", 80, "Attack Phase 2"),
            ScenarioPhase("Benign", 80, "Recovery Phase 2"),
        ],
    ),
    ScenarioID.E: ScenarioConfig(
        id=ScenarioID.E,
        name="Novelty Attacks",
        description="Zero-day detection — port scan, biometric, combined, drift",
        phases=[
            ScenarioPhase("benign", 50, "Normal baseline"),
            ScenarioPhase("e1_portscan", 100, "Novel port scan attack"),
            ScenarioPhase("benign", 30, "Recovery"),
            ScenarioPhase("e2_biometric", 100, "Biometric tampering"),
            ScenarioPhase("benign", 30, "Recovery"),
            ScenarioPhase("e3_combined", 100, "Combined network + biometric"),
            ScenarioPhase("e4_drift", 200, "Concept drift"),
        ],
    ),
}

# Timing multipliers per mode
MODE_MULTIPLIERS: dict[SimMode, float] = {
    SimMode.REALTIME: 1.0,
    SimMode.ACCELERATED: 0.1,   # 10x speed
    SimMode.STRESS: 0.001,      # near-instant
}
