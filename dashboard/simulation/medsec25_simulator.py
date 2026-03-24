"""MedSec-25 stochastic streaming simulator.

Reads MedSec-25 CSV, filters by scenario phase, and writes
individual flow CSVs to the streaming input directory at
distribution-matched inter-arrival rates.
"""

from __future__ import annotations

import csv
import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from dashboard.simulation.scenarios import (
    MODE_MULTIPLIERS,
    SCENARIOS,
    ScenarioConfig,
    ScenarioID,
    ScenarioPhase,
    SimMode,
)
from dashboard.streaming.feature_aligner import WUSTL_FEATURES, align_medsec25_batch

logger = logging.getLogger(__name__)

MEDSEC25_PATH: str = "data/external/MedSec-25.csv"
INPUT_DIR: str = "data/streaming/input"


class MedSec25StreamSimulator:
    """Simulates streaming flow injection from MedSec-25 dataset.

    Reads the full dataset once, then writes individual flow CSVs
    to INPUT_DIR at configurable inter-arrival rates per scenario.
    """

    def __init__(
        self,
        medsec_path: str = MEDSEC25_PATH,
        input_dir: str = INPUT_DIR,
    ) -> None:
        self._medsec_path = Path(medsec_path)
        self._input_dir = Path(input_dir)
        self._input_dir.mkdir(parents=True, exist_ok=True)

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._running = False
        self._flows_injected: int = 0
        self._current_scenario: Optional[ScenarioID] = None
        self._current_mode: SimMode = SimMode.ACCELERATED
        self._current_phase_label: str = ""
        self._started_at: Optional[datetime] = None

        # Pre-loaded and pre-aligned data partitions
        self._benign_df: Optional[pd.DataFrame] = None
        self._attack_dfs: Dict[str, pd.DataFrame] = {}
        self._loaded = False

    @property
    def running(self) -> bool:
        """Whether simulation is currently active."""
        return self._running

    @property
    def flows_injected(self) -> int:
        """Total flows injected in current session."""
        return self._flows_injected

    def load_dataset(self, sample_size: int = 5000) -> bool:
        """Pre-load and align MedSec-25 dataset.

        Args:
            sample_size: Max rows to load per category for memory efficiency.

        Returns:
            True if loaded successfully.
        """
        if self._loaded:
            return True

        if not self._medsec_path.exists():
            logger.error("MedSec-25 not found: %s", self._medsec_path)
            return False

        try:
            logger.info("Loading MedSec-25 dataset...")
            df = pd.read_csv(self._medsec_path, low_memory=False)

            # Separate by label
            benign = df[df["Label"].str.lower() == "benign"]
            attack_labels = df[df["Label"].str.lower() != "benign"]["Label"].unique()

            # Sample for memory efficiency
            if len(benign) > sample_size:
                benign = benign.sample(n=sample_size, random_state=42)

            # Align to WUSTL features
            self._benign_df = align_medsec25_batch(benign)

            for label in attack_labels:
                subset = df[df["Label"] == label]
                if len(subset) > sample_size:
                    subset = subset.sample(n=sample_size, random_state=42)
                self._attack_dfs[label.lower()] = align_medsec25_batch(subset)

            self._loaded = True
            logger.info(
                "MedSec-25 loaded: %d benign, %d attack categories",
                len(self._benign_df),
                len(self._attack_dfs),
            )
            return True

        except Exception as exc:
            logger.error("Failed to load MedSec-25: %s", exc)
            return False

    def start(
        self,
        scenario: ScenarioID = ScenarioID.C,
        mode: SimMode = SimMode.ACCELERATED,
    ) -> bool:
        """Start streaming simulation.

        Args:
            scenario: Which scenario to run (A/B/C/D).
            mode: Timing mode (REALTIME/ACCELERATED/STRESS).

        Returns:
            True if started successfully.
        """
        if self._running:
            logger.warning("Simulation already running")
            return False

        if not self._loaded:
            if not self.load_dataset():
                return False

        self._current_scenario = scenario
        self._current_mode = mode
        self._flows_injected = 0
        self._stop_event.clear()
        self._started_at = datetime.now(timezone.utc)

        self._thread = threading.Thread(
            target=self._run_scenario,
            args=(scenario, mode),
            daemon=True,
        )
        self._running = True
        self._thread.start()
        logger.info("Simulation started: Scenario %s, Mode %s",
                     scenario.value, mode.value)
        return True

    def stop(self) -> None:
        """Stop the simulation."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        self._running = False
        logger.info("Simulation stopped after %d flows", self._flows_injected)

    def reset(self) -> None:
        """Stop and reset the simulator."""
        self.stop()
        self._flows_injected = 0
        self._current_scenario = None
        self._current_phase_label = ""
        self._started_at = None

        # Clear input directory
        for f in self._input_dir.glob("*.csv"):
            try:
                f.unlink()
            except OSError:
                pass

        logger.info("Simulator reset")

    def _run_scenario(self, scenario_id: ScenarioID, mode: SimMode) -> None:
        """Execute a scenario in a background thread."""
        config = SCENARIOS[scenario_id]
        multiplier = MODE_MULTIPLIERS[mode]

        for phase in config.phases:
            if self._stop_event.is_set():
                break
            self._run_phase(phase, multiplier)

        self._running = False
        logger.info("Scenario %s completed: %d flows",
                     scenario_id.value, self._flows_injected)

    def _run_phase(self, phase: ScenarioPhase, multiplier: float) -> None:
        """Execute a single scenario phase."""
        self._current_phase_label = phase.description

        # Select source dataframe
        label = phase.label_filter.lower()
        if label == "benign":
            source_df = self._benign_df
        elif label == "all":
            # Mix benign and attack
            all_attack = pd.concat(list(self._attack_dfs.values()))
            source_df = pd.concat([self._benign_df, all_attack])
        else:
            # Find matching attack category
            source_df = None
            for key, adf in self._attack_dfs.items():
                if label in key:
                    source_df = adf
                    break
            if source_df is None:
                # Use any attack data
                source_df = pd.concat(list(self._attack_dfs.values()))

        if source_df is None or source_df.empty:
            logger.warning("No data for phase: %s", phase.label_filter)
            return

        # Sample flows for this phase
        n = min(phase.duration_flows, len(source_df))
        sample = source_df.sample(n=n, replace=True, random_state=None)

        # Write flows one at a time with inter-arrival delay
        for i, (_, row) in enumerate(sample.iterrows()):
            if self._stop_event.is_set():
                break

            # Write single-row CSV
            flow_file = self._input_dir / f"flow_{self._flows_injected:06d}.csv"
            row_df = pd.DataFrame([row[WUSTL_FEATURES]])
            if "Label" in row.index:
                row_df["Label"] = row["Label"]
            row_df.to_csv(flow_file, index=False)

            self._flows_injected += 1

            # Inter-arrival delay (stochastic with exponential jitter)
            base_delay = 0.05 * multiplier  # 50ms base at 1x
            jitter = np.random.exponential(base_delay)
            delay = min(base_delay + jitter, 2.0 * multiplier)
            time.sleep(delay)

    def get_status(self) -> Dict[str, Any]:
        """Get simulator status summary."""
        scenario = SCENARIOS.get(self._current_scenario) if self._current_scenario else None
        return {
            "running": self._running,
            "loaded": self._loaded,
            "scenario": self._current_scenario.value if self._current_scenario else None,
            "scenario_name": scenario.name if scenario else None,
            "mode": self._current_mode.value,
            "flows_injected": self._flows_injected,
            "current_phase": self._current_phase_label,
            "medsec25_available": self._medsec_path.exists(),
            "started_at": self._started_at.isoformat() if self._started_at else None,
        }
