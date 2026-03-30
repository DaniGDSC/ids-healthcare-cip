"""WUSTL flow simulator — streams pre-scaled flows through InferenceService.

Reads 24-feature WUSTL flow CSVs from data/streaming/wustl_flows/,
feeds them directly into WindowBuffer (no re-scaling — already
RobustScaler-transformed), runs InferenceService for real model
inference, and records predictions for dashboard display.

Supports scenarios A-E and RANDOM mode.
"""

from __future__ import annotations

import json
import logging
import random
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
from dashboard.streaming.feature_aligner import MODEL_FEATURES

logger = logging.getLogger(__name__)


class WUSTLFlowSimulator:
    """Streams WUSTL flows through the inference pipeline.

    Args:
        flows_dir: Path to wustl_flows directory.
        buffer: WindowBuffer to feed flows into.
        inference_service: InferenceService for model predictions.
    """

    # Map phase labels to file prefixes
    _PHASE_PREFIX: Dict[str, str] = {
        "benign": "scenario_a",
        "Benign": "scenario_a",
        "attack": "scenario_c",
        "Reconnaissance": "scenario_b",
        "e1_portscan": "scenario_e1_portscan",
        "e2_biometric": "scenario_e2_biometric",
        "e3_combined": "scenario_e3_combined",
        "e4_drift": "scenario_e4_drift",
    }

    def __init__(
        self,
        flows_dir: str | Path,
        buffer: Any,
        inference_service: Any = None,
    ) -> None:
        self._flows_dir = Path(flows_dir)
        self._buffer = buffer
        self._inference = inference_service
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._running = False
        self._flows_injected = 0
        self._inferences_run = 0
        self._current_scenario: Optional[ScenarioID] = None
        self._current_phase: str = ""
        self._started_at: Optional[datetime] = None
        self._latencies: List[float] = []

        # Load ground truth
        gt_path = self._flows_dir / "_ground_truth.json"
        if gt_path.exists():
            with open(gt_path) as f:
                self._ground_truth = json.load(f)
        else:
            self._ground_truth = {}

        # Index files by prefix
        self._file_index: Dict[str, List[Path]] = {}
        for csv_file in sorted(self._flows_dir.glob("*.csv")):
            for label, prefix in self._PHASE_PREFIX.items():
                if csv_file.name.startswith(prefix):
                    self._file_index.setdefault(label, []).append(csv_file)
                    break
            # Also index full_* files as mixed
            if csv_file.name.startswith("full_"):
                self._file_index.setdefault("full", []).append(csv_file)

        logger.info("WUSTL simulator: %d files indexed from %s",
                     sum(len(v) for v in self._file_index.values()), self._flows_dir)

    @property
    def running(self) -> bool:
        return self._running

    @property
    def flows_injected(self) -> int:
        return self._flows_injected

    def start(self, scenario: ScenarioID = ScenarioID.C, mode: SimMode = SimMode.ACCELERATED) -> bool:
        if self._running:
            return False

        self._current_scenario = scenario
        self._stop_event.clear()
        self._started_at = datetime.now(timezone.utc)

        self._thread = threading.Thread(
            target=self._run,
            args=(scenario, mode),
            daemon=True,
        )
        self._running = True
        self._thread.start()
        logger.info("Simulator started: %s %s", scenario.value, mode.value)
        return True

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        self._running = False

    def reset(self) -> None:
        self.stop()
        self._flows_injected = 0
        self._inferences_run = 0
        self._current_scenario = None
        self._current_phase = ""
        self._started_at = None
        self._latencies.clear()

    def _run(self, scenario_id: ScenarioID, mode: SimMode) -> None:
        if scenario_id == ScenarioID.RANDOM:
            self._run_random(mode)
        else:
            config = SCENARIOS.get(scenario_id)
            if config:
                self._run_scenario(config, mode)
        self._running = False

    def _run_random(self, mode: SimMode) -> None:
        options = [ScenarioID.A, ScenarioID.B, ScenarioID.C, ScenarioID.D, ScenarioID.E]
        while not self._stop_event.is_set():
            chosen = random.choice(options)
            self._current_scenario = chosen
            config = SCENARIOS.get(chosen)
            if config:
                self._run_scenario(config, mode)

    def _run_scenario(self, config: ScenarioConfig, mode: SimMode) -> None:
        multiplier = MODE_MULTIPLIERS[mode]
        for phase in config.phases:
            if self._stop_event.is_set():
                break
            self._run_phase(phase, multiplier)

    def _run_phase(self, phase: ScenarioPhase, multiplier: float) -> None:
        self._current_phase = phase.description
        label = phase.label_filter
        files = self._file_index.get(label, self._file_index.get("benign", []))

        if not files:
            logger.warning("No files for phase label: %s", label)
            return

        n = min(phase.duration_flows, len(files))
        sample = random.sample(files, n) if n < len(files) else files[:n]

        for path in sample:
            if self._stop_event.is_set():
                break

            try:
                df = pd.read_csv(path)
                vec = df[MODEL_FEATURES].values[0].astype(np.float32)
            except Exception as exc:
                logger.warning("Failed to read %s: %s", path.name, exc)
                continue

            # Feed buffer directly (pre-scaled, no transform)
            self._buffer.append(vec)
            self._flows_injected += 1

            # Run inference if window available
            window = self._buffer.get_window()
            if window is not None and self._inference is not None:
                gt_label = self._ground_truth.get(path.name, 0)
                attack_cat = self._infer_category(label)

                result = self._inference.process_window(
                    window=window,
                    raw_features=vec,
                    attack_category=attack_cat,
                )
                result["ground_truth"] = gt_label
                result["phase"] = self._current_phase
                self._buffer.record_prediction(result)
                self._inferences_run += 1

                if "latency_ms" in result:
                    self._latencies.append(result["latency_ms"])

            # Inter-arrival delay
            delay = 0.05 * multiplier + np.random.exponential(0.02 * multiplier)
            time.sleep(min(delay, 1.0))

    @staticmethod
    def _infer_category(label: str) -> str:
        if label in ("benign", "Benign"):
            return "normal"
        if "portscan" in label:
            return "unknown"
        if "biometric" in label:
            return "unknown"
        if "combined" in label:
            return "unknown"
        if "drift" in label:
            return "unknown"
        return "Spoofing"

    def get_status(self) -> Dict[str, Any]:
        p50 = sorted(self._latencies)[len(self._latencies) // 2] if self._latencies else 0
        return {
            "running": self._running,
            "scenario": self._current_scenario.value if self._current_scenario else None,
            "current_phase": self._current_phase,
            "flows_injected": self._flows_injected,
            "inferences_run": self._inferences_run,
            "latency_p50_ms": round(p50, 1),
            "started_at": self._started_at.isoformat() if self._started_at else None,
            "medsec25_available": False,
        }
