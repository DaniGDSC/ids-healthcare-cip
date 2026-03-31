"""WUSTL flow simulator — streams all flows with random timing.

Loads ALL CSV files from data/streaming/wustl_flows/ sequentially
(full_* first, then scenario files). Uses random inter-arrival
timing to simulate realistic network traffic patterns.

When the last file is reached, auto-stops and sets a flag so the
dashboard can alert: "Dataset exhausted — update dataset."
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

from dashboard.streaming.feature_aligner import MODEL_FEATURES

logger = logging.getLogger(__name__)


class WUSTLFlowSimulator:
    """Streams WUSTL flows through the inference pipeline.

    Reads all CSV files sequentially with random inter-arrival timing.
    Auto-stops when dataset is exhausted.

    Args:
        flows_dir: Path to wustl_flows directory.
        buffer: WindowBuffer to feed flows into.
        inference_service: InferenceService for model predictions.
    """

    def __init__(
        self,
        flows_dir: str | Path,
        buffer: Any,
        inference_service: Any = None,
        database: Any = None,
    ) -> None:
        self._flows_dir = Path(flows_dir)
        self._buffer = buffer
        self._inference = inference_service
        self._db = database
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._running = False
        self._exhausted = False
        self._flows_injected = 0
        self._inferences_run = 0
        self._total_files = 0
        self._current_file = ""
        self._current_phase = ""
        self._started_at: Optional[datetime] = None
        self._latencies: List[float] = []

        # Load ground truth
        gt_path = self._flows_dir / "_ground_truth.json"
        if gt_path.exists():
            with open(gt_path) as f:
                self._ground_truth = json.load(f)
        else:
            self._ground_truth = {}

        # Load memory-mapped arrays (instant access, zero copy)
        npy_path = self._flows_dir / "flows.npy"
        labels_path = self._flows_dir / "labels.npy"
        filenames_path = self._flows_dir / "filenames.json"

        if npy_path.exists():
            self._flows_mmap = np.load(str(npy_path), mmap_mode="r")
            self._labels_mmap = np.load(str(labels_path), mmap_mode="r")
            with open(filenames_path) as f:
                self._filenames = json.load(f)
            self._total_files = len(self._filenames)
            self._use_mmap = True
            logger.info("WUSTL simulator: %d flows via mmap (%.0f KB)",
                         self._total_files, self._flows_mmap.nbytes / 1024)
        else:
            # Fallback: scan CSV files
            self._use_mmap = False
            self._flows_mmap = None
            self._labels_mmap = None
            full_files = sorted(self._flows_dir.glob("full_*.csv"))
            scenario_files = sorted(f for f in self._flows_dir.glob("scenario_*.csv"))
            self._all_files: List[Path] = full_files + scenario_files
            self._total_files = len(self._all_files)
            self._filenames = [f.name for f in self._all_files]
            logger.info("WUSTL simulator: %d CSV files (no mmap)", self._total_files)

    @property
    def running(self) -> bool:
        with self._lock:
            return self._running

    @property
    def exhausted(self) -> bool:
        with self._lock:
            return self._exhausted

    @property
    def flows_injected(self) -> int:
        with self._lock:
            return self._flows_injected

    def start(self, **kwargs: Any) -> bool:
        """Start streaming. Ignores scenario/mode — always random timing."""
        if self._running:
            return False
        if self._exhausted:
            return False

        self._stop_event.clear()
        self._started_at = datetime.now(timezone.utc)

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._running = True
        self._thread.start()
        logger.info("Simulator started: %d files to stream", self._total_files - self._flows_injected)
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
        self._exhausted = False
        self._current_file = ""
        self._current_phase = ""
        self._started_at = None
        self._latencies.clear()

    def _run(self) -> None:
        """Stream all files: every flow triggers inference.

        Medical IoT requires per-flow assessment — no batching.
        The ~190ms inference latency per flow is acceptable
        (clinical SLA is 5 minutes = 1,500x margin).
        The dashboard uses @st.fragment(run_every=2) to poll
        the buffer independently, so UI stays responsive
        regardless of inference speed.
        """
        start_idx = self._flows_injected

        for i in range(start_idx, self._total_files):
            if self._stop_event.is_set():
                break

            # Read flow: mmap (0.001ms) or CSV fallback (0.8ms)
            if self._use_mmap:
                vec = np.array(self._flows_mmap[i], dtype=np.float32)
                filename = self._filenames[i]
                gt_label = int(self._labels_mmap[i])
            else:
                path = self._all_files[i]
                filename = path.name
                try:
                    df = pd.read_csv(path)
                    vec = df[MODEL_FEATURES].values[0].astype(np.float32)
                except Exception as exc:
                    logger.warning("Failed to read %s: %s", filename, exc)
                    continue
                gt_label = self._ground_truth.get(filename, 0)

            # Feed buffer, then update status (order matters for consistency)
            self._buffer.append(vec)
            phase = self._infer_phase(filename)
            with self._lock:
                self._flows_injected += 1
                self._current_file = filename
                self._current_phase = phase

            # Every flow triggers inference (medical safety requirement)
            if self._inference is not None:
                window = self._buffer.get_window()
                if window is not None:
                    attack_cat = self._infer_category(filename)

                    result = self._inference.process_window(
                        window=window,
                        raw_features=vec,
                        attack_category=attack_cat,
                    )
                    # Window-level GT: 1 if ANY flow in the current
                    # 20-flow window is attack (matches training strategy)
                    if self._use_mmap:
                        win_start = max(0, i - 19)
                        gt_window = int(self._labels_mmap[win_start:i + 1].max())
                    else:
                        gt_window = gt_label
                    result["ground_truth"] = gt_window
                    result["ground_truth_flow"] = gt_label
                    result["phase"] = phase
                    self._buffer.record_prediction(result)

                    # Persist HIGH/CRITICAL alerts to database
                    if self._db and result.get("risk_level") in ("MEDIUM", "HIGH", "CRITICAL"):
                        try:
                            self._db.insert_alert(result)
                        except Exception as exc:
                            logger.warning("DB insert_alert failed: %s", exc)

                    with self._lock:
                        self._inferences_run += 1
                        if "latency_ms" in result:
                            self._latencies.append(result["latency_ms"])

                    # Feed calibrator + ensemble for auto-fitting
                    if hasattr(self._inference, "calibrate_from_scores"):
                        self._inference.calibrate_from_scores(
                            result["anomaly_score"], gt_label, raw_features=vec,
                        )

            # Random inter-arrival (simulates real network jitter)
            from config.production_loader import cfg
            time.sleep(random.uniform(
                cfg("simulation.inter_arrival_min", 0.01),
                cfg("simulation.inter_arrival_max", 0.05),
            ))

        # Check if exhausted (reached end, not stopped by user)
        with self._lock:
            if not self._stop_event.is_set():
                self._exhausted = True
                logger.warning("Dataset exhausted: %d/%d files processed",
                               self._flows_injected, self._total_files)
            self._running = False

    @staticmethod
    def _infer_phase(filename: str) -> str:
        """Derive display phase from filename."""
        if filename.startswith("full_"):
            return "Full test set"
        if "e1_portscan" in filename:
            return "Novelty: Port scan"
        if "e2_biometric" in filename:
            return "Novelty: Biometric tampering"
        if "e3_combined" in filename:
            return "Novelty: Combined attack"
        if "e4_drift" in filename:
            return "Novelty: Concept drift"
        if "scenario_a" in filename:
            return "Benign traffic"
        if "scenario_b" in filename:
            return "Gradual attack"
        if "scenario_c" in filename:
            return "Abrupt attack"
        if "scenario_d" in filename:
            return "Mixed cycle"
        return "Unknown"

    @staticmethod
    def _infer_category(filename: str) -> str:
        """Derive attack category from filename."""
        if "scenario_a" in filename or "full_" in filename:
            return "normal"
        if "e1_" in filename or "e2_" in filename or "e3_" in filename or "e4_" in filename:
            return "unknown"
        return "Spoofing"

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            p50 = sorted(self._latencies)[len(self._latencies) // 2] if self._latencies else 0
            return {
                "running": self._running,
                "exhausted": self._exhausted,
                "current_phase": self._current_phase,
                "current_file": self._current_file,
                "flows_injected": self._flows_injected,
                "total_files": self._total_files,
                "progress_pct": round(self._flows_injected / max(self._total_files, 1) * 100, 1),
                "inferences_run": self._inferences_run,
                "latency_p50_ms": round(p50, 1),
                "started_at": self._started_at.isoformat() if self._started_at else None,
                "scenario": "STREAMING",
                "medsec25_available": False,
            }
