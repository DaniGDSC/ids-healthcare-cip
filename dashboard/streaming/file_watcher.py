"""Watchdog-based file system event handler for streaming input.

Monitors INPUT_DIR for new CSV files, validates schema, applies
preprocessing, and feeds the sliding window buffer.

Supports both WUSTL-native (24 features) and MedSec-25 (CIC-style)
input formats.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
import pandas as pd
from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from dashboard.streaming.feature_aligner import (
    MODEL_FEATURES,
    N_FEATURES,
    align_medsec25_row,
    align_wustl_native,
    validate_schema,
)
from dashboard.streaming.window_buffer import WindowBuffer

logger = logging.getLogger(__name__)

INPUT_DIR: str = "data/streaming/input"


class FlowFileHandler(FileSystemEventHandler):
    """Handles new CSV flow files in the streaming input directory.

    Validates schema, applies scaling, and appends to the window buffer.
    """

    def __init__(
        self,
        buffer: WindowBuffer,
        scaler: Any,
        on_prediction_ready: Optional[Callable[[np.ndarray], None]] = None,
    ) -> None:
        super().__init__()
        self._buffer = buffer
        self._scaler = scaler
        self._on_prediction_ready = on_prediction_ready
        self._processed_files: set = set()

    def on_created(self, event: FileSystemEvent) -> None:
        """Handle new file creation events."""
        if event.is_directory:
            return

        path = Path(event.src_path)
        if path.suffix.lower() != ".csv":
            return

        if str(path) in self._processed_files:
            return

        time.sleep(0.1)

        try:
            self._process_file(path)
            self._processed_files.add(str(path))
        except Exception as exc:
            logger.error("Failed to process %s: %s", path.name, exc)

    def _process_file(self, path: Path) -> None:
        """Process a single CSV file."""
        df = pd.read_csv(path)

        if df.empty:
            logger.warning("Empty file: %s", path.name)
            return

        # Try WUSTL-native schema first (24 features)
        valid, msg = validate_schema(df)
        if valid:
            features_df = align_wustl_native(df)
        else:
            # Fall back to MedSec-25 alignment
            logger.info("Aligning %s: %s", path.name, msg)
            features_df = self._align_medsec25(df)

        if features_df is None:
            return

        # Apply scaler (transform only, never fit)
        scaled = self._scaler.transform(features_df.values)

        for i in range(len(scaled)):
            self._buffer.append(scaled[i].astype(np.float32))

        logger.info("Ingested %d flows from %s (total: %d)",
                     len(scaled), path.name, self._buffer.flow_count)

        if self._on_prediction_ready and not self._buffer.suppress_alerts:
            window = self._buffer.get_window()
            if window is not None:
                self._on_prediction_ready(window)

    def _align_medsec25(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Align MedSec-25 columns to 24-feature model schema."""
        cols = list(df.columns)
        rows = []
        for _, row in df.iterrows():
            vec = align_medsec25_row(row, cols)
            rows.append(vec)

        if not rows:
            return None

        return pd.DataFrame(rows, columns=MODEL_FEATURES)


class StreamWatcher:
    """Manages the watchdog observer for the streaming input directory."""

    def __init__(
        self,
        buffer: WindowBuffer,
        scaler: Any,
        input_dir: str = INPUT_DIR,
        on_prediction_ready: Optional[Callable[[np.ndarray], None]] = None,
    ) -> None:
        self._input_dir = Path(input_dir)
        self._input_dir.mkdir(parents=True, exist_ok=True)
        self._handler = FlowFileHandler(
            buffer, scaler, on_prediction_ready,
        )
        self._observer: Optional[Observer] = None
        self._running = False

    @property
    def running(self) -> bool:
        return self._running

    def start(self) -> None:
        """Start watching the input directory."""
        if self._running:
            return

        self._observer = Observer()
        self._observer.schedule(
            self._handler, str(self._input_dir), recursive=False,
        )
        self._observer.daemon = True
        self._observer.start()
        self._running = True
        logger.info("StreamWatcher started: %s", self._input_dir)

    def stop(self) -> None:
        """Stop watching the input directory."""
        if self._observer and self._running:
            self._observer.stop()
            self._observer.join(timeout=5)
            self._running = False
            logger.info("StreamWatcher stopped")

    def get_status(self) -> Dict[str, Any]:
        return {
            "running": self._running,
            "input_dir": str(self._input_dir),
            "input_dir_exists": self._input_dir.exists(),
        }
