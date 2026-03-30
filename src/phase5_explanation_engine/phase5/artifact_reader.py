"""Phase4ArtifactReader — load and SHA-256-verify Phase 2/3/4 artifacts.

Rebuilds the classification model from Phase 2 builders + Phase 3 weights.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import tensorflow as tf

from src.phase2_detection_engine.phase2.assembler import (
    DetectionModelAssembler,
)
from src.phase2_detection_engine.phase2.attention_builder import (  # noqa: F401
    AttentionBuilder,
    BahdanauAttention,
)
from src.phase2_detection_engine.phase2.bilstm_builder import BiLSTMBuilder
from src.phase2_detection_engine.phase2.cnn_builder import CNNBuilder

logger = logging.getLogger(__name__)

_HASH_CHUNK: int = 65_536
from dashboard.streaming.feature_aligner import N_FEATURES as _N_FEATURES  # canonical: 24


class Phase4ArtifactReader:
    """Load and verify Phase 2/3/4 artifacts via SHA-256 for Phase 5.

    Args:
        project_root: Absolute path to project root.
        phase4_dir: Relative path to Phase 4 output directory.
        phase4_metadata: Relative path to Phase 4 metadata JSON.
        phase3_dir: Relative path to Phase 3 output directory.
        phase3_metadata: Relative path to Phase 3 metadata JSON.
        phase2_dir: Relative path to Phase 2 output directory.
        phase2_metadata: Relative path to Phase 2 metadata JSON.
        label_column: Name of the binary label column.
    """

    def __init__(
        self,
        project_root: Path,
        phase4_dir: Path,
        phase4_metadata: Path,
        phase3_dir: Path,
        phase3_metadata: Path,
        phase2_dir: Path,
        phase2_metadata: Path,
        label_column: str = "Label",
    ) -> None:
        self._root = project_root
        self._p4_dir = project_root / phase4_dir
        self._p4_meta = project_root / phase4_metadata
        self._p3_dir = project_root / phase3_dir
        self._p3_meta = project_root / phase3_metadata
        self._p2_dir = project_root / phase2_dir
        self._p2_meta = project_root / phase2_metadata
        self._label_column = label_column

    def verify_all(
        self,
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """Verify Phase 2, 3, and 4 artifacts via SHA-256.

        Returns:
            Tuple of (p2_metadata, p3_metadata, p4_metadata).

        Raises:
            ValueError: If any SHA-256 mismatch is detected.
        """
        logger.info("── Artifact verification (SHA-256) ──")
        verified = 0

        p2_meta = json.loads(self._p2_meta.read_text())
        for name, info in p2_meta["artifact_hashes"].items():
            actual = self._compute_sha256(self._p2_dir / name)
            if actual != info["sha256"]:
                raise ValueError(f"SHA-256 mismatch: Phase 2 {name}")
            verified += 1
        logger.info("  Phase 2: %d artifacts verified", verified)

        v3 = 0
        p3_meta = json.loads(self._p3_meta.read_text())
        for name, info in p3_meta["artifact_hashes"].items():
            actual = self._compute_sha256(self._p3_dir / name)
            if actual != info["sha256"]:
                raise ValueError(f"SHA-256 mismatch: Phase 3 {name}")
            v3 += 1
        logger.info("  Phase 3: %d artifacts verified", v3)

        v4 = 0
        p4_meta = json.loads(self._p4_meta.read_text())
        for name, info in p4_meta["artifact_hashes"].items():
            actual = self._compute_sha256(self._p4_dir / name)
            if actual != info["sha256"]:
                raise ValueError(f"SHA-256 mismatch: Phase 4 {name}")
            v4 += 1
        logger.info("  Phase 4: %d artifacts verified", v4)
        logger.info("  Total: %d artifacts verified", verified + v3 + v4)

        return p2_meta, p3_meta, p4_meta

    def rebuild_model(
        self,
        p2_metadata: Dict[str, Any],
        p3_metadata: Dict[str, Any],
    ) -> tf.keras.Model:
        """Rebuild classification model and load Phase 3 weights.

        Args:
            p2_metadata: Phase 2 metadata with hyperparameters.
            p3_metadata: Phase 3 metadata with head config.

        Returns:
            Loaded Keras model with sigmoid output.
        """
        logger.info("── Rebuilding classification model ──")
        hp = p2_metadata["hyperparameters"]

        builders = [
            CNNBuilder(
                filters_1=hp["cnn_filters_1"],
                filters_2=hp["cnn_filters_2"],
                kernel_size=hp["cnn_kernel_size"],
                activation=hp["cnn_activation"],
                pool_size=hp["cnn_pool_size"],
            ),
            BiLSTMBuilder(
                units_1=hp["bilstm_units_1"],
                units_2=hp["bilstm_units_2"],
                dropout_rate=hp["dropout_rate"],
            ),
            AttentionBuilder(units=hp["attention_units"]),
        ]

        assembler = DetectionModelAssembler(
            timesteps=hp["timesteps"],
            n_features=_N_FEATURES,
            builders=builders,
        )
        detection_model = assembler.assemble()

        p3_hp = p3_metadata["hyperparameters"]
        x = tf.keras.layers.Dense(
            p3_hp["dense_units"],
            activation=p3_hp["dense_activation"],
            name="dense_head",
        )(detection_model.output)
        x = tf.keras.layers.Dropout(p3_hp["head_dropout_rate"], name="drop_head")(x)
        output = tf.keras.layers.Dense(1, activation="sigmoid", name="output")(x)
        full_model = tf.keras.Model(detection_model.input, output, name="classification_engine")

        weights_path = self._p3_dir / "classification_model.weights.h5"
        full_model.load_weights(str(weights_path))
        logger.info(
            "  Model loaded: %d params, %d layers",
            full_model.count_params(),
            len(full_model.layers),
        )
        return full_model

    def load_risk_report(self) -> Dict[str, Any]:
        """Load risk_report.json from Phase 4 output."""
        path = self._p4_dir / "risk_report.json"
        report = json.loads(path.read_text())
        logger.info(
            "  Risk report loaded: %d samples, dist=%s",
            report["total_samples"],
            report["risk_distribution"],
        )
        return report

    def load_baseline(self) -> Dict[str, Any]:
        """Load baseline_config.json from Phase 4 output."""
        path = self._p4_dir / "baseline_config.json"
        baseline = json.loads(path.read_text())
        logger.info(
            "  Baseline loaded: threshold=%.6f, mad=%.6f",
            baseline["baseline_threshold"],
            baseline["mad"],
        )
        return baseline

    @staticmethod
    def _compute_sha256(path: Path) -> str:
        """Compute SHA-256 hex digest of a file."""
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(_HASH_CHUNK):
                h.update(chunk)
        return h.hexdigest()
