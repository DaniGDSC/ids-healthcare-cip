"""Phase 3 artifact reader — load and SHA-256-verify Phase 2/3 outputs.

Reads classification weights, attention output, metadata, and Phase 1
preprocessed data for the Phase 4 risk-adaptive pipeline.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf

from src.phase2_detection_engine.phase2.assembler import DetectionModelAssembler
from src.phase2_detection_engine.phase2.attention_builder import (  # noqa: F401
    AttentionBuilder,
    BahdanauAttention,
)
from src.phase2_detection_engine.phase2.bilstm_builder import BiLSTMBuilder
from src.phase2_detection_engine.phase2.cnn_builder import CNNBuilder

logger = logging.getLogger(__name__)

_HASH_CHUNK: int = 65_536
_N_FEATURES: int = 29


class Phase3ArtifactReader:
    """Load and verify Phase 3 and Phase 2 artifacts via SHA-256.

    Args:
        project_root: Absolute path to project root.
        phase3_dir: Relative path to Phase 3 output directory.
        phase3_metadata: Relative path to Phase 3 metadata JSON.
        phase2_dir: Relative path to Phase 2 output directory.
        phase2_metadata: Relative path to Phase 2 metadata JSON.
        label_column: Name of the binary label column.
    """

    def __init__(
        self,
        project_root: Path,
        phase3_dir: Path,
        phase3_metadata: Path,
        phase2_dir: Path,
        phase2_metadata: Path,
        label_column: str = "Label",
    ) -> None:
        self._root = project_root
        self._phase3_dir = project_root / phase3_dir
        self._phase3_meta_path = project_root / phase3_metadata
        self._phase2_dir = project_root / phase2_dir
        self._phase2_meta_path = project_root / phase2_metadata
        self._label_col = label_column

    def verify_phase3(self) -> Tuple[Path, Dict[str, Any]]:
        """Load metadata and verify all Phase 3 artifacts via SHA-256.

        Returns:
            Tuple of (weights_path, metadata_dict).

        Raises:
            FileNotFoundError: If metadata or artifacts missing.
            ValueError: If SHA-256 hash does not match.
        """
        logger.info("── Phase 3 artifact verification ──")

        if not self._phase3_meta_path.exists():
            raise FileNotFoundError(f"Phase 3 metadata not found: {self._phase3_meta_path}")

        metadata = json.loads(self._phase3_meta_path.read_text())

        for artifact_name, hash_info in metadata["artifact_hashes"].items():
            artifact_path = self._phase3_dir / artifact_name
            if not artifact_path.exists():
                raise FileNotFoundError(f"Phase 3 artifact missing: {artifact_path}")

            expected = hash_info["sha256"]
            actual = self._compute_sha256(artifact_path)
            if actual != expected:
                raise ValueError(
                    f"SHA-256 mismatch for {artifact_name}: "
                    f"expected={expected[:16]}…, actual={actual[:16]}…"
                )
            logger.info("  ✓ SHA-256 verified: %s", artifact_name)

        weights_path = self._phase3_dir / "classification_model.weights.h5"
        logger.info("  All Phase 3 artifacts verified.")
        return weights_path, metadata

    def verify_phase2(self) -> Tuple[Path, Dict[str, Any]]:
        """Load metadata and verify all Phase 2 artifacts via SHA-256.

        Returns:
            Tuple of (attention_output_path, metadata_dict).

        Raises:
            FileNotFoundError: If metadata or artifacts missing.
            ValueError: If SHA-256 hash does not match.
        """
        logger.info("── Phase 2 artifact verification ──")

        if not self._phase2_meta_path.exists():
            raise FileNotFoundError(f"Phase 2 metadata not found: {self._phase2_meta_path}")

        metadata = json.loads(self._phase2_meta_path.read_text())

        for artifact_name, hash_info in metadata["artifact_hashes"].items():
            artifact_path = self._phase2_dir / artifact_name
            if not artifact_path.exists():
                raise FileNotFoundError(f"Phase 2 artifact missing: {artifact_path}")

            expected = hash_info["sha256"]
            actual = self._compute_sha256(artifact_path)
            if actual != expected:
                raise ValueError(
                    f"SHA-256 mismatch for {artifact_name}: "
                    f"expected={expected[:16]}…, actual={actual[:16]}…"
                )
            logger.info("  ✓ SHA-256 verified: %s", artifact_name)

        attn_path = self._phase2_dir / "attention_output.parquet"
        logger.info("  All Phase 2 artifacts verified.")
        return attn_path, metadata

    def load_attention_output(self) -> pd.DataFrame:
        """Load Phase 2 attention output parquet.

        Returns:
            DataFrame with attn_0..attn_127, Label, split columns.
        """
        attn_path = self._phase2_dir / "attention_output.parquet"
        attn_df = pd.read_parquet(attn_path)
        logger.info("  Loaded attention output: %s", attn_df.shape)
        return attn_df

    def load_phase1_data(
        self, train_path: Path, test_path: Path
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """Load Phase 1 preprocessed train/test parquets.

        Args:
            train_path: Absolute path to train parquet.
            test_path: Absolute path to test parquet.

        Returns:
            Tuple of (X_train, y_train, X_test, y_test, feature_names).
        """
        logger.info("── Loading Phase 1 data ──")

        train_df = pd.read_parquet(train_path)
        test_df = pd.read_parquet(test_path)

        feature_names = [c for c in train_df.columns if c != self._label_col]
        X_train = train_df[feature_names].values.astype(np.float32)
        y_train = train_df[self._label_col].values.astype(np.int32)
        X_test = test_df[feature_names].values.astype(np.float32)
        y_test = test_df[self._label_col].values.astype(np.int32)

        logger.info(
            "  Train: %s, Test: %s, features=%d",
            X_train.shape,
            X_test.shape,
            len(feature_names),
        )
        return X_train, y_train, X_test, y_test, feature_names

    def rebuild_model(
        self,
        p2_metadata: Dict[str, Any],
        p3_metadata: Dict[str, Any],
    ) -> tf.keras.Model:
        """Rebuild full classification model and load Phase 3 weights.

        Uses Phase 2 builders for CNN-BiLSTM-Attention backbone,
        then attaches Phase 3 classification head.

        Args:
            p2_metadata: Phase 2 detection metadata with hyperparameters.
            p3_metadata: Phase 3 classification metadata with head config.

        Returns:
            Fully loaded classification model with sigmoid output.
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

        # Attach classification head (same as Phase 3)
        p3_hp = p3_metadata["hyperparameters"]
        x = tf.keras.layers.Dense(
            p3_hp["dense_units"],
            activation=p3_hp["dense_activation"],
            name="dense_head",
        )(detection_model.output)
        x = tf.keras.layers.Dropout(p3_hp["head_dropout_rate"], name="drop_head")(x)
        output = tf.keras.layers.Dense(1, activation="sigmoid", name="output")(x)
        full_model = tf.keras.Model(detection_model.input, output, name="classification_engine")

        # Load Phase 3 weights
        weights_path = self._phase3_dir / "classification_model.weights.h5"
        full_model.load_weights(str(weights_path))
        logger.info(
            "  Model loaded: %d params, %d layers",
            full_model.count_params(),
            len(full_model.layers),
        )
        return full_model

    @staticmethod
    def _compute_sha256(path: Path) -> str:
        """Compute SHA-256 hex digest of a file."""
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(_HASH_CHUNK):
                h.update(chunk)
        return h.hexdigest()
