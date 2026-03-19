"""Classification trainer — compile + fit with injectable callbacks.

Handles per-phase training for progressive unfreezing.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import tensorflow as tf

from .config import TrainingPhaseConfig
from .unfreezer import ProgressiveUnfreezer

logger = logging.getLogger(__name__)


class ClassificationTrainer:
    """Compile and fit the classification model for each unfreezing phase.

    Args:
        batch_size: Training batch size.
        validation_split: Fraction of training data for validation.
        early_stopping_patience: EarlyStopping patience (epochs).
        reduce_lr_patience: ReduceLROnPlateau patience (epochs).
        reduce_lr_factor: ReduceLROnPlateau factor.
    """

    def __init__(
        self,
        batch_size: int = 256,
        validation_split: float = 0.2,
        early_stopping_patience: int = 3,
        reduce_lr_patience: int = 2,
        reduce_lr_factor: float = 0.5,
    ) -> None:
        self._batch_size = batch_size
        self._val_split = validation_split
        self._es_patience = early_stopping_patience
        self._lr_patience = reduce_lr_patience
        self._lr_factor = reduce_lr_factor

    def train_phase(
        self,
        model: tf.keras.Model,
        phase_cfg: TrainingPhaseConfig,
        X_train: np.ndarray,
        y_train: np.ndarray,
        loss: str,
        output_dir: Path,
        phase_index: int = 0,
    ) -> Dict[str, Any]:
        """Run a single training phase (compile + fit).

        Args:
            model: Full classification model.
            phase_cfg: Phase configuration (name, epochs, lr, frozen).
            X_train: Windowed training features.
            y_train: Windowed training labels.
            loss: Loss function name.
            output_dir: Directory for checkpoint files.
            phase_index: Phase index for checkpoint naming.

        Returns:
            Phase history dict with loss/accuracy curves.
        """
        logger.info("── %s ──", phase_cfg.name)

        checkpoint_path = output_dir / f"checkpoint_phase_{phase_index}.weights.h5"

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=self._es_patience,
                restore_best_weights=True,
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(checkpoint_path),
                save_best_only=True,
                save_weights_only=True,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=self._lr_factor,
                patience=self._lr_patience,
            ),
        ]

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=phase_cfg.learning_rate),
            loss=loss,
            metrics=["accuracy"],
        )

        history = model.fit(
            X_train,
            y_train,
            epochs=phase_cfg.epochs,
            batch_size=self._batch_size,
            validation_split=self._val_split,
            callbacks=callbacks,
            verbose=1,
        )

        phase_history = {
            "phase": phase_cfg.name,
            "epochs_run": len(history.history["loss"]),
            "final_train_loss": float(history.history["loss"][-1]),
            "final_train_acc": float(history.history["accuracy"][-1]),
            "final_val_loss": float(history.history["val_loss"][-1]),
            "final_val_acc": float(history.history["val_accuracy"][-1]),
            "history": {k: [float(v) for v in vals] for k, vals in history.history.items()},
        }

        logger.info(
            "  %s → val_loss=%.4f, val_acc=%.4f",
            phase_cfg.name,
            phase_history["final_val_loss"],
            phase_history["final_val_acc"],
        )
        return phase_history

    def train_all_phases(
        self,
        model: tf.keras.Model,
        phases: List[TrainingPhaseConfig],
        unfreezer: ProgressiveUnfreezer,
        X_train: np.ndarray,
        y_train: np.ndarray,
        loss: str,
        output_dir: Path,
    ) -> List[Dict[str, Any]]:
        """Orchestrate all progressive-unfreezing training phases.

        Args:
            model: Full classification model.
            phases: List of phase configurations.
            unfreezer: ProgressiveUnfreezer for freeze/unfreeze.
            X_train: Windowed training features.
            y_train: Windowed training labels.
            loss: Loss function name.
            output_dir: Directory for checkpoint files.

        Returns:
            List of per-phase history dicts.
        """
        logger.info("═══ Progressive Unfreezing ═══")
        all_histories: List[Dict[str, Any]] = []

        for i, phase_cfg in enumerate(phases):
            unfreezer.apply_phase(model, phase_cfg.frozen)
            history = self.train_phase(
                model=model,
                phase_cfg=phase_cfg,
                X_train=X_train,
                y_train=y_train,
                loss=loss,
                output_dir=output_dir,
                phase_index=i,
            )
            all_histories.append(history)

        return all_histories
