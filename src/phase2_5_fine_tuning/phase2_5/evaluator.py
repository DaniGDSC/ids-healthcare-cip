"""Quick evaluator — fast train + eval cycle for tuning iterations.

Builds a detection model with given hyperparameters, adds a classification
head, trains with reduced epochs, and returns metrics.  Never evaluates on
training data (data leakage prevention).

Supports Optuna Hyperband pruning via epoch-level metric reporting.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# ── Phase 2 SOLID components (reused, NOT reimplemented) ────────
from src.phase2_detection_engine.phase2.assembler import DetectionModelAssembler
from src.phase2_detection_engine.phase2.attention_builder import (
    AttentionBuilder,
    BahdanauAttention,  # noqa: F401
)
from src.phase2_detection_engine.phase2.bilstm_builder import BiLSTMBuilder
from src.phase2_detection_engine.phase2.cnn_builder import CNNBuilder
from src.phase2_detection_engine.phase2.reshaper import DataReshaper

from .config import QuickTrainConfig

logger = logging.getLogger(__name__)


# ── Optuna pruning callback for Keras ─────────────────────────────

class _OptunaPruningCallback(tf.keras.callbacks.Callback):
    """Reports epoch-level metrics to Optuna for Hyperband pruning."""

    def __init__(self, trial: Any, metric_key: str = "val_loss") -> None:
        super().__init__()
        self._trial = trial
        self._metric_key = metric_key

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None) -> None:
        import optuna

        logs = logs or {}
        value = logs.get(self._metric_key)
        if value is not None:
            self._trial.report(float(value), epoch)
            if self._trial.should_prune():
                raise optuna.TrialPruned()


class QuickEvaluator:
    """Build, train, and evaluate a model variant in a single call.

    Designed for fast iteration during hyperparameter search and
    ablation studies.  Uses reduced epochs and early stopping.

    Args:
        quick_train: Quick training configuration.
        n_features: Number of input features (default 29).
        random_state: Seed for reproducibility.
    """

    def __init__(
        self,
        quick_train: QuickTrainConfig,
        n_features: int = 29,
        random_state: int = 42,
    ) -> None:
        self._qt = quick_train
        self._n_features = n_features
        self._random_state = random_state

    def evaluate_config(
        self,
        hp: Dict[str, Any],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        seed_override: Optional[int] = None,
        epochs_override: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Build, train, and evaluate a full model with the given hyperparameters.

        Args:
            hp: Hyperparameter dict (cnn_filters_1, bilstm_units_1, etc.).
            X_train: Raw training features (N, F).
            y_train: Training labels.
            X_test: Raw test features (N, F).
            y_test: Test labels.
            seed_override: Override random seed for multi-seed validation.
            epochs_override: Override epoch count for full training.

        Returns:
            Dict with metrics, hyperparameters, duration, and parameter count.
        """
        t0 = time.perf_counter()
        seed = seed_override if seed_override is not None else self._random_state
        epochs = epochs_override if epochs_override is not None else self._qt.epochs

        tf.random.set_seed(seed)
        np.random.seed(seed)  # noqa: NPY002

        timesteps = hp.get("timesteps", 20)
        stride = hp.get("stride", 1)

        # Reshape
        reshaper = DataReshaper(timesteps=timesteps, stride=stride)
        X_train_w, y_train_w = reshaper.reshape(X_train, y_train)
        X_test_w, y_test_w = reshaper.reshape(X_test, y_test)

        # Build detection model
        detection_model = self._build_detection_model(hp, timesteps)
        detection_params = detection_model.count_params()

        # Add classification head
        full_model, loss = self._add_classification_head(
            detection_model, int(len(np.unique(y_train_w)))
        )

        # Train with reduced epochs
        full_model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=hp.get("learning_rate", 0.001)
            ),
            loss=loss,
            metrics=["accuracy"],
        )

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=self._qt.early_stopping_patience,
                restore_best_weights=True,
            ),
        ]

        history = full_model.fit(
            X_train_w,
            y_train_w,
            epochs=epochs,
            batch_size=hp.get("batch_size", 256),
            validation_split=self._qt.validation_split,
            callbacks=callbacks,
            verbose=0,
        )

        # Evaluate on test set ONLY
        metrics = self._compute_metrics(full_model, X_test_w, y_test_w)

        duration = time.perf_counter() - t0

        # Clean up to free GPU memory
        tf.keras.backend.clear_session()

        return {
            "hyperparameters": hp,
            "metrics": metrics,
            "detection_params": detection_params,
            "total_params": full_model.count_params(),
            "epochs_run": len(history.history["loss"]),
            "final_val_loss": float(history.history["val_loss"][-1]),
            "duration_seconds": round(duration, 2),
        }

    def evaluate_config_with_pruning(
        self,
        hp: Dict[str, Any],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        trial: Any,
        metric: str = "f1_score",
        epochs: int = 3,
    ) -> Dict[str, Any]:
        """Train with Optuna Hyperband pruning — reports per-epoch metrics.

        If the trial is pruned, optuna.TrialPruned is raised (caught by Optuna).

        Args:
            hp: Hyperparameter dict.
            X_train: Raw training features.
            y_train: Training labels.
            X_test: Raw test features.
            y_test: Test labels.
            trial: Optuna trial for pruning reports.
            metric: Metric name used for pruning decisions.
            epochs: Max epochs to train.

        Returns:
            Dict with metrics, hyperparameters, duration, and parameter count.
        """
        t0 = time.perf_counter()

        tf.random.set_seed(self._random_state)
        np.random.seed(self._random_state)  # noqa: NPY002

        timesteps = hp.get("timesteps", 20)
        stride = hp.get("stride", 1)

        reshaper = DataReshaper(timesteps=timesteps, stride=stride)
        X_train_w, y_train_w = reshaper.reshape(X_train, y_train)
        X_test_w, y_test_w = reshaper.reshape(X_test, y_test)

        detection_model = self._build_detection_model(hp, timesteps)
        detection_params = detection_model.count_params()

        full_model, loss = self._add_classification_head(
            detection_model, int(len(np.unique(y_train_w)))
        )

        full_model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=hp.get("learning_rate", 0.001)
            ),
            loss=loss,
            metrics=["accuracy"],
        )

        # Use val_accuracy for pruning (higher = better, aligns with maximize)
        pruning_metric = "val_accuracy"
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=self._qt.early_stopping_patience,
                restore_best_weights=True,
            ),
            _OptunaPruningCallback(trial, metric_key=pruning_metric),
        ]

        history = full_model.fit(
            X_train_w,
            y_train_w,
            epochs=epochs,
            batch_size=hp.get("batch_size", 256),
            validation_split=self._qt.validation_split,
            callbacks=callbacks,
            verbose=0,
        )

        metrics = self._compute_metrics(full_model, X_test_w, y_test_w)
        duration = time.perf_counter() - t0

        tf.keras.backend.clear_session()

        return {
            "hyperparameters": hp,
            "metrics": metrics,
            "detection_params": detection_params,
            "total_params": full_model.count_params(),
            "epochs_run": len(history.history["loss"]),
            "final_val_loss": float(history.history["val_loss"][-1]),
            "duration_seconds": round(duration, 2),
        }

    def evaluate_ablation_variant(
        self,
        variant_name: str,
        base_hp: Dict[str, Any],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        remove: Optional[str] = None,
        replace: Optional[str] = None,
        override: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Evaluate an ablation variant with modified architecture."""
        t0 = time.perf_counter()

        tf.random.set_seed(self._random_state)
        np.random.seed(self._random_state)  # noqa: NPY002

        hp = dict(base_hp)
        if override:
            hp.update(override)

        timesteps = hp.get("timesteps", 20)
        stride = hp.get("stride", 1)

        reshaper = DataReshaper(timesteps=timesteps, stride=stride)
        X_train_w, y_train_w = reshaper.reshape(X_train, y_train)
        X_test_w, y_test_w = reshaper.reshape(X_test, y_test)

        detection_model = self._build_variant_model(
            hp, timesteps, remove=remove, replace=replace
        )
        detection_params = detection_model.count_params()

        full_model, loss = self._add_classification_head(
            detection_model, int(len(np.unique(y_train_w)))
        )

        full_model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=hp.get("learning_rate", 0.001)
            ),
            loss=loss,
            metrics=["accuracy"],
        )

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=self._qt.early_stopping_patience,
                restore_best_weights=True,
            ),
        ]

        history = full_model.fit(
            X_train_w,
            y_train_w,
            epochs=self._qt.epochs,
            batch_size=hp.get("batch_size", 256),
            validation_split=self._qt.validation_split,
            callbacks=callbacks,
            verbose=0,
        )

        metrics = self._compute_metrics(full_model, X_test_w, y_test_w)
        duration = time.perf_counter() - t0

        tf.keras.backend.clear_session()

        return {
            "variant": variant_name,
            "hyperparameters": hp,
            "modifications": {
                "remove": remove,
                "replace": replace,
                "override": override,
            },
            "metrics": metrics,
            "detection_params": detection_params,
            "total_params": full_model.count_params(),
            "epochs_run": len(history.history["loss"]),
            "final_val_loss": float(history.history["val_loss"][-1]),
            "duration_seconds": round(duration, 2),
        }

    # ── Private helpers ───────────────────────────────────────────

    def _build_detection_model(
        self, hp: Dict[str, Any], timesteps: int
    ) -> tf.keras.Model:
        """Build standard CNN-BiLSTM-Attention detection model."""
        builders = [
            CNNBuilder(
                filters_1=hp.get("cnn_filters_1", 64),
                filters_2=hp.get("cnn_filters_2", 128),
                kernel_size=hp.get("cnn_kernel_size", 3),
                activation="relu",
                pool_size=2,
            ),
            BiLSTMBuilder(
                units_1=hp.get("bilstm_units_1", 128),
                units_2=hp.get("bilstm_units_2", 64),
                dropout_rate=hp.get("dropout_rate", 0.3),
            ),
            AttentionBuilder(units=hp.get("attention_units", 128)),
        ]
        assembler = DetectionModelAssembler(
            timesteps=timesteps,
            n_features=self._n_features,
            builders=builders,
        )
        return assembler.assemble()

    def _build_variant_model(
        self,
        hp: Dict[str, Any],
        timesteps: int,
        remove: Optional[str] = None,
        replace: Optional[str] = None,
    ) -> tf.keras.Model:
        """Build an ablation variant model with modified architecture."""
        inp = tf.keras.Input(shape=(timesteps, self._n_features), name="input")
        x = inp

        x = tf.keras.layers.Conv1D(
            hp.get("cnn_filters_1", 64), hp.get("cnn_kernel_size", 3),
            activation="relu", padding="same", name="conv1",
        )(x)
        x = tf.keras.layers.MaxPooling1D(pool_size=2, name="pool1")(x)

        if remove != "cnn2":
            x = tf.keras.layers.Conv1D(
                hp.get("cnn_filters_2", 128), hp.get("cnn_kernel_size", 3),
                activation="relu", padding="same", name="conv2",
            )(x)
            x = tf.keras.layers.MaxPooling1D(pool_size=2, name="pool2")(x)

        dropout_rate = hp.get("dropout_rate", 0.3)

        if replace == "bilstm_to_lstm":
            x = tf.keras.layers.LSTM(
                hp.get("bilstm_units_1", 128),
                return_sequences=True, name="lstm1",
            )(x)
        else:
            x = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(
                    hp.get("bilstm_units_1", 128),
                    return_sequences=True,
                ),
                name="bilstm1",
            )(x)
        x = tf.keras.layers.Dropout(dropout_rate, name="drop1")(x)

        if remove != "bilstm2":
            if replace == "bilstm_to_lstm":
                x = tf.keras.layers.LSTM(
                    hp.get("bilstm_units_2", 64),
                    return_sequences=True, name="lstm2",
                )(x)
            else:
                x = tf.keras.layers.Bidirectional(
                    tf.keras.layers.LSTM(
                        hp.get("bilstm_units_2", 64),
                        return_sequences=True,
                    ),
                    name="bilstm2",
                )(x)
            x = tf.keras.layers.Dropout(dropout_rate, name="drop2")(x)

        if remove == "attention":
            x = tf.keras.layers.GlobalAveragePooling1D(name="gap")(x)
        else:
            x = AttentionBuilder(
                units=hp.get("attention_units", 128)
            ).build(x)

        return tf.keras.Model(inp, x, name="detection_variant")

    def _add_classification_head(
        self,
        detection_model: tf.keras.Model,
        n_classes: int,
    ) -> Tuple[tf.keras.Model, str]:
        """Attach a classification head to the detection backbone."""
        x = tf.keras.layers.Dense(
            self._qt.dense_units,
            activation=self._qt.dense_activation,
            name="dense_head",
        )(detection_model.output)
        x = tf.keras.layers.Dropout(
            self._qt.head_dropout_rate, name="drop_head"
        )(x)

        if n_classes == 2:
            output = tf.keras.layers.Dense(1, activation="sigmoid", name="output")(x)
            loss = "binary_crossentropy"
        else:
            output = tf.keras.layers.Dense(
                n_classes, activation="softmax", name="output"
            )(x)
            loss = "categorical_crossentropy"

        full_model = tf.keras.Model(
            detection_model.input, output, name="tuning_model"
        )
        return full_model, loss

    @staticmethod
    def _compute_metrics(
        model: tf.keras.Model,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, float]:
        """Compute classification metrics on test set only."""
        y_pred_prob = model.predict(X_test, verbose=0)

        if y_pred_prob.shape[-1] == 1:
            y_pred_prob = y_pred_prob.ravel()
            y_pred = (y_pred_prob > 0.5).astype(int)
        else:
            y_pred = np.argmax(y_pred_prob, axis=1)

        return {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "f1_score": float(f1_score(y_test, y_pred, average="weighted")),
            "precision": float(precision_score(y_test, y_pred, average="weighted")),
            "recall": float(recall_score(y_test, y_pred, average="weighted")),
            "auc_roc": float(roc_auc_score(y_test, y_pred_prob)),
        }
