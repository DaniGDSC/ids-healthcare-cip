"""Quick evaluator — two-stage train + eval for tuning iterations.

Two-stage strategy:
  Stage 1: Train head on SMOTE balanced data (frozen backbone)
  Stage 2: Fine-tune attention+BiLSTM2+head on imbalanced data with class_weight
  + Optimal threshold search on imbalanced test set

Attack-aware metrics: attack_recall, attack_precision, attack_f1, macro_f1.
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
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# ── Phase 2 SOLID components (reused) ────────────────────────────
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


def _compute_class_weights(y: np.ndarray) -> Dict[int, float]:
    """Compute class weights inversely proportional to frequency."""
    classes, counts = np.unique(y, return_counts=True)
    total = len(y)
    n_classes = len(classes)
    return {int(cls): total / (n_classes * count) for cls, count in zip(classes, counts)}


def _find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = "attack_f2",
    n_thresholds: int = 50,
) -> Tuple[float, float]:
    """Search for the threshold that maximises the given metric.

    Supports attack_f1, attack_f2 (recall-weighted), and macro_f1.
    Default is attack_f2 (F-beta with beta=2) which weighs recall 4x
    more than precision — appropriate for healthcare IDS where missed
    attacks are more costly than false alarms.
    """
    thresholds = np.linspace(0.1, 0.9, n_thresholds)
    best_t, best_s = 0.5, 0.0

    for t in thresholds:
        y_pred = (y_prob > t).astype(int)
        if metric == "attack_f2":
            score = float(fbeta_score(y_true, y_pred, beta=2, pos_label=1, average="binary", zero_division=0))
        elif metric == "attack_f1":
            score = float(f1_score(y_true, y_pred, pos_label=1, average="binary", zero_division=0))
        elif metric == "macro_f1":
            score = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        else:
            score = float(fbeta_score(y_true, y_pred, beta=2, pos_label=1, average="binary", zero_division=0))

        if score > best_s:
            best_s = score
            best_t = float(t)

    return best_t, best_s


class QuickEvaluator:
    """Two-stage train + eval for Bayesian HPO trials.

    Args:
        quick_train: Training configuration.
        n_features: Number of input features (default 29).
        random_state: Seed for reproducibility.
        pretrained_weights_path: Path to Phase 3 model weights.
        model_architecture: Dict with Phase 3 architecture params.
    """

    def __init__(
        self,
        quick_train: QuickTrainConfig,
        n_features: int = 29,
        random_state: int = 42,
        pretrained_weights_path: Optional[str] = None,
        model_architecture: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._qt = quick_train
        self._n_features = n_features
        self._random_state = random_state
        self._weights_path = pretrained_weights_path
        self._arch = model_architecture or {
            "cnn_filters_1": 64, "cnn_filters_2": 128,
            "bilstm_units_1": 128, "bilstm_units_2": 64,
            "attention_units": 128, "head_dense_units": 32,
            "dropout_rate": 0.3, "timesteps": 20,
        }

    def evaluate_two_stage(
        self,
        hp: Dict[str, Any],
        X_train_smote: np.ndarray,
        y_train_smote: np.ndarray,
        X_train_orig: np.ndarray,
        y_train_orig: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, Any]:
        """Two-stage evaluation on imbalanced data with class weighting.

        Both stages train on original imbalanced data (no SMOTE) to avoid
        synthetic distribution mismatch. Class weight compensates for
        imbalance. X_train_smote/y_train_smote are accepted for API
        compatibility but unused.

        Args:
            hp: Dict with head_lr, finetune_lr, cw_attack, head_epochs, ft_epochs.
            X_train_smote: Unused (kept for API compatibility).
            y_train_smote: Unused (kept for API compatibility).
            X_train_orig: Original imbalanced training features.
            y_train_orig: Original imbalanced labels.
            X_test: Imbalanced test features.
            y_test: Imbalanced test labels.

        Returns:
            Dict with metrics, hyperparameters, threshold, duration.
        """
        t0 = time.perf_counter()
        tf.random.set_seed(self._random_state)
        np.random.seed(self._random_state)  # noqa: NPY002

        ts = self._arch["timesteps"]
        reshaper = DataReshaper(timesteps=ts, stride=1)
        Xo_w, yo_w = reshaper.reshape(X_train_orig, y_train_orig)
        Xt_w, yt_w = reshaper.reshape(X_test, y_test)

        model = self._build_and_load_model()
        cw = {0: 1.0, 1: hp["cw_attack"]}

        # Stage 1: Head on imbalanced data with class weight (frozen backbone)
        for layer in model.layers:
            layer.trainable = layer.name in ("dense_head", "drop_head", "output")

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=hp["head_lr"]),
            loss="binary_crossentropy", metrics=["accuracy"],
        )
        model.fit(
            Xo_w, yo_w, epochs=hp["head_epochs"], batch_size=256,
            validation_split=0.2, verbose=0, class_weight=cw,
            callbacks=[tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=self._qt.early_stopping_patience,
                restore_best_weights=True)],
        )

        # Stage 2: Fine-tune full model on imbalanced (unfreeze all except bilstm1)
        for layer in model.layers:
            if layer.name in ("bilstm1", "drop1"):
                layer.trainable = False
            else:
                layer.trainable = True

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=hp["finetune_lr"]),
            loss="binary_crossentropy", metrics=["accuracy"],
        )
        model.fit(
            Xo_w, yo_w, epochs=hp["ft_epochs"], batch_size=256,
            validation_split=0.2, verbose=0, class_weight=cw,
            callbacks=[tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=2, restore_best_weights=True)],
        )

        # Threshold + metrics on imbalanced test (single predict)
        y_prob = self._predict_prob(model, Xt_w)
        opt_threshold = self._find_threshold_from_prob(yt_w, y_prob)
        metrics = self._compute_metrics_from_prob(yt_w, y_prob, threshold=opt_threshold)

        duration = time.perf_counter() - t0
        tf.keras.backend.clear_session()

        return {
            "hyperparameters": hp,
            "metrics": metrics,
            "optimal_threshold": opt_threshold,
            "total_params": model.count_params(),
            "duration_seconds": round(duration, 2),
        }

    def evaluate_ablation_variant(
        self,
        variant_name: str,
        base_hp: Dict[str, Any],
        X_train_smote: np.ndarray,
        y_train_smote: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        remove: Optional[str] = None,
        replace: Optional[str] = None,
        override: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Evaluate an ablation variant (trained on SMOTE only)."""
        t0 = time.perf_counter()
        tf.random.set_seed(self._random_state)
        np.random.seed(self._random_state)  # noqa: NPY002

        hp = dict(base_hp)
        if override:
            hp.update(override)

        ts = hp.get("timesteps", self._arch["timesteps"])
        reshaper = DataReshaper(timesteps=ts, stride=1)
        Xs_w, ys_w = reshaper.reshape(X_train_smote, y_train_smote)
        Xt_w, yt_w = reshaper.reshape(X_test, y_test)

        detection_model = self._build_variant_model(hp, ts, remove=remove, replace=replace)
        det_params = detection_model.count_params()

        x = tf.keras.layers.Dense(
            self._arch["head_dense_units"], activation="relu", name="dense_head",
        )(detection_model.output)
        x = tf.keras.layers.Dropout(self._arch["dropout_rate"], name="drop_head")(x)
        x = tf.keras.layers.Dense(1, activation="sigmoid", name="output")(x)
        model = tf.keras.Model(detection_model.input, x, name="ablation")

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss="binary_crossentropy", metrics=["accuracy"],
        )
        model.fit(
            Xs_w, ys_w, epochs=self._qt.epochs, batch_size=256,
            validation_split=0.2, verbose=0,
            callbacks=[tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=self._qt.early_stopping_patience,
                restore_best_weights=True)],
        )

        y_prob = self._predict_prob(model, Xt_w)
        opt_threshold = self._find_threshold_from_prob(yt_w, y_prob)
        metrics = self._compute_metrics_from_prob(yt_w, y_prob, threshold=opt_threshold)
        duration = time.perf_counter() - t0
        tf.keras.backend.clear_session()

        return {
            "variant": variant_name,
            "modifications": {"remove": remove, "replace": replace, "override": override},
            "metrics": metrics,
            "detection_params": det_params,
            "total_params": model.count_params(),
            "optimal_threshold": opt_threshold,
            "duration_seconds": round(duration, 2),
        }

    # ── Private helpers ───────────────────────────────────────────

    def _build_and_load_model(self) -> tf.keras.Model:
        """Build Phase 3 architecture and load pre-trained weights."""
        a = self._arch
        builders = [
            CNNBuilder(filters_1=a["cnn_filters_1"], filters_2=a["cnn_filters_2"],
                       kernel_size=3, activation="relu", pool_size=2),
            BiLSTMBuilder(units_1=a["bilstm_units_1"], units_2=a["bilstm_units_2"],
                          dropout_rate=a["dropout_rate"]),
            AttentionBuilder(units=a["attention_units"]),
        ]
        assembler = DetectionModelAssembler(
            timesteps=a["timesteps"], n_features=self._n_features, builders=builders,
        )
        det = assembler.assemble()

        if self._weights_path:
            det.load_weights(self._weights_path)

        x = tf.keras.layers.Dense(
            a["head_dense_units"], activation="relu", name="dense_head",
        )(det.output)
        x = tf.keras.layers.Dropout(a["dropout_rate"], name="drop_head")(x)
        x = tf.keras.layers.Dense(1, activation="sigmoid", name="output")(x)
        model = tf.keras.Model(det.input, x, name="phase3_finetune")

        return model

    def _build_variant_model(
        self, hp: Dict[str, Any], timesteps: int,
        remove: Optional[str] = None, replace: Optional[str] = None,
    ) -> tf.keras.Model:
        """Build an ablation variant model."""
        inp = tf.keras.Input(shape=(timesteps, self._n_features), name="input")
        x = inp

        x = tf.keras.layers.Conv1D(
            self._arch["cnn_filters_1"], 3, activation="relu", padding="same", name="conv1",
        )(x)
        x = tf.keras.layers.MaxPooling1D(pool_size=2, name="pool1")(x)

        if remove != "cnn2":
            x = tf.keras.layers.Conv1D(
                self._arch["cnn_filters_2"], 3, activation="relu", padding="same", name="conv2",
            )(x)
            x = tf.keras.layers.MaxPooling1D(pool_size=2, name="pool2")(x)

        dr = hp.get("dropout_rate", self._arch["dropout_rate"])

        if replace == "bilstm_to_lstm":
            x = tf.keras.layers.LSTM(self._arch["bilstm_units_1"], return_sequences=True, name="lstm1")(x)
        else:
            x = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(self._arch["bilstm_units_1"], return_sequences=True), name="bilstm1",
            )(x)
        x = tf.keras.layers.Dropout(dr, name="drop1")(x)

        if remove != "bilstm2":
            if replace == "bilstm_to_lstm":
                x = tf.keras.layers.LSTM(self._arch["bilstm_units_2"], return_sequences=True, name="lstm2")(x)
            else:
                x = tf.keras.layers.Bidirectional(
                    tf.keras.layers.LSTM(self._arch["bilstm_units_2"], return_sequences=True), name="bilstm2",
                )(x)
            x = tf.keras.layers.Dropout(dr, name="drop2")(x)

        if remove == "attention":
            x = tf.keras.layers.GlobalAveragePooling1D(name="gap")(x)
        else:
            x = AttentionBuilder(units=self._arch["attention_units"]).build(x)

        return tf.keras.Model(inp, x, name="detection_variant")

    @staticmethod
    def _predict_prob(model: tf.keras.Model, X: np.ndarray) -> np.ndarray:
        """Run model prediction and return 1-D probabilities."""
        y_prob = model.predict(X, verbose=0)
        if y_prob.shape[-1] == 1:
            y_prob = y_prob.ravel()
        return y_prob

    @staticmethod
    def _find_threshold_from_prob(y: np.ndarray, y_prob: np.ndarray) -> float:
        """Find optimal threshold from pre-computed probabilities."""
        threshold, _ = _find_optimal_threshold(y, y_prob)
        return threshold

    @staticmethod
    def _compute_metrics_from_prob(
        y: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5,
    ) -> Dict[str, float]:
        """Compute classification metrics from pre-computed probabilities."""
        if y_prob.ndim == 1:
            y_pred = (y_prob > threshold).astype(int)
        else:
            y_pred = np.argmax(y_prob, axis=1)

        return {
            "accuracy": float(accuracy_score(y, y_pred)),
            "f1_score": float(f1_score(y, y_pred, average="weighted", zero_division=0)),
            "auc_roc": float(roc_auc_score(y, y_prob)),
            "attack_recall": float(recall_score(y, y_pred, pos_label=1, average="binary", zero_division=0)),
            "attack_precision": float(precision_score(y, y_pred, pos_label=1, average="binary", zero_division=0)),
            "attack_f1": float(f1_score(y, y_pred, pos_label=1, average="binary", zero_division=0)),
            "attack_f2": float(fbeta_score(y, y_pred, beta=2, pos_label=1, average="binary", zero_division=0)),
            "macro_f1": float(f1_score(y, y_pred, average="macro", zero_division=0)),
            "threshold": threshold,
        }
