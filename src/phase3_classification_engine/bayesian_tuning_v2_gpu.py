# Bayesian Optimization v2 — GPU accelerated:
#  - Hardware: TensorFlow GPU with mixed_float16 precision
#  - Batch size increased: 64 → 128 (GPU optimization)
#  - tf.data pipeline with AUTOTUNE prefetching
#  - Objective: val_AUC (not accuracy — imbalanced dataset)
#  - class_weight applied in every trial
#  - Output layer: float32 (required for mixed precision)
#  - Test set never accessed during tuning
#  - random_state=42 — reproducible across CPU/GPU

"""Phase 3 — Bayesian hyperparameter optimization with GPU acceleration.

Uses keras_tuner.BayesianOptimization to find optimal CNN-BiLSTM-Attention
hyperparameters.  Objective: val_AUC (better than accuracy for imbalanced
data).  Includes ablation study comparing CNN, CNN+BiLSTM, and full
architectures.
"""

from __future__ import annotations

import gc
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import keras_tuner as kt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.phase2_detection_engine.phase2.attention_builder import BahdanauAttention
from src.phase2_detection_engine.phase2.reshaper import DataReshaper

# ── Logging ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  [%(name)s]  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
TRAIN_PATH: Path = PROJECT_ROOT / "data" / "processed" / "train_phase1.parquet"
OUTPUT_DIR: Path = PROJECT_ROOT / "data" / "phase3"
REPORT_DIR: Path = PROJECT_ROOT / "results" / "phase0_analysis"
TUNING_DIR: str = str(PROJECT_ROOT / "tuning_logs_v2")
TB_DIR: str = str(PROJECT_ROOT / "tensorboard_logs")

# ── Constants ────────────────────────────────────────────────────────────
N_FEATURES: int = 29
LABEL_COLUMN: str = "Label"
BATCH_SIZE: int = 128
RANDOM_STATE: int = 42
PATIENCE: int = 5
TUNER_PATIENCE: int = 3
MAX_TRIALS: int = 20
EXECUTIONS_PER_TRIAL: int = 2
PHASE_A_EPOCHS: int = 20
PHASE_B_EPOCHS: int = 15
PHASE_C_EPOCHS: int = 15
KERNEL_SIZE: int = 3
ATTENTION_UNITS: int = 128
VALIDATION_SPLIT: float = 0.2

# Pre-SMOTE original counts
NORMAL_COUNT_ORIGINAL: int = 9_990
ATTACK_COUNT_ORIGINAL: int = 1_432
TOTAL_ORIGINAL: int = NORMAL_COUNT_ORIGINAL + ATTACK_COUNT_ORIGINAL

# Search space
TIMESTEP_CHOICES: List[int] = [10, 20, 30]
CNN_FILTER_CHOICES: List[int] = [64, 128]
BILSTM_UNIT_CHOICES: List[int] = [64, 128, 256]
DROPOUT_CHOICES: List[float] = [0.2, 0.3, 0.5]
LR_CHOICES: List[float] = [0.001, 0.0005]
DENSE_UNIT_CHOICES: List[int] = [32, 64, 128]

# V2 baseline (test-set AUC from retrain_v2)
V2_AUC_BASELINE: float = 0.7243
V2_ATTACK_RECALL: float = 0.5768

# Fallback hyperparameters — Phase 2 defaults
V2_DEFAULT_HP: Dict[str, Any] = {
    "timesteps": 20,
    "cnn_filters": 64,
    "bilstm_units": 128,
    "dropout_rate": 0.3,
    "learning_rate_a": 0.001,
    "dense_units": 64,
}


# ── GPU Setup ────────────────────────────────────────────────────────────


def configure_gpu() -> Dict[str, Any]:
    """Configure GPU memory growth and mixed precision.

    Must be called BEFORE any model operation.

    Returns:
        Dict with GPU configuration details.
    """
    gpu_info: Dict[str, Any] = {
        "gpu_available": False,
        "gpu_count": 0,
        "gpu_names": [],
        "mixed_precision": False,
    }

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            gpu_info["gpu_available"] = True
            gpu_info["gpu_count"] = len(gpus)
            gpu_info["gpu_names"] = [g.name for g in gpus]
            logger.info("GPU enabled: %d device(s) found", len(gpus))
            logger.info("GPU devices: %s", [g.name for g in gpus])
        except RuntimeError as e:
            logger.warning("GPU config error: %s", e)
    else:
        logger.warning("WARNING: No GPU found — falling back to CPU")

    # Mixed precision — only beneficial on GPU hardware
    if gpus:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        gpu_info["mixed_precision"] = True
        logger.info("Mixed precision: float16 enabled")
    else:
        logger.info("Mixed precision: disabled (CPU mode)")

    # Reproducibility
    tf.random.set_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    logger.info("TensorFlow version: %s", tf.__version__)

    return gpu_info


# ── Data Loading ─────────────────────────────────────────────────────────


def _load_and_split_train() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load training data and split into tuning train/val sets.

    Test data is NEVER loaded — prevents data leakage.

    Returns:
        (X_train_flat, y_train_flat, X_val_flat, y_val_flat) as 2D arrays.
    """
    logger.info("── Loading training data (test set excluded) ──")

    train_df = pd.read_parquet(TRAIN_PATH)
    y_all = train_df[LABEL_COLUMN].values.astype(np.float32)
    X_all = train_df.drop(columns=[LABEL_COLUMN]).values.astype(np.float32)

    logger.info("  Full train set: %s", X_all.shape)

    X_train, X_val, y_train, y_val = train_test_split(
        X_all,
        y_all,
        test_size=VALIDATION_SPLIT,
        random_state=RANDOM_STATE,
        stratify=y_all,
    )

    logger.info(
        "  Tuning on %d samples, validating on %d samples",
        len(X_train),
        len(X_val),
    )

    return X_train, y_train, X_val, y_val


def _precompute_windows(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Pre-compute windowed datasets for all timestep values.

    Args:
        X_train: Flat training features (2D).
        y_train: Training labels (1D).
        X_val: Flat validation features (2D).
        y_val: Validation labels (1D).

    Returns:
        Dict mapping timesteps to (X_train_w, y_train_w, X_val_w, y_val_w).
    """
    logger.info("── Pre-computing windowed datasets ──")
    cache: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}

    for ts in TIMESTEP_CHOICES:
        reshaper = DataReshaper(timesteps=ts, stride=1)
        X_t, y_t = reshaper.reshape(X_train, y_train)
        X_v, y_v = reshaper.reshape(X_val, y_val)
        cache[ts] = (X_t, y_t, X_v, y_v)
        logger.info(
            "  timesteps=%d: train=%s, val=%s",
            ts,
            X_t.shape,
            X_v.shape,
        )

    return cache


# ── Class Weight ─────────────────────────────────────────────────────────


def _compute_class_weight() -> Dict[int, float]:
    """Compute class weights from original pre-SMOTE distribution.

    Returns:
        Dict mapping class label to weight.
    """
    w_normal = TOTAL_ORIGINAL / (2 * NORMAL_COUNT_ORIGINAL)
    w_attack = TOTAL_ORIGINAL / (2 * ATTACK_COUNT_ORIGINAL)
    cw = {0: w_normal, 1: w_attack}
    logger.info(
        "  class_weight: Normal=%.3f, Attack=%.3f (%.1fx ratio)",
        w_normal,
        w_attack,
        w_attack / w_normal,
    )
    return cw


# ── HyperModel ───────────────────────────────────────────────────────────


class IoMTHyperModel(kt.HyperModel):
    """Tunable CNN-BiLSTM-Attention for IoMT intrusion detection.

    Handles variable timesteps by selecting pre-computed windowed
    datasets in fit() based on the trial's hyperparameters.

    Args:
        windowed_cache: Pre-computed {timesteps: (X_t, y_t, X_v, y_v)}.
        class_weight: Per-class loss weights.
    """

    def __init__(
        self,
        windowed_cache: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
        class_weight: Dict[int, float],
    ) -> None:
        super().__init__()
        self.windowed_cache = windowed_cache
        self.class_weight = class_weight

    def build(self, hp: kt.HyperParameters) -> tf.keras.Model:
        """Build CNN-BiLSTM-Attention model with trial hyperparameters.

        Args:
            hp: Keras Tuner hyperparameters.

        Returns:
            Compiled Keras model.
        """
        timesteps = hp.Choice("timesteps", TIMESTEP_CHOICES)
        cnn_filters = hp.Choice("cnn_filters", CNN_FILTER_CHOICES)
        bilstm_units = hp.Choice("bilstm_units", BILSTM_UNIT_CHOICES)
        dropout_rate = hp.Choice("dropout_rate", DROPOUT_CHOICES)
        lr_a = hp.Choice("learning_rate_a", LR_CHOICES)
        dense_units = hp.Choice("dense_units", DENSE_UNIT_CHOICES)

        inp = tf.keras.Input(shape=(timesteps, N_FEATURES), name="input")

        # CNN block
        x = tf.keras.layers.Conv1D(
            cnn_filters,
            KERNEL_SIZE,
            activation="relu",
            padding="same",
            name="conv1",
        )(inp)
        x = tf.keras.layers.MaxPooling1D(2, name="pool1")(x)
        x = tf.keras.layers.Conv1D(
            cnn_filters * 2,
            KERNEL_SIZE,
            activation="relu",
            padding="same",
            name="conv2",
        )(x)
        x = tf.keras.layers.MaxPooling1D(2, name="pool2")(x)

        # BiLSTM block
        x = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(bilstm_units, return_sequences=True),
            name="bilstm1",
        )(x)
        x = tf.keras.layers.Dropout(dropout_rate, name="drop1")(x)
        x = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(bilstm_units // 2, return_sequences=True),
            name="bilstm2",
        )(x)
        x = tf.keras.layers.Dropout(dropout_rate, name="drop2")(x)

        # Attention
        x = BahdanauAttention(units=ATTENTION_UNITS, name="attention")(x)

        # Classification head
        x = tf.keras.layers.Dense(dense_units, activation="relu", name="dense_head")(x)
        x = tf.keras.layers.Dropout(dropout_rate, name="drop_head")(x)
        # float32 output required for mixed precision
        x = tf.keras.layers.Dense(1, activation="sigmoid", dtype="float32", name="output")(x)

        model = tf.keras.Model(inp, x, name="classification_tuning")
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr_a),
            loss="binary_crossentropy",
            metrics=[
                tf.keras.metrics.AUC(name="auc"),
                tf.keras.metrics.Recall(name="recall"),
                tf.keras.metrics.Precision(name="precision"),
            ],
        )
        return model

    def fit(
        self,
        hp: kt.HyperParameters,
        model: tf.keras.Model,
        *args: Any,
        **kwargs: Any,
    ) -> tf.keras.callbacks.History:
        """Train model with timestep-appropriate windowed data.

        Overrides default fit to select the correct pre-computed
        windowed dataset based on the trial's timesteps choice.

        Args:
            hp: Keras Tuner hyperparameters.
            model: Compiled model from build().
            *args: Positional args from tuner.search() (ignored).
            **kwargs: Keyword args from tuner.search() (callbacks used).

        Returns:
            Training History object.
        """
        timesteps = hp.get("timesteps")
        X_t, y_t, X_v, y_v = self.windowed_cache[timesteps]

        train_ds = (
            tf.data.Dataset.from_tensor_slices((X_t, y_t))
            .shuffle(buffer_size=10000, seed=RANDOM_STATE)
            .batch(BATCH_SIZE)
            .prefetch(tf.data.AUTOTUNE)
        )
        val_ds = (
            tf.data.Dataset.from_tensor_slices((X_v, y_v))
            .batch(BATCH_SIZE)
            .prefetch(tf.data.AUTOTUNE)
        )

        return model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=PHASE_A_EPOCHS,
            class_weight=self.class_weight,
            callbacks=kwargs.get("callbacks", []),
            verbose=0,
        )


# ── Ablation Study ───────────────────────────────────────────────────────

ABLATION_VARIANTS: List[Tuple[str, str]] = [
    ("cnn", "A: CNN"),
    ("cnn_bilstm", "B: CNN+BiLSTM"),
    ("full", "C: Full"),
]


def _build_ablation_variant(
    variant: str,
    timesteps: int,
    cnn_filters: int,
    bilstm_units: int,
    dropout_rate: float,
    lr_a: float,
    dense_units: int,
) -> tf.keras.Model:
    """Build an ablation model variant.

    Args:
        variant: One of "cnn", "cnn_bilstm", "full".
        timesteps: Window length.
        cnn_filters: Base CNN filter count (doubled for conv2).
        bilstm_units: BiLSTM units per direction (halved for layer 2).
        dropout_rate: Dropout rate for all dropout layers.
        lr_a: Learning rate for Adam optimizer.
        dense_units: Head dense layer units.

    Returns:
        Compiled model for the specified variant.
    """
    inp = tf.keras.Input(shape=(timesteps, N_FEATURES), name="input")

    # CNN — always present
    x = tf.keras.layers.Conv1D(
        cnn_filters,
        KERNEL_SIZE,
        activation="relu",
        padding="same",
        name="conv1",
    )(inp)
    x = tf.keras.layers.MaxPooling1D(2, name="pool1")(x)
    x = tf.keras.layers.Conv1D(
        cnn_filters * 2,
        KERNEL_SIZE,
        activation="relu",
        padding="same",
        name="conv2",
    )(x)
    x = tf.keras.layers.MaxPooling1D(2, name="pool2")(x)

    if variant in ("cnn_bilstm", "full"):
        # BiLSTM block
        x = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(bilstm_units, return_sequences=True),
            name="bilstm1",
        )(x)
        x = tf.keras.layers.Dropout(dropout_rate, name="drop1")(x)
        x = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                bilstm_units // 2,
                return_sequences=(variant == "full"),
            ),
            name="bilstm2",
        )(x)
        x = tf.keras.layers.Dropout(dropout_rate, name="drop2")(x)

    if variant == "full":
        # Attention reduces temporal dim
        x = BahdanauAttention(units=ATTENTION_UNITS, name="attention")(x)
    elif variant == "cnn":
        # Flatten temporal dim for CNN-only
        x = tf.keras.layers.GlobalAveragePooling1D(name="gap")(x)
    # cnn_bilstm: bilstm2 return_sequences=False → already flat

    # Classification head
    x = tf.keras.layers.Dense(dense_units, activation="relu", name="dense_head")(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="drop_head")(x)
    x = tf.keras.layers.Dense(1, activation="sigmoid", dtype="float32", name="output")(x)

    name_map = {
        "cnn": "ablation_cnn",
        "cnn_bilstm": "ablation_cnn_bilstm",
        "full": "ablation_full",
    }
    model = tf.keras.Model(inp, x, name=name_map[variant])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_a),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.Precision(name="precision"),
        ],
    )
    return model


def _evaluate_model(
    model: tf.keras.Model,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> Dict[str, Any]:
    """Evaluate a model on validation data at threshold=0.5.

    Args:
        model: Trained model.
        X_val: Validation features (windowed).
        y_val: Validation labels.

    Returns:
        Metrics dict with AUC, F1, attack_recall, params.
    """
    y_prob = model.predict(X_val, batch_size=BATCH_SIZE, verbose=0)
    y_prob = y_prob.ravel()
    y_pred = (y_prob >= 0.5).astype(int)

    return {
        "auc_roc": float(roc_auc_score(y_val, y_prob)),
        "f1_score": float(f1_score(y_val, y_pred, average="weighted")),
        "accuracy": float(accuracy_score(y_val, y_pred)),
        "precision": float(precision_score(y_val, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_val, y_pred, average="weighted")),
        "attack_recall": float(recall_score(y_val, y_pred, pos_label=1, zero_division=0)),
        "params": model.count_params(),
    }


def _run_ablation_study(
    best_hp: Dict[str, Any],
    windowed_cache: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    class_weight: Dict[int, float],
) -> List[Dict[str, Any]]:
    """Train and evaluate 3 ablation model variants.

    Args:
        best_hp: Best hyperparameters from tuner.
        windowed_cache: Pre-computed windowed data.
        class_weight: Per-class weights.

    Returns:
        List of per-variant result dicts.
    """
    logger.info("═══ Ablation Study ═══")

    ts = best_hp["timesteps"]
    X_t, y_t, X_v, y_v = windowed_cache[ts]

    train_ds = (
        tf.data.Dataset.from_tensor_slices((X_t, y_t))
        .shuffle(buffer_size=10000, seed=RANDOM_STATE)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = (
        tf.data.Dataset.from_tensor_slices((X_v, y_v)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    )

    results: List[Dict[str, Any]] = []

    for variant_key, variant_label in ABLATION_VARIANTS:
        logger.info("── %s ──", variant_label)

        model = _build_ablation_variant(
            variant=variant_key,
            timesteps=ts,
            cnn_filters=best_hp["cnn_filters"],
            bilstm_units=best_hp["bilstm_units"],
            dropout_rate=best_hp["dropout_rate"],
            lr_a=best_hp["learning_rate_a"],
            dense_units=best_hp["dense_units"],
        )

        logger.info("  Params: %d", model.count_params())

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_auc",
                patience=PATIENCE,
                mode="max",
                restore_best_weights=True,
            ),
        ]

        t0 = time.time()
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=PHASE_A_EPOCHS,
            class_weight=class_weight,
            callbacks=callbacks,
            verbose=0,
        )
        elapsed = time.time() - t0

        metrics = _evaluate_model(model, X_v, y_v)
        metrics["variant"] = variant_label
        metrics["train_time_s"] = round(elapsed, 1)
        metrics["epochs_run"] = len(history.history["loss"])
        results.append(metrics)

        logger.info(
            "  %s — AUC=%.4f, F1=%.4f, Atk Recall=%.4f, " "%d params, %.1fs",
            variant_label,
            metrics["auc_roc"],
            metrics["f1_score"],
            metrics["attack_recall"],
            metrics["params"],
            elapsed,
        )

        del model
        tf.keras.backend.clear_session()
        gc.collect()

    # Log comparison table
    logger.info("")
    header = (
        f"{'Model':<20} | {'AUC':>6} | {'F1':>6} "
        f"| {'Atk Recall':>10} | {'Params':>8} | {'Time':>6}"
    )
    logger.info("  %s", header)
    logger.info("  %s", "-" * len(header))
    for r in results:
        logger.info(
            "  %-20s | %6.4f | %6.4f | %10.4f | %8d | %5.1fs",
            r["variant"],
            r["auc_roc"],
            r["f1_score"],
            r["attack_recall"],
            r["params"],
            r["train_time_s"],
        )

    return results


# ── GPU Memory Logging ───────────────────────────────────────────────────


def _log_gpu_memory() -> None:
    """Log current GPU memory usage if available."""
    try:
        mem_info = tf.config.experimental.get_memory_info("GPU:0")
        mem_gb = mem_info["current"] / 1e9
        logger.info("  GPU memory used: %.2f GB", mem_gb)
    except (ValueError, RuntimeError):
        pass  # No GPU or not supported


# ── Export ────────────────────────────────────────────────────────────────


def _export_results(
    best_hp: Dict[str, Any],
    best_val_auc: float,
    ablation_results: List[Dict[str, Any]],
    gpu_info: Dict[str, Any],
    tuning_time_s: float,
) -> None:
    """Export best hyperparameters, ablation results, and summary.

    Args:
        best_hp: Best hyperparameters from tuner.
        best_val_auc: Best validation AUC achieved.
        ablation_results: Ablation study results.
        gpu_info: GPU configuration info.
        tuning_time_s: Total tuning wall time in seconds.
    """
    logger.info("── Export ──")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    gpu_name = gpu_info["gpu_names"][0] if gpu_info["gpu_names"] else "N/A"

    # 1. best_hyperparams_v2.json
    hp_path = OUTPUT_DIR / "best_hyperparams_v2.json"
    hp_report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "timesteps": best_hp["timesteps"],
        "cnn_filters": best_hp["cnn_filters"],
        "bilstm_units": best_hp["bilstm_units"],
        "dropout_rate": best_hp["dropout_rate"],
        "learning_rate_a": best_hp["learning_rate_a"],
        "dense_units": best_hp["dense_units"],
        "val_auc": round(float(best_val_auc), 6),
        "beats_v2_baseline": bool(best_val_auc > V2_AUC_BASELINE),
        "v2_baseline_auc": V2_AUC_BASELINE,
        "gpu_used": gpu_info["gpu_available"],
        "gpu_name": gpu_name,
        "mixed_precision": gpu_info["mixed_precision"],
        "tuning_trials": MAX_TRIALS,
        "executions_per_trial": EXECUTIONS_PER_TRIAL,
        "objective": "val_auc",
        "tuning_time_s": round(tuning_time_s, 1),
    }
    with open(hp_path, "w") as f:
        json.dump(hp_report, f, indent=2)
    logger.info("  Saved %s", hp_path.name)

    # 2. ablation_results_v2.json
    abl_path = OUTPUT_DIR / "ablation_results_v2.json"
    abl_report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "best_hyperparameters": best_hp,
        "results": ablation_results,
    }
    with open(abl_path, "w") as f:
        json.dump(abl_report, f, indent=2)
    logger.info("  Saved %s", abl_path.name)

    # 3. tuning_summary.md
    summary_path = REPORT_DIR / "tuning_summary.md"
    beats = "YES" if best_val_auc > V2_AUC_BASELINE else "NO"

    lines = [
        "## Bayesian Hyperparameter Optimization — Summary",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "### Search Configuration",
        "",
        "- Objective: val_AUC (maximize)",
        f"- Max trials: {MAX_TRIALS}",
        f"- Executions per trial: {EXECUTIONS_PER_TRIAL}",
        f"- Hardware: {gpu_name}",
        f"- Mixed precision: " f"{'float16' if gpu_info['mixed_precision'] else 'disabled'}",
        f"- Tuning time: {tuning_time_s:.1f}s "
        f"(~{tuning_time_s / 60:.0f} min, "
        f"{'GPU' if gpu_info['gpu_available'] else 'CPU'} mode)",
        "",
        "### Best Hyperparameters",
        "",
        "| Parameter | Value |",
        "|-----------|-------|",
        f"| timesteps | {best_hp['timesteps']} |",
        f"| cnn_filters | {best_hp['cnn_filters']} |",
        f"| bilstm_units | {best_hp['bilstm_units']} |",
        f"| dropout_rate | {best_hp['dropout_rate']} |",
        f"| learning_rate_a | {best_hp['learning_rate_a']} |",
        f"| dense_units | {best_hp['dense_units']} |",
        "",
        f"**Best val_AUC: {best_val_auc:.4f}**",
        "",
        f"Beats v2 baseline ({V2_AUC_BASELINE}): {beats}",
        "",
        "### Ablation Study",
        "",
        "| Model | AUC | F1 | Attack Recall | Params | Time |",
        "|-------|-----|----|---------------|--------|------|",
    ]

    for r in ablation_results:
        auc_v = f"{r['auc_roc']:.4f}"
        f1_v = f"{r['f1_score']:.4f}"
        atk_v = f"{r['attack_recall']:.4f}"
        lines.append(
            f"| {r['variant']} | {auc_v} | {f1_v} | {atk_v} "
            f"| {r['params']} | {r['train_time_s']:.1f}s |"
        )

    lines.extend(
        [
            "",
            "### Disclosure",
            "",
            "- Test set never accessed during tuning",
            "- class_weight applied in every trial",
            "- random_state=42 for reproducibility",
            f"- Validation split: " f"{VALIDATION_SPLIT * 100:.0f}% of training data",
            "- Models trained from scratch " "(no Phase 2 weight transfer during tuning)",
            "- val_AUC baseline comparison is approximate " "(v2 baseline measured on test set)",
            "",
            "---",
            "",
            "**Reviewer signature:** " "____________________  " "**Date:** ____/____/______",
            "",
        ]
    )

    with open(summary_path, "w") as f:
        f.write("\n".join(lines))
    logger.info("  Saved %s", summary_path.name)


# ── Main ─────────────────────────────────────────────────────────────────


def main() -> None:
    """Execute Bayesian optimization pipeline."""
    # Step 1: GPU setup — FIRST
    gpu_info = configure_gpu()

    logger.info("═══════════════════════════════════════════════════")
    logger.info("  Phase 3 — Bayesian Hyperparameter Optimization")
    logger.info(
        "  Objective: val_AUC | Trials: %d | GPU: %s",
        MAX_TRIALS,
        gpu_info["gpu_available"],
    )
    logger.info("═══════════════════════════════════════════════════")

    device = "GPU" if gpu_info["gpu_available"] else "CPU"
    logger.info("Training device: %s", device)

    total_combos = (
        len(TIMESTEP_CHOICES)
        * len(CNN_FILTER_CHOICES)
        * len(BILSTM_UNIT_CHOICES)
        * len(DROPOUT_CHOICES)
        * len(LR_CHOICES)
        * len(DENSE_UNIT_CHOICES)
    )
    logger.info(
        "Search space: %d combinations, exploring %d trials",
        total_combos,
        MAX_TRIALS,
    )

    # Step 2: Load data (train only — no test set)
    X_train_flat, y_train_flat, X_val_flat, y_val_flat = _load_and_split_train()

    # Step 3: Pre-compute windowed datasets + tf.data
    windowed_cache = _precompute_windows(X_train_flat, y_train_flat, X_val_flat, y_val_flat)
    logger.info("tf.data pipeline configured with AUTOTUNE")

    # Class weight
    class_weight = _compute_class_weight()

    # Step 4-5: Bayesian Optimization
    logger.info("═══ Bayesian Optimization ═══")

    hypermodel = IoMTHyperModel(
        windowed_cache=windowed_cache,
        class_weight=class_weight,
    )

    tuner = kt.BayesianOptimization(
        hypermodel=hypermodel,
        objective=kt.Objective("val_auc", direction="max"),
        max_trials=MAX_TRIALS,
        executions_per_trial=EXECUTIONS_PER_TRIAL,
        directory=TUNING_DIR,
        project_name="iomt_classification_v2",
        overwrite=True,
        seed=RANDOM_STATE,
    )

    tuner.search_space_summary()

    # Step 6: Run search
    t0 = time.time()
    tuner.search(
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_auc",
                patience=TUNER_PATIENCE,
                mode="max",
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir=TB_DIR,
                histogram_freq=1,
            ),
        ],
    )
    tuning_time = time.time() - t0

    _log_gpu_memory()

    tuner.results_summary(num_trials=5)

    # Step 7: Extract best hyperparameters
    logger.info("── Best trial results ──")
    best_hp_obj = tuner.get_best_hyperparameters(1)[0]
    best_hp: Dict[str, Any] = {
        "timesteps": best_hp_obj.get("timesteps"),
        "cnn_filters": best_hp_obj.get("cnn_filters"),
        "bilstm_units": best_hp_obj.get("bilstm_units"),
        "dropout_rate": best_hp_obj.get("dropout_rate"),
        "learning_rate_a": best_hp_obj.get("learning_rate_a"),
        "dense_units": best_hp_obj.get("dense_units"),
    }

    best_trials = tuner.oracle.get_best_trials(1)
    best_val_auc = best_trials[0].score if best_trials else 0.0

    logger.info("  Best hyperparameters: %s", best_hp)
    logger.info("  Best val_AUC: %.4f", best_val_auc)

    if best_val_auc > V2_AUC_BASELINE:
        logger.info("  Beats v2 baseline (%.4f): YES", V2_AUC_BASELINE)
    else:
        logger.warning(
            "  WARNING: Best val_AUC %.4f does NOT beat v2 "
            "baseline %.4f — using tuner result anyway "
            "(v2 fallback: %s)",
            best_val_auc,
            V2_AUC_BASELINE,
            V2_DEFAULT_HP,
        )

    # Step 8: Ablation study
    tf.keras.backend.clear_session()
    gc.collect()

    ablation_results = _run_ablation_study(best_hp, windowed_cache, class_weight)

    # Step 9: Export
    _export_results(best_hp, best_val_auc, ablation_results, gpu_info, tuning_time)

    logger.info("═══════════════════════════════════════════════════")
    logger.info(
        "  Tuning complete — best AUC=%.4f, time=%.1fs",
        best_val_auc,
        tuning_time,
    )
    logger.info("═══════════════════════════════════════════════════")


if __name__ == "__main__":
    main()
