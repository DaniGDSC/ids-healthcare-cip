"""Phase 2: Detection Engine — CNN→BiLSTM→Attention feature extraction.

Pipeline
--------
1. Load Phase 1 parquet artifacts (train/test)
2. Verify SHA-256 against preprocessing_metadata.json
3. Reshape: (N, 29) → (N_win, timesteps, 29) via sliding windows
4. CNN: Conv1D(64)→MaxPool→Conv1D(128)→MaxPool
5. BiLSTM: BiLSTM(128)→Drop→BiLSTM(64)→Drop
6. Attention: Dense(tanh)→Dense(softmax)→Multiply→GlobalAvgPool
7. Export: detection_model.h5, attention_output.parquet, detection_report.json

Output: weighted sum vectors (128-dim context), NOT predictions.
Classification is deferred to Phase 3.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml

# ── Named Constants ────────────────────────────────────────────────────
LABEL_COLUMN: str = "Label"
RANDOM_STATE: int = 42
CNN_PADDING: str = "same"
PREDICT_BATCH_SIZE: int = 256
HASH_CHUNK_SIZE: int = 65_536

# ── Logging ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [Phase2] %(message)s",
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# Step 1 — Load & Verify
# ═══════════════════════════════════════════════════════════════════════


def compute_sha256(path: Path) -> str:
    """Compute SHA-256 hash of a file.

    Args:
        path: Absolute path to the file.

    Returns:
        Hex-encoded SHA-256 digest.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(HASH_CHUNK_SIZE):
            h.update(chunk)
    return h.hexdigest()


def load_and_verify(
    train_path: Path,
    test_path: Path,
    metadata_path: Path,
    report_path: Path,
    label_column: str = LABEL_COLUMN,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Load Phase 1 parquet files, verify SHA-256, separate X and y.

    Args:
        train_path: Path to train_phase1.parquet.
        test_path: Path to test_phase1.parquet.
        metadata_path: Path to preprocessing_metadata.json (SHA-256 hashes).
        report_path: Path to phase1_report.json (feature names).
        label_column: Name of the label column.

    Returns:
        Tuple of (X_train, y_train, X_test, y_test, feature_names).

    Raises:
        ValueError: If SHA-256 hash does not match expected value.
    """
    # Read expected hashes
    with open(metadata_path) as f:
        metadata = json.load(f)
    hashes = metadata["artifact_hashes"]

    # Verify integrity
    for name, fpath in [
        ("train_phase1.parquet", train_path),
        ("test_phase1.parquet", test_path),
    ]:
        actual = compute_sha256(fpath)
        expected = hashes[name]["sha256"]
        if actual != expected:
            raise ValueError(
                f"SHA-256 mismatch: {name} "
                f"({actual[:16]}… ≠ {expected[:16]}…)"
            )
        logger.info("SHA-256 verified: %s", name)

    # Feature names from Phase 1 report
    with open(report_path) as f:
        report = json.load(f)
    feature_names: list[str] = report["output"]["feature_names"]

    # Parquet → NumPy
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    X_train = train_df[feature_names].values.astype(np.float32)
    y_train = train_df[label_column].values.astype(np.int32)
    X_test = test_df[feature_names].values.astype(np.float32)
    y_test = test_df[label_column].values.astype(np.int32)

    logger.info(
        "Loaded — train: %s (%d features), test: %s",
        X_train.shape,
        len(feature_names),
        X_test.shape,
    )
    return X_train, y_train, X_test, y_test, feature_names


# ═══════════════════════════════════════════════════════════════════════
# Step 2 — Reshape (Sliding Windows)
# ═══════════════════════════════════════════════════════════════════════


def create_sliding_windows(
    X: np.ndarray,
    y: np.ndarray,
    timesteps: int,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Create sliding windows: (N, F) → (N_win, timesteps, F).

    The label for each window is the label of the **last** sample
    in the window, following the standard IDS temporal windowing
    convention.

    Args:
        X: 2-D array of shape (n_samples, n_features).
        y: 1-D array of shape (n_samples,).
        timesteps: Window length.
        stride: Step between consecutive windows.

    Returns:
        Tuple of (X_windows, y_windows).

    Raises:
        ValueError: If n_samples < timesteps.
    """
    n_samples, n_features = X.shape
    if n_samples < timesteps:
        raise ValueError(
            f"Cannot create windows: n_samples={n_samples} < timesteps={timesteps}"
        )

    n_windows = (n_samples - timesteps) // stride + 1

    # Vectorised index construction
    starts = np.arange(0, n_windows * stride, stride)
    offsets = np.arange(timesteps)
    indices = starts[:, None] + offsets[None, :]  # (n_windows, timesteps)

    X_windows = X[indices]  # (n_windows, timesteps, n_features)
    y_windows = y[indices[:, -1]]  # last sample per window

    logger.info(
        "Reshape: (%d, %d) → %s  [timesteps=%d, stride=%d]",
        n_samples,
        n_features,
        X_windows.shape,
        timesteps,
        stride,
    )
    return X_windows.astype(np.float32), y_windows


# ═══════════════════════════════════════════════════════════════════════
# Steps 3–5 — Model Architecture (CNN → BiLSTM → Attention)
# ═══════════════════════════════════════════════════════════════════════


@tf.keras.utils.register_keras_serializable(package="phase2")
class BahdanauAttention(tf.keras.layers.Layer):
    """Additive (Bahdanau) attention over a temporal sequence.

    Steps:
        1. Dense(units, tanh)   → score vector per timestep
        2. Dense(1) + softmax   → normalised attention weight per timestep
        3. Multiply             → weighted sequence
        4. GlobalAveragePool    → fixed-length context vector

    Input shape:  (batch, timesteps, features)
    Output shape: (batch, features)
    """

    def __init__(self, units: int = 128, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.units = units
        self.score_dense = tf.keras.layers.Dense(
            units, activation="tanh", name="score"
        )
        self.weight_dense = tf.keras.layers.Dense(
            1, use_bias=False, name="attn_weights"
        )
        self.pool = tf.keras.layers.GlobalAveragePooling1D(name="context_pool")

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Forward pass: compute attention-weighted context vector."""
        scores = self.score_dense(x)  # (B, T, units)
        weights = self.weight_dense(scores)  # (B, T, 1)
        weights = tf.nn.softmax(weights, axis=1)  # normalise over T
        weighted = x * weights  # broadcast multiply
        return self.pool(weighted)  # (B, features)

    def get_config(self) -> dict[str, Any]:
        """Serialisation support for model saving."""
        return {**super().get_config(), "units": self.units}


def build_detection_model(
    timesteps: int,
    n_features: int,
    cnn_filters_1: int,
    cnn_filters_2: int,
    cnn_kernel_size: int,
    cnn_activation: str,
    cnn_pool_size: int,
    bilstm_units_1: int,
    bilstm_units_2: int,
    dropout_rate: float,
    attention_units: int,
) -> tf.keras.Model:
    """Build the CNN→BiLSTM→Attention detection model.

    The model outputs a fixed-length context vector per window.
    There is no classification head — that is Phase 3.

    Args:
        timesteps: Sliding window length.
        n_features: Number of input features per timestep.
        cnn_filters_1: Filters in the first Conv1D layer.
        cnn_filters_2: Filters in the second Conv1D layer.
        cnn_kernel_size: Kernel size for both Conv1D layers.
        cnn_activation: Activation function for Conv1D layers.
        cnn_pool_size: Pool size for MaxPooling1D layers.
        bilstm_units_1: Units in the first BiLSTM layer.
        bilstm_units_2: Units in the second BiLSTM layer.
        dropout_rate: Dropout rate after each BiLSTM layer.
        attention_units: Hidden units in the attention score layer.

    Returns:
        Keras Model with output shape (batch, bilstm_units_2 * 2).
    """
    inp = tf.keras.Input(shape=(timesteps, n_features), name="input")

    # ── CNN block ──────────────────────────────────────────────
    x = tf.keras.layers.Conv1D(
        cnn_filters_1,
        cnn_kernel_size,
        activation=cnn_activation,
        padding=CNN_PADDING,
        name="conv1",
    )(inp)
    x = tf.keras.layers.MaxPooling1D(cnn_pool_size, name="pool1")(x)

    x = tf.keras.layers.Conv1D(
        cnn_filters_2,
        cnn_kernel_size,
        activation=cnn_activation,
        padding=CNN_PADDING,
        name="conv2",
    )(x)
    x = tf.keras.layers.MaxPooling1D(cnn_pool_size, name="pool2")(x)

    # ── BiLSTM block ───────────────────────────────────────────
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(bilstm_units_1, return_sequences=True),
        name="bilstm1",
    )(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="drop1")(x)

    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(bilstm_units_2, return_sequences=True),
        name="bilstm2",
    )(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="drop2")(x)

    # ── Attention block ────────────────────────────────────────
    context = BahdanauAttention(units=attention_units, name="attention")(x)

    return tf.keras.Model(inp, context, name="detection_engine")


# ═══════════════════════════════════════════════════════════════════════
# Step 6 — Artifact Export
# ═══════════════════════════════════════════════════════════════════════


def export_model_weights(
    model: tf.keras.Model,
    output_path: Path,
) -> None:
    """Save model weights in HDF5 format (no classification head).

    Args:
        model: Trained or initialised Keras model.
        output_path: Destination .h5 file.
    """
    model.save_weights(str(output_path))
    logger.info("Model weights saved: %s", output_path)


def export_attention_vectors(
    train_context: np.ndarray,
    test_context: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    output_path: Path,
) -> None:
    """Save attention-weighted context vectors as parquet.

    Args:
        train_context: (n_train_windows, context_dim) float32.
        test_context: (n_test_windows, context_dim) float32.
        y_train: (n_train_windows,) int32 labels.
        y_test: (n_test_windows,) int32 labels.
        output_path: Destination .parquet file.
    """
    context_dim = train_context.shape[1]
    context_cols = [f"attn_{i}" for i in range(context_dim)]

    train_df = pd.DataFrame(train_context, columns=context_cols)
    train_df[LABEL_COLUMN] = y_train
    train_df["split"] = "train"

    test_df = pd.DataFrame(test_context, columns=context_cols)
    test_df[LABEL_COLUMN] = y_test
    test_df["split"] = "test"

    combined = pd.concat([train_df, test_df], ignore_index=True)
    combined.to_parquet(output_path, index=False)
    logger.info("Attention output: %s  shape=%s", output_path, combined.shape)


def export_detection_report(
    model: tf.keras.Model,
    config: dict[str, Any],
    feature_names: list[str],
    train_context: np.ndarray,
    test_context: np.ndarray,
    train_windows_shape: tuple[int, ...],
    test_windows_shape: tuple[int, ...],
    elapsed: float,
    output_path: Path,
) -> dict[str, Any]:
    """Generate and save detection_report.json.

    Args:
        model: The detection model.
        config: Parsed YAML config dict.
        feature_names: List of input feature names.
        train_context: Train context vectors.
        test_context: Test context vectors.
        train_windows_shape: Shape of train sliding windows.
        test_windows_shape: Shape of test sliding windows.
        elapsed: Pipeline wall-clock time in seconds.
        output_path: Destination .json file.

    Returns:
        The report dict (also saved to disk).
    """
    # Collect layer info
    layers_info: list[dict[str, Any]] = []
    for layer in model.layers:
        if layer.name == "input":
            continue
        info: dict[str, Any] = {
            "name": layer.name,
            "type": type(layer).__name__,
        }
        if hasattr(layer, "output") and layer.output is not None:
            out_shape = layer.output.shape
            info["output_shape"] = str(tuple(out_shape))
        info["params"] = int(layer.count_params())
        layers_info.append(info)

    report: dict[str, Any] = {
        "phase": "Phase 2 — Detection Engine",
        "model_name": model.name,
        "architecture": "CNN→BiLSTM→Attention",
        "total_parameters": int(model.count_params()),
        "trainable_parameters": int(
            sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
        ),
        "layers": layers_info,
        "hyperparameters": {
            "timesteps": config["reshape"]["timesteps"],
            "stride": config["reshape"].get("stride", 1),
            "cnn_filters": [config["cnn"]["filters_1"], config["cnn"]["filters_2"]],
            "cnn_kernel_size": config["cnn"]["kernel_size"],
            "cnn_activation": config["cnn"]["activation"],
            "cnn_pool_size": config["cnn"]["pool_size"],
            "bilstm_units": [
                config["bilstm"]["units_1"],
                config["bilstm"]["units_2"],
            ],
            "dropout_rate": config["bilstm"]["dropout_rate"],
            "attention_units": config["attention"]["units"],
            "random_state": config.get("random_state", RANDOM_STATE),
        },
        "input_features": feature_names,
        "n_features": len(feature_names),
        "output_dim": int(train_context.shape[1]),
        "shapes": {
            "train_windows": str(train_windows_shape),
            "test_windows": str(test_windows_shape),
            "train_context": str(train_context.shape),
            "test_context": str(test_context.shape),
        },
        "elapsed_seconds": round(elapsed, 3),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": platform.python_version(),
            "tensorflow": tf.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "platform": platform.platform(),
        },
    }

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Detection report: %s", output_path)

    return report


# ═══════════════════════════════════════════════════════════════════════
# Report Section Generation (§5.1)
# ═══════════════════════════════════════════════════════════════════════


def generate_report_section(
    report: dict[str, Any],
    config: dict[str, Any],
    output_path: Path,
) -> None:
    """Generate report_section_detection.md (§5.1) for thesis.

    Args:
        report: The detection_report.json dict.
        config: Parsed YAML config.
        output_path: Destination .md file.
    """
    hp = report["hyperparameters"]
    shapes = report["shapes"]
    env = report["environment"]

    # Build layer table rows
    layer_rows: list[str] = []
    for layer in report["layers"]:
        params = layer.get("params", 0)
        out_shape = layer.get("output_shape", "—")
        layer_rows.append(
            f"| {layer['name']} | {layer['type']} | {out_shape} | {params:,} |"
        )
    layer_table = "\n".join(layer_rows)

    # Compute CNN output shape for diagram
    cnn_timesteps = hp["timesteps"]
    for _ in range(2):  # two MaxPool layers
        cnn_timesteps = cnn_timesteps // hp["cnn_pool_size"]
    bilstm_out_dim = hp["bilstm_units"][1] * 2  # bidirectional doubles units

    md = f"""## 5.1 Detection Engine Architecture

This section describes the CNN→BiLSTM→Attention feature extraction
model that transforms preprocessed IoMT network traffic into
fixed-length representation vectors for downstream classification
(Phase 3).

### 5.1.1 Architecture Overview

The detection engine implements a three-stage deep feature extractor:

1. **CNN (Convolutional Neural Network):** Extracts local spatial
   patterns from sliding windows of consecutive network events.
2. **BiLSTM (Bidirectional Long Short-Term Memory):** Captures
   temporal dependencies in both forward and backward directions.
3. **Bahdanau Attention:** Computes adaptive weights over timesteps
   to produce a fixed-length context vector (weighted sum).

```
Input: (batch, {hp['timesteps']}, {report['n_features']})
  │
  ├─── [CNN Block]
  │      Conv1D({hp['cnn_filters'][0]}, k={hp['cnn_kernel_size']}, {hp['cnn_activation']}) → MaxPool({hp['cnn_pool_size']})
  │      Conv1D({hp['cnn_filters'][1]}, k={hp['cnn_kernel_size']}, {hp['cnn_activation']}) → MaxPool({hp['cnn_pool_size']})
  │      Output: (batch, {cnn_timesteps}, {hp['cnn_filters'][1]})
  │
  ├─── [BiLSTM Block]
  │      Bidirectional LSTM({hp['bilstm_units'][0]}, return_seq=True) → Dropout({hp['dropout_rate']})
  │      Bidirectional LSTM({hp['bilstm_units'][1]}, return_seq=True) → Dropout({hp['dropout_rate']})
  │      Output: (batch, {cnn_timesteps}, {bilstm_out_dim})
  │
  └─── [Attention Block]
         Dense({hp['attention_units']}, tanh) → Dense(1, softmax) → Multiply → GlobalAvgPool
         Output: (batch, {report['output_dim']}) — context vector
```

### 5.1.2 Layer Summary

| Layer | Type | Output Shape | Parameters |
|-------|------|-------------|------------|
{layer_table}

**Total parameters:** {report['total_parameters']:,}
**Trainable parameters:** {report['trainable_parameters']:,}

### 5.1.3 Sliding Window Reshape

Network traffic samples are grouped into temporal windows of
**{hp['timesteps']}** consecutive events with stride **{hp['stride']}**,
creating a 3-D tensor suitable for 1-D convolution. The label for
each window is the label of the last sample in the window.

| Parameter | Value |
|-----------|-------|
| Window length (timesteps) | {hp['timesteps']} |
| Stride | {hp['stride']} |
| Input features | {report['n_features']} |
| Train windows | {shapes['train_windows']} |
| Test windows | {shapes['test_windows']} |
| Train context | {shapes['train_context']} |
| Test context | {shapes['test_context']} |
| Window label strategy | Last sample in window |

### 5.1.4 Attention Mechanism

The Bahdanau (additive) attention mechanism computes a context
vector by learning to weight each timestep according to its
relevance for intrusion detection:

1. **Score:** `Dense({hp['attention_units']}, tanh)` projects each
   timestep to a score vector
2. **Weight:** `Dense(1)` + `softmax(axis=timesteps)` normalises
   scores to a probability distribution over timesteps
3. **Multiply:** Element-wise multiplication weights the BiLSTM
   output sequence by learned attention weights
4. **Pool:** `GlobalAveragePooling1D` produces the final
   {report['output_dim']}-dimensional context vector

Output dimension: **{report['output_dim']}** (one vector per window).

### 5.1.5 Hyperparameters

| Hyperparameter | Value | Rationale |
|----------------|-------|-----------|
| Timesteps | {hp['timesteps']} | Captures short-term traffic patterns |
| Stride | {hp['stride']} | Maximum window overlap for data coverage |
| CNN filters | {hp['cnn_filters']} | Hierarchical spatial feature extraction |
| CNN kernel size | {hp['cnn_kernel_size']} | 3-sample receptive field |
| CNN activation | {hp['cnn_activation']} | Standard non-linearity |
| BiLSTM units | {hp['bilstm_units']} | Forward + backward temporal context |
| Dropout rate | {hp['dropout_rate']} | Regularisation against overfitting |
| Attention units | {hp['attention_units']} | Score vector dimensionality |
| Random state | {hp['random_state']} | Reproducibility |

### 5.1.6 Output Artifacts

| Artifact | Description |
|----------|-------------|
| `detection_model.h5` | Model weights ({report['total_parameters']:,} parameters, no classification head) |
| `attention_output.parquet` | Weighted sum vectors ({report['output_dim']}-dim) + labels + split indicator |
| `detection_report.json` | Model summary, layer shapes, hyperparameters, environment |

### 5.1.7 Execution Summary

| Metric | Value |
|--------|-------|
| Total execution time | **{report['elapsed_seconds']:.2f} s** |
| Python | {env['python']} |
| TensorFlow | {env['tensorflow']} |
| NumPy | {env['numpy']} |
| pandas | {env['pandas']} |
| Platform | {env['platform']} |
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(md)
    logger.info("Report section: %s", output_path)


# ═══════════════════════════════════════════════════════════════════════
# Main Pipeline
# ═══════════════════════════════════════════════════════════════════════


def run_pipeline(config_path: Path, project_root: Path) -> dict[str, Any]:
    """Execute the complete Phase 2 detection pipeline.

    Args:
        config_path: Path to phase2_config.yaml.
        project_root: Repository root directory.

    Returns:
        The detection_report.json dict.
    """
    t0 = time.perf_counter()

    # Reproducibility seeds
    tf.random.set_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    # Load config
    with open(config_path) as f:
        config: dict[str, Any] = yaml.safe_load(f)
    logger.info("Phase 2 Detection Engine — starting")

    # ── Step 1: Load & Verify ──────────────────────────────────
    X_train, y_train, X_test, y_test, feature_names = load_and_verify(
        train_path=project_root / config["data"]["train_parquet"],
        test_path=project_root / config["data"]["test_parquet"],
        metadata_path=project_root / config["data"]["metadata_file"],
        report_path=project_root / config["data"]["report_file"],
        label_column=config["data"].get("label_column", LABEL_COLUMN),
    )
    n_features = len(feature_names)

    # ── Step 2: Reshape ────────────────────────────────────────
    timesteps = config["reshape"]["timesteps"]
    stride = config["reshape"].get("stride", 1)

    X_train_w, y_train_w = create_sliding_windows(
        X_train, y_train, timesteps, stride
    )
    X_test_w, y_test_w = create_sliding_windows(
        X_test, y_test, timesteps, stride
    )

    # ── Steps 3–5: Build Model ─────────────────────────────────
    cnn_cfg = config["cnn"]
    bilstm_cfg = config["bilstm"]
    attn_cfg = config["attention"]

    model = build_detection_model(
        timesteps=timesteps,
        n_features=n_features,
        cnn_filters_1=cnn_cfg["filters_1"],
        cnn_filters_2=cnn_cfg["filters_2"],
        cnn_kernel_size=cnn_cfg["kernel_size"],
        cnn_activation=cnn_cfg["activation"],
        cnn_pool_size=cnn_cfg["pool_size"],
        bilstm_units_1=bilstm_cfg["units_1"],
        bilstm_units_2=bilstm_cfg["units_2"],
        dropout_rate=bilstm_cfg["dropout_rate"],
        attention_units=attn_cfg["units"],
    )

    model.summary(print_fn=logger.info)

    # Forward pass — extract context vectors (no training)
    train_context = model.predict(
        X_train_w, batch_size=PREDICT_BATCH_SIZE, verbose=0
    )
    test_context = model.predict(
        X_test_w, batch_size=PREDICT_BATCH_SIZE, verbose=0
    )

    logger.info("Train context: %s", train_context.shape)
    logger.info("Test context:  %s", test_context.shape)

    # ── Step 6: Export ─────────────────────────────────────────
    elapsed = time.perf_counter() - t0
    output_dir = project_root / config["output"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    # 6a. Model weights
    export_model_weights(
        model, output_dir / config["output"]["model_file"]
    )

    # 6b. Attention vectors
    export_attention_vectors(
        train_context,
        test_context,
        y_train_w,
        y_test_w,
        output_dir / config["output"]["attention_parquet"],
    )

    # 6c. Detection report
    report = export_detection_report(
        model=model,
        config=config,
        feature_names=feature_names,
        train_context=train_context,
        test_context=test_context,
        train_windows_shape=X_train_w.shape,
        test_windows_shape=X_test_w.shape,
        elapsed=elapsed,
        output_path=output_dir / config["output"]["report_file"],
    )

    # 6d. Thesis report section
    report_section_path = (
        project_root / "results" / "phase0_analysis" / "report_section_detection.md"
    )
    generate_report_section(report, config, report_section_path)

    logger.info("Phase 2 complete in %.2fs", elapsed)
    return report


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    CONFIG_PATH = PROJECT_ROOT / "config" / "phase2_config.yaml"

    run_pipeline(CONFIG_PATH, PROJECT_ROOT)
