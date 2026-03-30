"""Detection report renderer for thesis defence / IEEE Q1.

Renders ``report_section_detection.md`` (§5.1) from the
detection report dict.  No computation — pure presentation.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def render_detection_report(report: Dict[str, Any]) -> str:
    """Render the detection thesis report section (§5.1).

    Args:
        report: Detection report dict from ``DetectionExporter.build_report``.

    Returns:
        Complete Markdown string.
    """
    lines: List[str] = []
    w = lines.append

    hp = report.get("hyperparameters", {})
    shapes = report.get("shapes", {})
    env = report.get("environment", {})

    # Compute derived values for the architecture diagram
    cnn_timesteps = hp.get("timesteps", 20)
    cnn_pool_size = hp.get("cnn_pool_size", 2)
    for _ in range(2):
        cnn_timesteps = cnn_timesteps // cnn_pool_size
    bilstm_out_dim = hp.get("bilstm_units_2", 64) * 2

    cnn_filters = [hp.get("cnn_filters_1", 64), hp.get("cnn_filters_2", 128)]
    bilstm_units = [hp.get("bilstm_units_1", 128), hp.get("bilstm_units_2", 64)]

    w("## 5.1 Detection Engine Architecture")
    w("")
    w(
        "This section describes the CNN→BiLSTM→Attention feature extraction "
        "model that transforms preprocessed IoMT network traffic into "
        "fixed-length representation vectors for downstream classification "
        "(Phase 3)."
    )
    w("")

    # ── 5.1.1 Architecture Overview ──
    w("### 5.1.1 Architecture Overview")
    w("")
    w("The detection engine implements a three-stage deep feature extractor:")
    w("")
    w(
        "1. **CNN (Convolutional Neural Network):** Extracts local spatial "
        "patterns from sliding windows of consecutive network events."
    )
    w(
        "2. **BiLSTM (Bidirectional Long Short-Term Memory):** Captures "
        "temporal dependencies in both forward and backward directions."
    )
    w(
        "3. **Bahdanau Attention:** Computes adaptive weights over timesteps "
        "to produce a fixed-length context vector (weighted sum)."
    )
    w("")
    w("```")
    w(f"Input: (batch, {hp.get('timesteps', 20)}, {report.get('n_features', 24)})")
    w("  │")
    w("  ├─── [CNN Block]")
    w(
        f"  │      Conv1D({cnn_filters[0]}, k={hp.get('cnn_kernel_size', 3)}, "
        f"{hp.get('cnn_activation', 'relu')}) → MaxPool({cnn_pool_size})"
    )
    w(
        f"  │      Conv1D({cnn_filters[1]}, k={hp.get('cnn_kernel_size', 3)}, "
        f"{hp.get('cnn_activation', 'relu')}) → MaxPool({cnn_pool_size})"
    )
    w(f"  │      Output: (batch, {cnn_timesteps}, {cnn_filters[1]})")
    w("  │")
    w("  ├─── [BiLSTM Block]")
    w(
        f"  │      Bidirectional LSTM({bilstm_units[0]}, return_seq=True) → "
        f"Dropout({hp.get('dropout_rate', 0.3)})"
    )
    w(
        f"  │      Bidirectional LSTM({bilstm_units[1]}, return_seq=True) → "
        f"Dropout({hp.get('dropout_rate', 0.3)})"
    )
    w(f"  │      Output: (batch, {cnn_timesteps}, {bilstm_out_dim})")
    w("  │")
    w("  └─── [Attention Block]")
    w(
        f"         Dense({hp.get('attention_units', 128)}, tanh) → Dense(1, softmax) "
        f"→ Multiply → GlobalAvgPool"
    )
    w(f"         Output: (batch, {report.get('output_dim', 128)}) — context vector")
    w("```")
    w("")

    # ── 5.1.2 Layer Summary ──
    w("### 5.1.2 Layer Summary")
    w("")
    w("| Layer | Type | Output Shape | Parameters |")
    w("|-------|------|-------------|------------|")
    for layer in report.get("layers", []):
        params = layer.get("params", 0)
        out_shape = layer.get("output_shape", "—")
        w(f"| {layer['name']} | {layer['type']} | {out_shape} | {params:,} |")
    w("")
    w(f"**Total parameters:** {report.get('total_parameters', 0):,}")
    w(f"**Trainable parameters:** {report.get('trainable_parameters', 0):,}")
    w("")

    # ── 5.1.3 Sliding Window Reshape ──
    w("### 5.1.3 Sliding Window Reshape")
    w("")
    w(
        f"Network traffic samples are grouped into temporal windows of "
        f"**{hp.get('timesteps', 20)}** consecutive events with stride "
        f"**{hp.get('stride', 1)}**, creating a 3-D tensor suitable for "
        f"1-D convolution. The label for each window is the label of the "
        f"last sample in the window."
    )
    w("")
    w("| Parameter | Value |")
    w("|-----------|-------|")
    w(f"| Window length (timesteps) | {hp.get('timesteps', 20)} |")
    w(f"| Stride | {hp.get('stride', 1)} |")
    w(f"| Input features | {report.get('n_features', 24)} |")
    w(f"| Train windows | {shapes.get('train_windows', '—')} |")
    w(f"| Test windows | {shapes.get('test_windows', '—')} |")
    w(f"| Train context | {shapes.get('train_context', '—')} |")
    w(f"| Test context | {shapes.get('test_context', '—')} |")
    w("| Window label strategy | Last sample in window |")
    w("")

    # ── 5.1.4 Attention Mechanism ──
    w("### 5.1.4 Attention Mechanism")
    w("")
    w(
        "The Bahdanau (additive) attention mechanism computes a context "
        "vector by learning to weight each timestep according to its "
        "relevance for intrusion detection:"
    )
    w("")
    w(
        f"1. **Score:** `Dense({hp.get('attention_units', 128)}, tanh)` projects "
        f"each timestep to a score vector"
    )
    w(
        "2. **Weight:** `Dense(1)` + `softmax(axis=timesteps)` normalises "
        "scores to a probability distribution over timesteps"
    )
    w(
        "3. **Multiply:** Element-wise multiplication weights the BiLSTM "
        "output sequence by learned attention weights"
    )
    w(
        f"4. **Pool:** `GlobalAveragePooling1D` produces the final "
        f"{report.get('output_dim', 128)}-dimensional context vector"
    )
    w("")
    w(f"Output dimension: **{report.get('output_dim', 128)}** (one vector per window).")
    w("")

    # ── 5.1.5 Hyperparameters ──
    w("### 5.1.5 Hyperparameters")
    w("")
    w("| Hyperparameter | Value | Rationale |")
    w("|----------------|-------|-----------|")
    w(f"| Timesteps | {hp.get('timesteps', 20)} | " f"Captures short-term traffic patterns |")
    w(f"| Stride | {hp.get('stride', 1)} | " f"Maximum window overlap for data coverage |")
    w(
        f"| CNN filters | [{cnn_filters[0]}, {cnn_filters[1]}] | "
        f"Hierarchical spatial feature extraction |"
    )
    w(f"| CNN kernel size | {hp.get('cnn_kernel_size', 3)} | " f"3-sample receptive field |")
    w(f"| CNN activation | {hp.get('cnn_activation', 'relu')} | " f"Standard non-linearity |")
    w(
        f"| BiLSTM units | [{bilstm_units[0]}, {bilstm_units[1]}] | "
        f"Forward + backward temporal context |"
    )
    w(f"| Dropout rate | {hp.get('dropout_rate', 0.3)} | " f"Regularisation against overfitting |")
    w(f"| Attention units | {hp.get('attention_units', 128)} | " f"Score vector dimensionality |")
    w(f"| Random state | {hp.get('random_state', 42)} | " f"Reproducibility |")
    w("")

    # ── 5.1.6 Output Artifacts ──
    w("### 5.1.6 Output Artifacts")
    w("")
    w("| Artifact | Description |")
    w("|----------|-------------|")
    w(
        f"| `detection_model.weights.h5` | Model weights "
        f"({report.get('total_parameters', 0):,} parameters, "
        f"no classification head) |"
    )
    w(
        f"| `attention_output.parquet` | Weighted sum vectors "
        f"({report.get('output_dim', 128)}-dim) + labels + split indicator |"
    )
    w("| `detection_report.json` | Model summary, layer shapes, " "hyperparameters, environment |")
    w("")

    # ── 5.1.7 Execution Summary ──
    w("### 5.1.7 Execution Summary")
    w("")
    w("| Metric | Value |")
    w("|--------|-------|")
    w(f"| Total execution time | " f"**{report.get('elapsed_seconds', 0):.2f} s** |")
    w(f"| Python | {env.get('python', '—')} |")
    w(f"| TensorFlow | {env.get('tensorflow', '—')} |")
    w(f"| NumPy | {env.get('numpy', '—')} |")
    w(f"| pandas | {env.get('pandas', '—')} |")
    w(f"| Platform | {env.get('platform', '—')} |")
    w("")

    content = "\n".join(lines)
    logger.info("Detection report rendered: %d lines", len(lines))
    return content
