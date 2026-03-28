## 5.1 Detection Engine Architecture

This section describes the CNN→BiLSTM→Attention feature extraction model that transforms preprocessed IoMT network traffic into fixed-length representation vectors for downstream classification (Phase 3).

### 5.1.1 Architecture Overview

The detection engine implements a three-stage deep feature extractor:

1. **CNN (Convolutional Neural Network):** Extracts local spatial patterns from sliding windows of consecutive network events.
2. **BiLSTM (Bidirectional Long Short-Term Memory):** Captures temporal dependencies in both forward and backward directions.
3. **Bahdanau Attention:** Computes adaptive weights over timesteps to produce a fixed-length context vector (weighted sum).

```
Input: (batch, 20, 24)
  │
  ├─── [CNN Block]
  │      Conv1D(64, k=3, relu) → MaxPool(2)
  │      Conv1D(128, k=3, relu) → MaxPool(2)
  │      Output: (batch, 5, 128)
  │
  ├─── [BiLSTM Block]
  │      Bidirectional LSTM(128, return_seq=True) → Dropout(0.3)
  │      Bidirectional LSTM(64, return_seq=True) → Dropout(0.3)
  │      Output: (batch, 5, 128)
  │
  └─── [Attention Block]
         Dense(128, tanh) → Dense(1, softmax) → Multiply → GlobalAvgPool
         Output: (batch, 128) — context vector
```

### 5.1.2 Layer Summary

| Layer | Type | Output Shape | Parameters |
|-------|------|-------------|------------|
| conv1 | Conv1D | (None, 20, 64) | 4,672 |
| pool1 | MaxPooling1D | (None, 10, 64) | 0 |
| conv2 | Conv1D | (None, 10, 128) | 24,704 |
| pool2 | MaxPooling1D | (None, 5, 128) | 0 |
| bilstm1 | Bidirectional | (None, 5, 256) | 263,168 |
| drop1 | Dropout | (None, 5, 256) | 0 |
| bilstm2 | Bidirectional | (None, 5, 128) | 164,352 |
| drop2 | Dropout | (None, 5, 128) | 0 |
| attention | BahdanauAttention | (None, 128) | 16,640 |

**Total parameters:** 473,536
**Trainable parameters:** 473,536

### 5.1.3 Sliding Window Reshape

Network traffic samples are grouped into temporal windows of **20** consecutive events with stride **1**, creating a 3-D tensor suitable for 1-D convolution. The label for each window is the label of the last sample in the window.

| Parameter | Value |
|-----------|-------|
| Window length (timesteps) | 20 |
| Stride | 1 |
| Input features | 24 |
| Train windows | (19961, 20, 24) |
| Test windows | (4877, 20, 24) |
| Train context | (19961, 128) |
| Test context | (4877, 128) |
| Window label strategy | Last sample in window |

### 5.1.4 Attention Mechanism

The Bahdanau (additive) attention mechanism computes a context vector by learning to weight each timestep according to its relevance for intrusion detection:

1. **Score:** `Dense(128, tanh)` projects each timestep to a score vector
2. **Weight:** `Dense(1)` + `softmax(axis=timesteps)` normalises scores to a probability distribution over timesteps
3. **Multiply:** Element-wise multiplication weights the BiLSTM output sequence by learned attention weights
4. **Pool:** `GlobalAveragePooling1D` produces the final 128-dimensional context vector

Output dimension: **128** (one vector per window).

### 5.1.5 Hyperparameters

| Hyperparameter | Value | Rationale |
|----------------|-------|-----------|
| Timesteps | 20 | Captures short-term traffic patterns |
| Stride | 1 | Maximum window overlap for data coverage |
| CNN filters | [64, 128] | Hierarchical spatial feature extraction |
| CNN kernel size | 3 | 3-sample receptive field |
| CNN activation | relu | Standard non-linearity |
| BiLSTM units | [128, 64] | Forward + backward temporal context |
| Dropout rate | 0.3 | Regularisation against overfitting |
| Attention units | 128 | Score vector dimensionality |
| Random state | 42 | Reproducibility |

### 5.1.6 Output Artifacts

| Artifact | Description |
|----------|-------------|
| `detection_model.weights.h5` | Model weights (473,536 parameters, no classification head) |
| `attention_output.parquet` | Weighted sum vectors (128-dim) + labels + split indicator |
| `detection_report.json` | Model summary, layer shapes, hyperparameters, environment |

### 5.1.7 Execution Summary

| Metric | Value |
|--------|-------|
| Total execution time | **2.88 s** |
| Python | 3.10.20 |
| TensorFlow | 2.21.0 |
| NumPy | 2.2.6 |
| pandas | 2.3.3 |
| Platform | Linux-6.17.0-19-generic-x86_64-with-glibc2.39 |
