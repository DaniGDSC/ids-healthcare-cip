## 6.1 Classification Engine Results

This section documents the Phase 3 Classification Engine: architecture,
progressive unfreezing strategy, training history, and evaluation metrics.

### 6.1.1 Classification Architecture

```
Phase 1 parquets (19980×29, 4896×29)
  ↓ reshape (timesteps=20, stride=1)
Windows (19961×20×29, 4877×20×29)
  ↓ CNN → BiLSTM → Attention (474,496 params, frozen/unfrozen)
Context vectors (batch, 128)
  ↓ Dense(64, relu)
  ↓ Dropout(0.3)
  ↓ Dense(1, sigmoid)
Predictions (batch, 1)
```

| Component | Parameters |
|-----------|-----------|
| Detection backbone (CNN→BiLSTM→Attention) | 474,496 |
| Classification head (Dense→Dropout→Dense) | 8,321 |
| **Total** | **482,817** |

### 6.1.2 Progressive Unfreezing Strategy

| Phase | Epochs | Learning Rate | Frozen Groups | Trainable |
|-------|--------|---------------|---------------|-----------|
| Phase A — Head only | 5 | 0.001 | cnn, bilstm1, bilstm2, attention | — + head |
| Phase B — Attention + Head | 5 | 0.0001 | cnn, bilstm1, bilstm2 | attention + head |
| Phase C — BiLSTM-2 + Attention + Head | 5 | 1e-05 | cnn, bilstm1 | bilstm2, attention + head |

### 6.1.3 Training History

| Phase | Epochs Run | Train Loss | Train Acc | Val Loss | Val Acc |
|-------|-----------|------------|-----------|----------|---------|
| Phase A — Head only | 5 | 0.3914 | 0.8554 | 0.1931 | 0.9785 |
| Phase B — Attention + Head | 5 | 0.3723 | 0.8610 | 0.1715 | 0.9827 |
| Phase C — BiLSTM-2 + Attention + Head | 5 | 0.3525 | 0.8703 | 0.1358 | 0.9850 |

### 6.1.4 Evaluation Metrics

| Metric | Value |
|--------|-------|
| Accuracy | 0.8345 |
| F1-score (weighted) | 0.8133 |
| Precision (weighted) | 0.7971 |
| Recall (weighted) | 0.8345 |
| AUC-ROC | 0.6144 |
| Test samples | 4,877 |
| Threshold | 0.5 |

### 6.1.5 Confusion Matrix

| | Predicted Normal | Predicted Attack |
|---|---|---|
| **Actual Normal** | TN=3998 | FP=267 |
| **Actual Attack** | FN=540 | TP=72 |

### 6.1.6 Output Artifacts

| Artifact | Path |
|----------|------|
| Model weights | `data/phase3/classification_model.weights.h5` |
| Metrics report | `data/phase3/metrics_report.json` |
| Confusion matrix | `data/phase3/confusion_matrix.csv` |
| Training history | `data/phase3/training_history.json` |

### 6.1.7 Execution Summary

| Property | Value |
|----------|-------|
| Device | CPU: x86_64 |
| TensorFlow | 2.20.0 |
| CUDA | N/A (CPU execution) |
| Python | 3.12.3 |
| Platform | Linux-6.17.0-14-generic-x86_64-with-glibc2.39 |
| Duration | 42.25s |
| Git commit | `d3de1e30139d` |
| Config file | `config/phase3_config.yaml` (version-controlled) |
| Random state | 42 |

---

**Reviewer signature:** ____________________  **Date:** ____/____/______
