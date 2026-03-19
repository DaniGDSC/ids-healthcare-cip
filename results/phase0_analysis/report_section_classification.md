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

### 6.1.2 Auto Classification Head

| n_classes | Activation | Loss | Output units |
|-----------|------------|------|--------------|
| 2 | sigmoid | binary_crossentropy | 1 |
| >2 | softmax | categorical_crossentropy | n |

### 6.1.3 Progressive Unfreezing Strategy

Progressive unfreezing chosen to prevent catastrophic forgetting
of Phase 2 feature extraction weights while adapting to
classification task.

| Phase | Epochs | Learning Rate | Frozen Groups | Trainable |
|-------|--------|---------------|---------------|-----------|
| Phase A — Head only | 5 | 0.001 | cnn, bilstm1, bilstm2, attention | — + head |
| Phase B — Attention + Head | 5 | 0.0001 | cnn, bilstm1, bilstm2 | attention + head |
| Phase C — BiLSTM-2 + Attention + Head | 5 | 1e-05 | cnn, bilstm1 | bilstm2, attention + head |

### 6.1.4 Training History

| Phase | Epochs Run | Train Loss | Train Acc | Val Loss | Val Acc |
|-------|-----------|------------|-----------|----------|---------|
| Phase A — Head only | 5 | 0.3919 | 0.8572 | 0.1994 | 0.9787 |
| Phase B — Attention + Head | 5 | 0.3799 | 0.8591 | 0.1769 | 0.9835 |
| Phase C — BiLSTM-2 + Attention + Head | 5 | 0.3574 | 0.8688 | 0.1376 | 0.9865 |

### 6.1.5 Evaluation Metrics

| Metric | Value |
|--------|-------|
| Accuracy | 0.8321 |
| F1-score (weighted) | 0.8128 |
| Precision (weighted) | 0.7977 |
| Recall (weighted) | 0.8321 |
| AUC-ROC | 0.6119 |
| Test samples | 4,877 |
| Threshold | 0.5 |

### 6.1.6 Confusion Matrix

| | Predicted Normal | Predicted Attack |
|---|---|---|
| **Actual Normal** | TN=3981 | FP=284 |
| **Actual Attack** | FN=535 | TP=77 |

### 6.1.7 Output Artifacts

| Artifact | Path |
|----------|------|
| Model weights | `data/phase3/classification_model.weights.h5` |
| Metrics report | `data/phase3/metrics_report.json` |
| Confusion matrix | `data/phase3/confusion_matrix.csv` |
| Training history | `data/phase3/training_history.json` |

### 6.1.8 Execution Summary

| Property | Value |
|----------|-------|
| Device | CPU: x86_64 |
| TensorFlow | 2.20.0 |
| CUDA | N/A (CPU execution) |
| Python | 3.12.3 |
| Platform | Linux-6.17.0-14-generic-x86_64-with-glibc2.39 |
| Duration | 43.87s |
| Git commit | `3259f3eee118` |
| Config file | `config/phase3_config.yaml` (version-controlled) |
| Random state | 42 |

---

**Reviewer signature:** ____________________  **Date:** ____/____/______
