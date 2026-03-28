## 5.3 Detection Engine Reproducibility and CI/CD Integration

This section documents the hardware environment, model versioning,
end-to-end pipeline timing, and integration test results for the
Phase 2 Detection Engine.

### 5.3.1 Hardware Specification

| Metric | Value |
|--------|-------|
| Device | GPU: /physical_device:GPU:0 |
| TensorFlow | 2.21.0 |
| CUDA | 12.5.1 |
| cuDNN | 9 |
| Python | 3.10.20 |
| Platform | Linux-6.17.0-19-generic-x86_64-with-glibc2.39 |

Training executed on **GPU: /physical_device:GPU:0**.
TensorFlow version: 2.21.0, CUDA: 12.5.1, cuDNN: 9.

### 5.3.2 Model Versioning

| Property | Value |
|----------|-------|
| `detection_model.weights.h5` git commit | `7d444ad562ccb791a463411a24afe3b4ab209ae9` |
| Architecture | CNN-BiLSTM-Attention (473,536 parameters) |
| Output dimensionality | 128 |
| Config file | `config/phase2_config.yaml` (version-controlled) |

Hyperparameters frozen in `config/phase2_config.yaml` — version controlled.
Model weights tagged with git commit `7d444ad562cc` in `detection_metadata.json`.

### 5.3.3 End-to-End Pipeline Timing

| Phase | Duration | Hardware |
|-------|----------|----------|
| Phase 0 | Data analysis | CPU |
| Phase 1 | Preprocessing | CPU |
| Phase 2 | **2.89 s** | GPU: /physical_device:GPU:0 |

### 5.3.4 Reproducibility Statement

Detection model reproducible via:

```bash
docker run analyst/phase0-phase2:3.0 src.phase2_detection_engine.security_hardened_phase2
```

| Parameter | Value |
|-----------|-------|
| `random_state` | 42 |
| `tf.random.set_seed()` | 42 |
| `numpy.random.seed()` | 42 |
| `TF_DETERMINISTIC_OPS` | 1 |
| Expected train context shape | (19961, 128) |
| Expected test context shape | (4877, 128) |
| Expected `attention_output` dim | 128 |
| Timesteps | 20 |
| Stride | 1 |
| CNN filters | 64, 128 |
| BiLSTM units | 128, 64 |
| Attention units | 128 |
| Dropout rate | 0.3 |

All stochastic operations use `random_state=42`,
`tf.random.set_seed(42)`, and
`numpy.random.seed(42)`.
Configuration is externalised in `config/phase2_config.yaml` (version-controlled).
The pipeline reads Phase 1 artifacts — it never recomputes preprocessing steps.

### 5.3.5 Integration Test Results

| Test | Status | Details |
|------|--------|---------|
| No classification head | PASS | last_layer=BahdanauAttention, has_head=False |
| Output shape (train_context) | PASS | (19961, 128) |
| Output shape (test_context) | PASS | (4877, 128) |
| No NaN/Inf in train_context | PASS | 0 NaN, 0 Inf |
| No NaN/Inf in test_context | PASS | 0 NaN, 0 Inf |
| Attention weights sum to 1.0 per sample | PASS | mean=1.000000, min=1.000000, max=1.000000 |

**Overall:** ALL PASSED

---

**Reviewer signature:** ____________________  **Date:** ____/____/______
