# Archive

Historical / superseded artefacts kept for provenance. Nothing here is
loaded by any active code path. Files are stored as-is to preserve the
original intent without polluting the active config / module namespace.

## Inventory

### `module2_phase2_5_config.yaml`

**Provenance:** former `module2_detection/phase2_5_config.yaml`.

**What it described:** the Phase 2.5 BiLSTM + Bahdanau-attention
hyperparameter-tuning + ablation suite from research_spec v1
(Bayesian TPE search via Optuna, 8 ablation variants — no_attention,
no_bilstm2, no_cnn2, unidirectional_lstm, timesteps overrides,
dropout overrides).

**Why archived:** the v2.0 Module 2 implementation uses sklearn
`GradientBoostingClassifier` + `RandomForestClassifier` +
`DecisionTreeClassifier` for Track A and a Keras dense DAE for Track B.
There is no BiLSTM, no attention module, no ablation runner — the
config was dead and operators reading it would be misled about
Module 2's actual scope.

**Restoration:** if a future revision adds the BiLSTM ablation suite,
move this back to `module2_detection/phase2_5_config.yaml` and wire a
matching tuning runner. Until then, treat as historical record only.
