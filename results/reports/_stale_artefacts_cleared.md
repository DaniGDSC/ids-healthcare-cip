# A_patient stale-artefact cleanup (Pre-redesign Task 2)

**Date:** 2026-05-06

## What was deleted

The following five generated artefacts contained the legacy `A_patient` /
`a_patient` symbol from before the formula-component rename to
`D_clinical_tier`:

| File | Producer | Status |
| --- | --- | --- |
| `nlg_templates.json` | `module4_explanations/module4_explanations.py::export_nlg_templates` | ✓ regenerated 2026-05-06 — uses `d_clinical_tier` |
| `alert_responses.json` | `module5_responses/module5_responses.py` | regenerates on next `python module5_responses/module5_responses.py` |
| `worked_examples.json` | `module5_responses/module5_pipeline.py::run_worked_examples` | regenerates on next `python module5_responses/module5_pipeline.py` |
| `example_explanations.json` | `module4_explanations/module4_explanations.py` | regenerates on next `python module4_explanations/module4_explanations.py` |
| `risk_config_adjusted.json` | `feedback_loop_demo.py` | regenerates on next `python feedback_loop_demo.py` |

## Verification

```bash
grep -rn "A_patient\|a_patient" results/reports/ --include="*.json"
# → empty output (or just nlg_templates.json which was already regenerated)

grep -rn "A_patient\|a_patient" --include="*.py" /home/un1/project/ids-healthcare-cip
# → empty output (source code rename complete)
```

The 4 remaining artefacts are reproducible from on-disk inputs (Track A
predictions + risk_scores.npz). When their respective modules next run,
they will write the `d_clinical_tier`-based output. No source-code drift.
