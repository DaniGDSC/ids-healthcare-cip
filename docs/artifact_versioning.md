# Artifact schema versioning — Sprint 6 / Tầng 3.5

## Why

The upgrade work surfaced a class of bugs (Category 1 in the post-Sprint-4
remediation analysis) where:

  - a producer's output format changes (npz keys renamed, JSON field
    added),
  - downstream consumers keep their old assumption silently,
  - errors surface much later as KeyError / IndexError / wrong-number
    deep inside the pipeline.

Sprint 6 closes this by stamping every artifact with a
``_schema_version`` and gating builds against a central registry.

## How it works

The registry lives in ``common.artifact_versioning``::

    ARTIFACT_VERSIONS = {
        "risk_scores.npz":         "2.0",
        "alert_responses.json":    "3.2",
        "phase0_baseline.json":    "2.1",
        "faithfulness_gate.json":  "1.1",
        ...
    }

For each artifact:

  - **Writers** call ``version_kwarg_for(name)`` (npz) or
    ``embed_version_in_dict(payload, name)`` (json) so the version is
    embedded automatically.
  - **Readers** call ``check_compatibility(path)`` or
    ``assert_compatible(path)`` to verify the on-disk version matches
    the registry. Mismatch → ``ArtifactVersionMismatch``.
  - **CI** runs ``python -m tools.version_gate --check`` to walk
    every registered artifact and fail the build on any drift.

## Where versions live in the file

  - **npz**: a 0-D string array under the key ``schema_version``.
  - **JSON dict**: top-level key ``_schema_version`` for flat dicts;
    nested under ``_provenance._schema_version`` for envelope-shaped
    artifacts (Module 5's ``AlertResponsesEnvelope``).
  - **JSON list**: not supported yet — these artifacts
    (``analyst_report.json``, ``clinician_summaries.json``) are
    listed in ``PENDING_ENVELOPE_MIGRATION`` so the gate skips them
    until they migrate to envelope shape.

## How to bump a version

### MINOR bump (backwards-compatible)

1. Update the producer code to write the new field.
2. Bump the version in ``ARTIFACT_VERSIONS`` (e.g. ``"3.2"`` →
   ``"3.3"``).
3. Re-run the producer once to refresh the artifact.
4. The version gate now passes on the new shape.

No migration function needed — consumers loading older artifacts
ignore the new field via pydantic ``Optional`` / default values.

### MAJOR bump (incompatible)

1. Update the producer code.
2. Bump the version in ``ARTIFACT_VERSIONS`` (e.g. ``"3.2"`` →
   ``"4.0"``).
3. Register a migration in
   ``common.artifact_versioning_migrations.MIGRATIONS``::

       def _alert_responses_3_2_to_4_0(payload: dict) -> dict:
           # transform the payload in-place
           ...
           return payload

       MIGRATIONS[("alert_responses.json", "3.2", "4.0")] = (
           _alert_responses_3_2_to_4_0
       )

4. Re-run the producer once.
5. Optionally write a one-time migration script under ``tools/``
   to bring already-on-disk older artifacts up to the new shape.

## What's currently versioned

Registered (the version gate checks these):

| Artifact | Current | Producer |
|---|---|---|
| ``risk_scores.npz`` | 2.0 | ``module3_risk_scoring.io.save_outputs`` |
| ``alert_responses.json`` | 3.2 | ``tools.phase1_regen_module5`` |
| ``phase0_baseline.json`` | 2.1 | ``tools.phase0_baseline`` |
| ``faithfulness_gate.json`` | 1.1 | ``tools.faithfulness_gate`` |
| ``coverage_audit.json`` | 1.0 | ``tools.coverage_audit`` |
| ``shap_values_xgboost.npz`` | 1.0 | ``module4_explanations.io.save_shap_values`` |
| ``formula_comparison.json`` | 1.0 | ``tools.formula_comparison`` |
| ``v1_v2_comparison.json`` | 1.0 | ``tools.compare_v1_v2`` |
| ``stability_variant_comparison.json`` | 1.0 | ``tools.compare_stability_variants`` |

Pending envelope migration (skipped by gate until rebuilt with
envelope shape):

| Artifact | Reason |
|---|---|
| ``analyst_report.json`` | Bare list — needs envelope wrapper |
| ``clinician_summaries.json`` | Bare list — needs envelope wrapper |

Opt-out (not actively maintained by Sprint-N regen tools — module 4
full-mode run embeds version automatically when it next runs):

| Artifact |
|---|
| ``shap_values_random_forest.npz`` |
| ``shap_values_decision_tree.npz`` |
| ``dae_feature_errors.npz`` |

## CI integration

```bash
# Walk every registered artifact, fail on any mismatch
python -m tools.version_gate --check
```

Returns 0 on PASS, 1 on FAIL. Add to the pre-merge CI job alongside
the existing ``phase0_baseline --check`` and ``faithfulness_gate
--check`` calls.

## Rollback

The version gate is purely advisory — no artifact is rewritten by it.
To roll back to a previous schema:

1. Restore the producer code (git checkout).
2. Restore the entry in ``ARTIFACT_VERSIONS``.
3. Re-run the producer.

There's no "v1-only" point of no return. The gate is the *detector*;
the producer code is the *truth*.

## Sprint 6 deliverables

  - ``common/artifact_versioning.py`` — registry + helpers
  - ``common/artifact_versioning_migrations.py`` — migration registry
    (empty, scaffold ready)
  - ``tools/version_gate.py`` — gate runner with ``--check``
  - 11 writers updated to embed version (npz savers + JSON dumpers)
  - ``tests/test_artifact_versioning.py`` — 25 tests pinning the
    registry semantics + end-to-end gate behaviour
  - ``docs/artifact_versioning.md`` — this document
