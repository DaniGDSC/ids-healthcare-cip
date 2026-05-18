# RQ1 / RQ3 — Tier × Patchable × Maintenance Truth Table

Derived by invoking ``src.risk_scorer.score_alert`` with synthetic ``AlertContext`` dicts (one per cell).  The ``should_surface`` column reflects the real decision function — no mocking.

Safety floor (RQ3 invariant): CRITICAL + unpatchable always surfaces, even during a maintenance window.

| risk_tier | patchable | maintenance | device_class | anomaly | adjusted | threshold | mult | should_surface | reason |
|---|---|---|---|---|---|---|---|---|---|
| CRITICAL | True | True | ehr_workstation | 0.85 | 0.425 | 0.5 | 0.5 | False | suppressed_maintenance |
| CRITICAL | True | False | ehr_workstation | 0.85 | 1.0 | 0.475 | 1.3 | True | above_threshold |
| CRITICAL | False | True | infusion_pump | 0.85 | 0.425 | 0.5 | 0.5 | True | safety_floor |
| CRITICAL | False | False | infusion_pump | 0.85 | 1.0 | 0.35 | 1.5 | True | safety_floor |
| HIGH | True | True | ehr_workstation | 0.65 | 0.325 | 0.5 | 0.5 | False | suppressed_maintenance |
| HIGH | True | False | ehr_workstation | 0.65 | 0.715 | 0.475 | 1.1 | True | above_threshold |
| HIGH | False | True | infusion_pump | 0.65 | 0.325 | 0.5 | 0.5 | False | suppressed_maintenance |
| HIGH | False | False | infusion_pump | 0.65 | 0.78 | 0.35 | 1.2 | True | above_threshold |
| MEDIUM | True | True | ehr_workstation | 0.45 | 0.225 | 0.5 | 0.5 | False | suppressed_maintenance |
| MEDIUM | True | False | ehr_workstation | 0.45 | 0.45 | 0.475 | 1.0 | False | below_threshold |
| MEDIUM | False | True | infusion_pump | 0.45 | 0.225 | 0.5 | 0.5 | False | suppressed_maintenance |
| MEDIUM | False | False | infusion_pump | 0.45 | 0.4725 | 0.35 | 1.05 | True | above_threshold |
| LOW | True | True | ehr_workstation | 0.2 | 0.1 | 0.5 | 0.5 | False | suppressed_maintenance |
| LOW | True | False | ehr_workstation | 0.2 | 0.2 | 0.475 | 1.0 | False | below_threshold |
| LOW | False | True | infusion_pump | 0.2 | 0.1 | 0.5 | 0.5 | False | suppressed_maintenance |
| LOW | False | False | infusion_pump | 0.2 | 0.2 | 0.35 | 1.0 | False | below_threshold |
