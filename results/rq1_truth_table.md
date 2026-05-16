# RQ1.4 — Tier × Patchable × Maintenance Truth Table

Derived from `src.risk_scorer.score_alert()` (live function call).

| Risk Tier | Patchable | Maintenance Active | should_surface (code) | should_surface (doc) | Match |
|---|---|---|---|---|---|
| CRITICAL | True | True | True | True | ✓ |
| CRITICAL | True | False | True | True | ✓ |
| CRITICAL | False | True | True | True | ✓ |
| CRITICAL | False | False | True | True | ✓ |
| HIGH | True | True | True | True | ✓ |
| HIGH | True | False | True | True | ✓ |
| HIGH | False | True | True | True | ✓ |
| HIGH | False | False | True | True | ✓ |
| MEDIUM | True | True | False | False | ✓ |
| MEDIUM | True | False | False | True | ✗ DISCREPANCY |
| MEDIUM | False | True | True | False | ✗ DISCREPANCY |
| MEDIUM | False | False | True | True | ✓ |
| LOW | True | True | False | False | ✓ |
| LOW | True | False | False | False | ✓ |
| LOW | False | True | False | False | ✓ |
| LOW | False | False | False | False | ✓ |
