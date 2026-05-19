# RQ2 Merge & Figures Pipeline — Canonical Aggregator + Paper Figures

**Project:** XAI-IDS-Healthcare
**Scope:** Phases 18–20 of the RQ2 pipeline: the file rename to fix the historical naming clash, the canonical `compute_rq2_metrics.py` aggregator that pulls all Track 1–5 outputs into `results/rq2_metrics.json`, the figure generator producing four paper PDFs, and the CI gate that verifies all RQ2 targets are met.
**Purpose:** Single, self-contained spec for closing the RQ2 pipeline loop. Hand to Claude Code.
**Status of design:** All decisions locked. Three `DO NOT GUESS` checkpoints (existing file content, import graph, figure aesthetics).

---

## 0. How to use this spec

1. Phase 0 is **mandatory** — Claude Code must inspect the existing `compute_rq2_metrics.py` and identify every place in the codebase that imports it BEFORE renaming.
2. Phases 1–4 are sequential after Phase 0.
3. Markers:
   - **DO NOT GUESS** — stop and ask the developer
   - **DEFENSE-CRITICAL** — directly defends the senior engineer review's naming-clash concern
   - **TARGET** — a numeric goal from `RQ2_expected_outputs.md`
4. Total expected output: 1 file rename, 1 new canonical aggregator, 1 figure script, 4 PDFs, 1 acceptance test addition. Runtime: aggregator sub-second, figures ~10s.

---

## 1. Background: what this completes

| Phase | Deliverable | Status before this spec | Status after |
|---|---|---|---|
| **Phase 18** | `compute_rq2_metrics.py` (canonical MVE aggregator) | naming-clash file produces detection metrics | renamed; new file produces MVE metrics |
| **Phase 18** | `compute_detection_metrics.py` | did not exist | created (= old `compute_rq2_metrics.py` content) |
| **Phase 18** | `results/rq2_metrics.json` (canonical) | contains detection metrics confusingly | contains MVE aggregation |
| **Phase 19** | `make_rq2_figures.py` | did not exist | created with 4-PDF CLI |
| **Phase 19** | 4 PDFs in `results/figures/` | did not exist | created |
| **Phase 20** | `tests/acceptance_tests.py::test_rq2_targets_met` | did not exist | created |

After this spec, RQ2 has a single canonical metrics file, paper-ready figures, and a CI gate.

---

## 2. Locked design decisions

| Decision | Resolution |
|---|---|
| Naming clash | Rename existing `module6_evaluation/compute_rq2_metrics.py` → `compute_detection_metrics.py`; create new `compute_rq2_metrics.py` for MVE aggregation |
| Figure scope | 4 PDFs: SHAP stability histogram, Mode A vs B alignment, MITRE grounding per category, per-role user study results |
| Failure catalog figure | Included as a 5th figure (failure category counts) for §5.3 |
| Figure script | Single `make_rq2_figures.py` with CLI flag `--only <figure_id>` to regenerate one figure at a time |
| Output location | All PDFs in `results/figures/rq2_*.pdf` (consistent prefix) |
| Aggregator pattern | Mirror of `compute_rq1_metrics.py`: pure JSON-only aggregator, loads sub-files, no model inference, no figure generation |
| Status semantics | Each sub-block carries `_status` ∈ {`complete`, `partial`, `pending`, `source_unavailable`} — single source of truth for "what's ready" |

---

## 3. Phase 0 — Pre-rename audit (DO NOT GUESS)

The existing `compute_rq2_metrics.py` produces detection metrics (FNR_critical, sensitivity, etc.). Before renaming, Claude Code must:

### 3.1 Discovery script

```python
# scripts/discover_compute_rq2_metrics_callers.py — TRANSIENT, delete after Phase 0
"""
Find every place in the codebase that imports or invokes compute_rq2_metrics.
The rename will break each of these unless updated atomically.
"""
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# 1. Existing file content (preserve before rename)
existing = REPO_ROOT / "module6_evaluation/compute_rq2_metrics.py"
if not existing.exists():
    print(f"NOT FOUND: {existing}")
    print("Verify with developer — naming clash may already be resolved.")
    sys.exit(1)

content = existing.read_text()
print(f"# Existing compute_rq2_metrics.py — {len(content)} bytes")
print(f"# First 10 lines:")
for line in content.splitlines()[:10]:
    print(f"#   {line}")

# 2. grep for callers across the repo
patterns = [
    "compute_rq2_metrics",
    "from module6_evaluation.compute_rq2_metrics",
    "from module6_evaluation import compute_rq2_metrics",
    "module6_evaluation.compute_rq2_metrics",
]
print("\n# Caller search results:")
for pat in patterns:
    result = subprocess.run(
        ["grep", "-rn", "--include=*.py", "--include=*.md",
         "--include=*.yaml", "--include=*.sh", pat, str(REPO_ROOT)],
        capture_output=True, text=True
    )
    if result.stdout:
        print(f"\n## Matches for '{pat}':")
        for line in result.stdout.splitlines():
            # Exclude the file itself + this script
            if "compute_rq2_metrics.py" in line and "module6_evaluation" in line:
                continue
            if "discover_compute_rq2_metrics_callers" in line:
                continue
            print(f"  {line}")
```

### 3.2 What to confirm before Phase 1

1. **What does the existing file actually compute?** Confirm it produces detection metrics (FNR_critical, sensitivity, specificity, confusion matrix) as the architecture states. If it has drifted to also compute some MVE metrics, the rename strategy changes.
2. **How many callers exist?** Tests? Documentation? CI scripts? Each needs updating in the same commit as the rename.
3. **Is `results/rq2_metrics.json` currently the output of the existing file?** If yes, the rename also means `results/rq2_metrics.json` semantically changes (from detection JSON to MVE JSON). The detection JSON moves to `results/detection_metrics.json`.

### 3.3 Verification

```bash
python scripts/discover_compute_rq2_metrics_callers.py > /tmp/rename_audit.txt
# Developer reviews — confirms call graph + agrees rename scope
```

**DO NOT GUESS** any of the three confirmation items. Renaming a file that other modules depend on without finding all callers leaves stealth import errors that only fire at runtime.

---

## 4. Phase 1 — Atomic rename

This must be a **single commit** so the codebase is never in a half-renamed state.

### 4.1 Rename steps

```bash
# 1. Move the file
git mv module6_evaluation/compute_rq2_metrics.py \
       module6_evaluation/compute_detection_metrics.py

# 2. Update the docstring/header inside the renamed file
# Add at top:
"""
compute_detection_metrics.py
(Formerly compute_rq2_metrics.py — renamed 2026-MM-DD to resolve the naming
clash flagged in senior_engineer_review.md. This file computes RQ1 detection
metrics; the new compute_rq2_metrics.py aggregates MVE metrics for RQ2.)
"""

# 3. Update the output path inside the file:
#    BEFORE: OUT = REPO_ROOT / "results/rq2_metrics.json"
#    AFTER:  OUT = REPO_ROOT / "results/detection_metrics.json"

# 4. Update every caller found in Phase 0
#    e.g., tests/acceptance_tests.py, ARCHITECTURE.md, etc.

# 5. Update ARCHITECTURE.md line 39 reference
#    BEFORE: "- compute_rq2_metrics.py — reads evaluation_alerts.json, outputs results/rq2_metrics.json"
#    AFTER:  "- compute_detection_metrics.py — reads evaluation_alerts.json, outputs results/detection_metrics.json
#             (formerly compute_rq2_metrics.py; renamed to resolve RQ2 naming clash)"
```

### 4.2 Verification

```bash
# Atomic state check after the commit:
pytest tests/                                        # All previously-passing tests still pass
python -m module6_evaluation.compute_detection_metrics   # Should run, write detection_metrics.json
ls -l results/detection_metrics.json                # Exists
test ! -e results/rq2_metrics.json                  # Should NOT exist yet (Phase 2 creates it)
```

If any test fails on this step, the rename missed a caller. Investigate Phase 0's audit output.

---

## 5. Phase 2 — Canonical `compute_rq2_metrics.py` (DEFENSE-CRITICAL)

### 5.1 Create `module6_evaluation/compute_rq2_metrics.py`

**Contract:**
- **Inputs:** every sub-file from Tracks 1–5 (any subset; missing inputs produce `_status: pending`).
- **Output:** `results/rq2_metrics.json`.
- **Runtime:** sub-second.
- **Side effects:** writes one file. No model inference, no slow computation.

### 5.2 Output schema

```json
{
  "_meta": {
    "schema_version": "1.0",
    "generated_at": "<ISO-8601>",
    "generated_by": "module6_evaluation/compute_rq2_metrics.py",
    "research_question": "RQ2 — Can MVE provide role-tailored security explanations enabling non-specialist hospital stakeholders to make informed threat triage decisions?",
    "active_subquestions": ["RQ2.a", "RQ2.b", "RQ2.c", "RQ2.e"],
    "rescoped_subquestions": ["RQ2.d (moved to thesis §7.2.3)"],
    "tracks_present": ["1", "2", "3"],
    "tracks_pending": ["4", "5"]
  },
  "headline": {
    "_description": "Highest-level pass/fail per sub-RQ. Read this first.",
    "rq2_a_compliance": "complete",
    "rq2_b_faithfulness": "complete",
    "rq2_c_user_study": "pending",
    "rq2_e_mitre_grounding": "complete",
    "rq2_d_failure_catalog": "complete",
    "_overall_status": "partial — user study pending"
  },
  "faithfulness": {
    "_status": "complete",
    "_merged_at": "<ISO-8601>",
    "_subfile_paths": [
      "results/rq2_shap_stability.json",
      "results/rq2_mve_shap_alignment.json"
    ],
    "shap_stability": { ... contents of rq2_shap_stability.json ... },
    "mve_shap_alignment": { ... contents of rq2_mve_shap_alignment.json ... }
  },
  "mitre_grounding": {
    "_status": "complete",
    "_merged_at": "<ISO-8601>",
    "_subfile_paths": [
      "results/rq2_mitre_audit.json",
      "results/rq2_mitre_grounding.json"
    ],
    "config_audit": { ... },
    "layer1_grounding": { ... }
  },
  "compliance": {
    "_status": "complete",
    "_merged_at": "<ISO-8601>",
    "_subfile_paths": [
      "results/rq2_word_budget_audit.json",
      "results/rq2_compliance_audit.json"
    ],
    "word_budget_audit": { ... },
    "compliance_manifest_audit": { ... },
    "_note": "PHI flow control and cross-role consistency are pytest-only gates; check CI status. See tests/test_phi_not_in_llm_prompt.py and tests/test_step13_cross_role_consistency.py."
  },
  "user_study": {
    "_status": "pending",
    "_merged_at": null,
    "_subfile_paths": [
      "survey/study_data_audit.json",
      "survey/rq2c_exclusions.json",
      "analysis/outputs/rq2c_per_role.json",
      "survey/qualitative_themes.yaml"
    ],
    "data_audit": null,
    "exclusions": null,
    "per_role_analysis": null,
    "qualitative_themes_path": null
  },
  "failure_catalog": {
    "_status": "complete",
    "_merged_at": "<ISO-8601>",
    "_subfile_paths": ["results/rq2_failure_mode_catalog.json"],
    "summary": { ... },
    "disclosure": { ... },
    "catalog_path": "results/rq2_failure_mode_catalog.json",
    "catalog_md_path": "results/rq2_failure_mode_catalog.md"
  },
  "targets": {
    "_description": "Boolean pass/fail per RQ2 target. Used by tests/acceptance_tests.py::test_rq2_targets_met.",
    "shap_stability_mean": {
      "value": 0.92, "target": 0.90, "pass": true
    },
    "shap_stability_pass_rate": {
      "value": 0.84, "target": 0.80, "pass": true
    },
    "alignment_all_three": {
      "value": 0.84, "target": 0.80, "pass": true
    },
    "alignment_at_least_two": {
      "value": 0.97, "target": 0.95, "pass": true
    },
    "mitre_audit": {
      "value": true, "target": true, "pass": true
    },
    "mitre_grounding_rate": {
      "value": 0.92, "target": 0.90, "pass": true
    },
    "word_budget_audit": {
      "value": true, "target": true, "pass": true
    },
    "compliance_manifest_evidence": {
      "value": true, "target": true, "pass": true
    },
    "failure_catalog_disclosure": {
      "value": "observation_not_improvement", "target": "observation_not_improvement", "pass": true
    }
  }
}
```

### 5.3 Implementation

```python
"""
compute_rq2_metrics.py
Canonical aggregator for RQ2 — pulls every Track 1-5 sub-file into one JSON.

This file replaces the old compute_rq2_metrics.py (renamed to
compute_detection_metrics.py on 2026-MM-DD) to resolve the naming clash
flagged in senior_engineer_review.md.

Inputs: any subset of Track 1-5 sub-files (missing → _status: pending).
Output: results/rq2_metrics.json
Runtime: sub-second. No model inference.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = REPO_ROOT / "results/rq2_metrics.json"


# ─── Helpers ───────────────────────────────────────────────────

def _try_load_json(rel_path: str) -> Optional[dict]:
    """Load JSON if it exists; return None otherwise."""
    p = REPO_ROOT / rel_path
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _make_block(status: str, subfile_paths: list, **contents) -> dict:
    """Build a sub-block with consistent structure."""
    out = {
        "_status": status,
        "_merged_at": _now_iso() if status != "pending" else None,
        "_subfile_paths": subfile_paths,
    }
    out.update(contents)
    return out


# ─── Sub-block loaders ─────────────────────────────────────────

def _load_faithfulness():
    stability = _try_load_json("results/rq2_shap_stability.json")
    alignment = _try_load_json("results/rq2_mve_shap_alignment.json")

    paths = [
        "results/rq2_shap_stability.json",
        "results/rq2_mve_shap_alignment.json",
    ]
    if stability and alignment:
        return _make_block("complete", paths,
                           shap_stability=stability,
                           mve_shap_alignment=alignment)
    if stability or alignment:
        return _make_block("partial", paths,
                           shap_stability=stability,
                           mve_shap_alignment=alignment)
    return _make_block("pending", paths,
                       shap_stability=None,
                       mve_shap_alignment=None)


def _load_mitre_grounding():
    audit = _try_load_json("results/rq2_mitre_audit.json")
    grounding = _try_load_json("results/rq2_mitre_grounding.json")

    paths = [
        "results/rq2_mitre_audit.json",
        "results/rq2_mitre_grounding.json",
    ]
    if audit and grounding:
        return _make_block("complete", paths,
                           config_audit=audit,
                           layer1_grounding=grounding)
    if audit or grounding:
        return _make_block("partial", paths,
                           config_audit=audit,
                           layer1_grounding=grounding)
    return _make_block("pending", paths,
                       config_audit=None,
                       layer1_grounding=None)


def _load_compliance():
    word_budget = _try_load_json("results/rq2_word_budget_audit.json")
    manifest = _try_load_json("results/rq2_compliance_audit.json")

    paths = [
        "results/rq2_word_budget_audit.json",
        "results/rq2_compliance_audit.json",
    ]
    note = (
        "PHI flow control and cross-role consistency are pytest-only gates; "
        "check CI status. See tests/test_phi_not_in_llm_prompt.py and "
        "tests/test_step13_cross_role_consistency.py."
    )
    if word_budget and manifest:
        return _make_block("complete", paths,
                           word_budget_audit=word_budget,
                           compliance_manifest_audit=manifest,
                           _note=note)
    if word_budget or manifest:
        return _make_block("partial", paths,
                           word_budget_audit=word_budget,
                           compliance_manifest_audit=manifest,
                           _note=note)
    return _make_block("pending", paths,
                       word_budget_audit=None,
                       compliance_manifest_audit=None,
                       _note=note)


def _load_user_study():
    audit = _try_load_json("survey/study_data_audit.json")
    exclusions = _try_load_json("survey/rq2c_exclusions.json")
    per_role = _try_load_json("analysis/outputs/rq2c_per_role.json")
    themes_path = REPO_ROOT / "survey/qualitative_themes.yaml"

    paths = [
        "survey/study_data_audit.json",
        "survey/rq2c_exclusions.json",
        "analysis/outputs/rq2c_per_role.json",
        "survey/qualitative_themes.yaml",
    ]

    quant_complete = bool(audit and per_role)
    themes_complete = themes_path.exists()

    if quant_complete and themes_complete:
        status = "complete"
    elif quant_complete or themes_complete:
        status = "partial"
    else:
        status = "pending"

    return _make_block(
        status, paths,
        data_audit=audit,
        exclusions=exclusions,
        per_role_analysis=per_role,
        qualitative_themes_path=(
            str(themes_path.relative_to(REPO_ROOT)) if themes_complete else None
        ),
    )


def _load_failure_catalog():
    catalog = _try_load_json("results/rq2_failure_mode_catalog.json")
    paths = ["results/rq2_failure_mode_catalog.json"]

    if catalog is None:
        return _make_block("pending", paths,
                           summary=None, disclosure=None,
                           catalog_path=None, catalog_md_path=None)

    md_path = REPO_ROOT / "results/rq2_failure_mode_catalog.md"

    # Status comes from the catalog's own _meta.sources_missing list
    cat_meta = catalog.get("_meta", {})
    if cat_meta.get("sources_missing"):
        status = "partial"
    else:
        status = "complete"

    return _make_block(
        status, paths,
        summary=catalog.get("summary"),
        disclosure=catalog.get("_disclosure"),
        catalog_path="results/rq2_failure_mode_catalog.json",
        catalog_md_path=(
            "results/rq2_failure_mode_catalog.md" if md_path.exists() else None
        ),
    )


# ─── Targets extraction ────────────────────────────────────────

def _extract_targets(faithfulness, mitre, compliance, failure):
    """Pull pass/fail targets from each sub-block into a flat namespace."""
    out = {}

    # SHAP stability
    if faithfulness["_status"] in ("complete", "partial") and faithfulness.get("shap_stability"):
        h = faithfulness["shap_stability"].get("headline", {})
        out["shap_stability_mean"] = {
            "value": h.get("mean_stability_score"),
            "target": h.get("mean_stability_target"),
            "pass": h.get("headline_pass"),
        }
        out["shap_stability_pass_rate"] = {
            "value": h.get("pass_rate"),
            "target": h.get("pass_rate_target"),
            "pass": h.get("pass_rate_pass"),
        }

    # MVE-SHAP alignment
    if faithfulness["_status"] in ("complete", "partial") and faithfulness.get("mve_shap_alignment"):
        h = faithfulness["mve_shap_alignment"].get("headline", {})
        out["alignment_all_three"] = {
            "value": h.get("all_three_present_pct"),
            "target": h.get("all_three_target"),
            "pass": h.get("all_three_pass"),
        }
        out["alignment_at_least_two"] = {
            "value": h.get("at_least_two_present_pct"),
            "target": h.get("at_least_two_target"),
            "pass": h.get("at_least_two_pass"),
        }

    # MITRE audit + grounding
    if mitre["_status"] in ("complete", "partial"):
        if mitre.get("config_audit"):
            h = mitre["config_audit"].get("headline", {})
            out["mitre_audit"] = {
                "value": h.get("audit_pass"),
                "target": True,
                "pass": bool(h.get("audit_pass")),
            }
        if mitre.get("layer1_grounding"):
            h = mitre["layer1_grounding"].get("headline", {})
            out["mitre_grounding_rate"] = {
                "value": h.get("grounded_pct"),
                "target": h.get("target"),
                "pass": h.get("pass"),
            }

    # Compliance
    if compliance["_status"] in ("complete", "partial"):
        if compliance.get("word_budget_audit"):
            h = compliance["word_budget_audit"].get("headline", {})
            out["word_budget_audit"] = {
                "value": h.get("audit_pass"),
                "target": True,
                "pass": bool(h.get("audit_pass")),
            }
        if compliance.get("compliance_manifest_audit"):
            out["compliance_manifest_evidence"] = {
                "value": compliance["compliance_manifest_audit"].get("all_evidence_present"),
                "target": True,
                "pass": bool(compliance["compliance_manifest_audit"].get("all_evidence_present")),
            }

    # Failure catalog disclosure framing
    if failure["_status"] in ("complete", "partial") and failure.get("disclosure"):
        framing = failure["disclosure"].get("framing")
        out["failure_catalog_disclosure"] = {
            "value": framing,
            "target": "observation_not_improvement",
            "pass": framing == "observation_not_improvement",
        }

    return out


# ─── Headline ──────────────────────────────────────────────────

def _build_headline(faithfulness, mitre, compliance, user_study, failure):
    """One-line status per sub-RQ. Read this first."""
    blocks = {
        "rq2_a_compliance": compliance["_status"],
        "rq2_b_faithfulness": faithfulness["_status"],
        "rq2_c_user_study": user_study["_status"],
        "rq2_e_mitre_grounding": mitre["_status"],
        "rq2_d_failure_catalog": failure["_status"],
    }

    statuses = list(blocks.values())
    if all(s == "complete" for s in statuses):
        overall = "complete"
    elif all(s == "pending" for s in statuses):
        overall = "pending"
    else:
        overall = "partial"
        missing = [k for k, v in blocks.items() if v != "complete"]
        overall = f"partial — incomplete: {', '.join(missing)}"

    return {
        "_description": "Highest-level pass/fail per sub-RQ. Read this first.",
        **blocks,
        "_overall_status": overall,
    }


# ─── Main ──────────────────────────────────────────────────────

def main():
    faithfulness = _load_faithfulness()
    mitre = _load_mitre_grounding()
    compliance = _load_compliance()
    user_study = _load_user_study()
    failure = _load_failure_catalog()

    blocks = {
        "faithfulness": faithfulness,
        "mitre_grounding": mitre,
        "compliance": compliance,
        "user_study": user_study,
        "failure_catalog": failure,
    }
    tracks_present = [k for k, v in blocks.items()
                      if v["_status"] in ("complete", "partial")]
    tracks_pending = [k for k, v in blocks.items() if v["_status"] == "pending"]

    out = {
        "_meta": {
            "schema_version": "1.0",
            "generated_at": _now_iso(),
            "generated_by": "module6_evaluation/compute_rq2_metrics.py",
            "research_question": (
                "RQ2 — Can MVE provide role-tailored security explanations "
                "enabling non-specialist hospital stakeholders to make informed "
                "threat triage decisions?"
            ),
            "active_subquestions": ["RQ2.a", "RQ2.b", "RQ2.c", "RQ2.e"],
            "rescoped_subquestions": ["RQ2.d (moved to thesis §7.2.3)"],
            "blocks_present": tracks_present,
            "blocks_pending": tracks_pending,
        },
        "headline": _build_headline(faithfulness, mitre, compliance,
                                    user_study, failure),
        "faithfulness": faithfulness,
        "mitre_grounding": mitre,
        "compliance": compliance,
        "user_study": user_study,
        "failure_catalog": failure,
        "targets": _extract_targets(faithfulness, mitre, compliance, failure),
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2, default=str))
    print(f"Wrote {OUT_PATH.relative_to(REPO_ROOT)}")
    print(f"Overall: {out['headline']['_overall_status']}")
    for k, v in out['headline'].items():
        if k.startswith("rq2_"):
            print(f"  {k}: {v}")
    print(f"\nTargets ({sum(1 for t in out['targets'].values() if t.get('pass'))}"
          f"/{len(out['targets'])} pass):")
    for k, t in out['targets'].items():
        mark = "✓" if t.get("pass") else "✗"
        print(f"  {mark} {k}: value={t.get('value')} target={t.get('target')}")


if __name__ == "__main__":
    main()
```

### 5.4 Verification

```bash
python -m module6_evaluation.compute_rq2_metrics
cat results/rq2_metrics.json | python -m json.tool | head -30
# Inspect _meta, headline, targets
```

---

## 6. Phase 3 — Figure generator

### 6.1 Create `module6_evaluation/make_rq2_figures.py`

**Contract:**
- **Inputs:** sub-files from Tracks 1, 2, 4, 5 (matching the figure requirements).
- **Outputs:** PDFs in `results/figures/rq2_*.pdf`.
- **CLI:** `python -m module6_evaluation.make_rq2_figures` produces all; `--only <id>` regenerates one.
- **Runtime:** ~10 seconds total (matplotlib startup dominates).

### 6.2 Figure inventory

| Figure ID | Filename | Source | Paper section |
|---|---|---|---|
| `stability` | `rq2_shap_stability_histogram.pdf` | `rq2_shap_stability.json::distribution` | §5.2 |
| `alignment` | `rq2_mve_alignment_modes.pdf` | `rq2_mve_shap_alignment.json::by_mode` | §5.2 |
| `mitre` | `rq2_mitre_grounding_per_category.pdf` | `rq2_mitre_grounding.json::by_attack_category` | §5.5 |
| `user_study` | `rq2_user_study_per_role.pdf` | `analysis/outputs/rq2c_per_role.json::per_role` | §5.3 |
| `failures` | `rq2_failure_categories.pdf` | `rq2_failure_mode_catalog.json::summary.by_category` | §5.3 |

### 6.3 Implementation

```python
"""
make_rq2_figures.py
Generate paper-ready PDFs for RQ2 from canonical sub-files.

Usage:
  python -m module6_evaluation.make_rq2_figures              # all figures
  python -m module6_evaluation.make_rq2_figures --only stability  # one figure
  python -m module6_evaluation.make_rq2_figures --list       # list available IDs

Runtime: ~10 seconds total.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = REPO_ROOT / "results/figures"

FIGURES = {
    "stability":   "rq2_shap_stability_histogram.pdf",
    "alignment":   "rq2_mve_alignment_modes.pdf",
    "mitre":       "rq2_mitre_grounding_per_category.pdf",
    "user_study":  "rq2_user_study_per_role.pdf",
    "failures":    "rq2_failure_categories.pdf",
}


# ─── Helpers ───────────────────────────────────────────────────

def _load_json(rel: str):
    p = REPO_ROOT / rel
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return None


def _skip(name, reason):
    print(f"  [SKIP] {name}: {reason}")


def _saved(name, path):
    print(f"  [OK]   {name} → {path.relative_to(REPO_ROOT)}")


# ─── Figure 1: SHAP stability histogram ────────────────────────

def make_stability(out_path):
    data = _load_json("results/rq2_shap_stability.json")
    if not data:
        return _skip("stability", "rq2_shap_stability.json missing")

    dist = data.get("distribution", {})
    bins = dist.get("histogram_bins")
    counts = dist.get("histogram_counts")
    if not bins or not counts:
        return _skip("stability", "histogram data missing in distribution block")

    fig, ax = plt.subplots(figsize=(6, 4))
    bin_centers = [(bins[i] + bins[i+1]) / 2 for i in range(len(counts))]
    width = bins[1] - bins[0]
    ax.bar(bin_centers, counts, width=width * 0.9, edgecolor="black", alpha=0.7)
    threshold = data["_meta"]["config"]["stability_threshold"]
    ax.axvline(threshold, linestyle="--", color="red",
               label=f"Stability threshold ({threshold})")
    ax.set_xlabel("Per-alert stability score (mean top-3 overlap)")
    ax.set_ylabel("Count")
    ax.set_title("SHAP Stability — surfaced alerts")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, format="pdf")
    plt.close()
    _saved("stability", out_path)


# ─── Figure 2: Mode A vs B alignment ────────────────────────────

def make_alignment(out_path):
    data = _load_json("results/rq2_mve_shap_alignment.json")
    if not data:
        return _skip("alignment", "rq2_mve_shap_alignment.json missing")

    by_mode = data.get("by_mode", {})
    if not by_mode:
        return _skip("alignment", "by_mode block missing")

    modes = sorted(by_mode.keys())
    metrics = ["all_three_pct", "at_least_two_pct"]
    metric_labels = ["All 3 SHAP features in Layer 1", "≥2 SHAP features in Layer 1"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(modes))
    bar_width = 0.35

    for i, (m, lbl) in enumerate(zip(metrics, metric_labels)):
        vals = [by_mode[mode].get(m, 0) for mode in modes]
        ax.bar(x + i * bar_width, vals, bar_width, label=lbl)

    ax.set_xticks(x + bar_width / 2)
    ax.set_xticklabels([f"Mode {m}\n(n={by_mode[m]['n_evaluated']})" for m in modes])
    ax.set_ylabel("Proportion of alerts")
    ax.set_ylim(0, 1.05)
    ax.set_title("MVE-SHAP Alignment by Generation Mode")
    ax.axhline(0.80, linestyle=":", color="gray",
               label="Target (all 3) ≥ 0.80")
    ax.axhline(0.95, linestyle=":", color="lightgray",
               label="Target (≥2) ≥ 0.95")
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, format="pdf")
    plt.close()
    _saved("alignment", out_path)


# ─── Figure 3: MITRE grounding per attack category ──────────────

def make_mitre(out_path):
    data = _load_json("results/rq2_mitre_grounding.json")
    if not data:
        return _skip("mitre", "rq2_mitre_grounding.json missing")

    by_cat = data.get("by_attack_category", {})
    if not by_cat:
        return _skip("mitre", "by_attack_category block missing")

    cats = sorted(by_cat.keys())
    grounded = [by_cat[c].get("grounded_pct", 0) for c in cats]
    strict = [by_cat[c].get("strict_grounded_pct", 0) for c in cats]
    ns = [by_cat[c].get("n_evaluated", 0) for c in cats]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(cats))
    bar_width = 0.35
    ax.bar(x, grounded, bar_width, label="T-ID OR human name (lenient)")
    ax.bar(x + bar_width, strict, bar_width, label="T-ID AND human name (strict)")
    ax.set_xticks(x + bar_width / 2)
    ax.set_xticklabels([f"{c}\n(n={n})" for c, n in zip(cats, ns)],
                       rotation=15, ha="right")
    ax.set_ylabel("Proportion of alerts grounding MITRE")
    ax.set_ylim(0, 1.05)
    ax.axhline(0.90, linestyle=":", color="gray", label="Target ≥ 0.90")
    ax.set_title("MITRE Layer 1 Grounding — per attack category")
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, format="pdf")
    plt.close()
    _saved("mitre", out_path)


# ─── Figure 4: User study per role ──────────────────────────────

def make_user_study(out_path):
    data = _load_json("analysis/outputs/rq2c_per_role.json")
    if not data:
        return _skip("user_study", "rq2c_per_role.json missing — Track 4 incomplete")

    per_role = data.get("per_role", {})
    if not per_role:
        return _skip("user_study", "per_role block missing")

    roles = [r for r in ["IT_GENERALIST", "BIOMED_ENGINEER", "NURSE_MANAGER"]
             if r in per_role and "decision_time" in per_role[r]]
    if not roles:
        return _skip("user_study", "no roles with metrics present")

    # NOTE: under Path C (LLM personas), decision_time is dropped.
    # The renderer checks for its presence per cell.
    metrics_available = []
    for m in ["decision_time", "accuracy", "confidence"]:
        if all(per_role[r].get(m, {}).get("median_A") is not None
               for r in roles):
            metrics_available.append(m)

    if not metrics_available:
        return _skip("user_study", "no metrics have data across all roles")

    fig, axes = plt.subplots(1, len(metrics_available),
                             figsize=(4 * len(metrics_available), 4.5),
                             sharey=False)
    if len(metrics_available) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics_available):
        x = np.arange(len(roles))
        bar_width = 0.35
        a_vals = [per_role[r][metric].get("median_A", 0) for r in roles]
        b_vals = [per_role[r][metric].get("median_B", 0) for r in roles]
        ax.bar(x, a_vals, bar_width, label="Group A (with MVE)")
        ax.bar(x + bar_width, b_vals, bar_width, label="Group B (without MVE)")
        ax.set_xticks(x + bar_width / 2)
        ax.set_xticklabels([r.replace("_", "\n") for r in roles], fontsize=9)
        ax.set_title(metric.replace("_", " ").title())

        # Mark low-N cells visually
        for i, r in enumerate(roles):
            if per_role[r][metric].get("n_warning"):
                ax.annotate("low-n", xy=(i + bar_width / 2,
                                          max(a_vals[i], b_vals[i])),
                            ha="center", fontsize=7, color="red")
        ax.legend(fontsize=7, loc="best")

    fig.suptitle("User Study Outcomes — per role × per metric "
                 "(low-n cells marked)", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, format="pdf")
    plt.close()
    _saved("user_study", out_path)


# ─── Figure 5: Failure category counts ──────────────────────────

def make_failures(out_path):
    data = _load_json("results/rq2_failure_mode_catalog.json")
    if not data:
        return _skip("failures", "rq2_failure_mode_catalog.json missing")

    by_cat = data.get("summary", {}).get("by_category", {})
    if not by_cat:
        return _skip("failures", "summary.by_category missing")

    cats = list(by_cat.keys())
    counts = [by_cat[c] for c in cats]

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#4477aa", "#cc6677", "#117733", "#ddcc77", "#aaaaaa"]
    ax.bar(cats, counts, color=colors[:len(cats)], edgecolor="black")
    for i, n in enumerate(counts):
        ax.text(i, n + 0.5, str(n), ha="center", fontsize=9)
    ax.set_ylabel("Observations")
    ax.set_title("Failure Mode Catalog — observations per category\n"
                 "(observation, not improvement; see §7.2.3 for future work)",
                 fontsize=10)
    plt.xticks(rotation=15, ha="right", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, format="pdf")
    plt.close()
    _saved("failures", out_path)


# ─── Dispatch ──────────────────────────────────────────────────

GENERATORS = {
    "stability":   make_stability,
    "alignment":   make_alignment,
    "mitre":       make_mitre,
    "user_study":  make_user_study,
    "failures":    make_failures,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=list(FIGURES.keys()),
                    help="Generate only one figure")
    ap.add_argument("--list", action="store_true",
                    help="List figure IDs and exit")
    args = ap.parse_args()

    if args.list:
        for fid, fname in FIGURES.items():
            print(f"  {fid:12s} → results/figures/{fname}")
        return

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    to_run = [args.only] if args.only else list(FIGURES.keys())
    for fid in to_run:
        out_path = FIG_DIR / FIGURES[fid]
        GENERATORS[fid](out_path)


if __name__ == "__main__":
    main()
```

### 6.4 Verification

```bash
python -m module6_evaluation.make_rq2_figures
# Expected: 5 lines of [OK] or [SKIP] output

python -m module6_evaluation.make_rq2_figures --only stability
# Expected: only stability regenerated

ls results/figures/rq2_*.pdf
```

---

## 7. Phase 4 — CI acceptance test

### 7.1 Extend `tests/acceptance_tests.py`

Add this function (don't replace the file — append):

```python
def test_rq2_targets_met():
    """
    Asserts every RQ2 numeric target is met.
    Sub-tracks may be pending; this test fails only on actively-failed targets.
    """
    import json
    from pathlib import Path

    metrics_path = Path("results/rq2_metrics.json")
    assert metrics_path.exists(), (
        "Run module6_evaluation/compute_rq2_metrics.py first"
    )

    m = json.loads(metrics_path.read_text())
    targets = m.get("targets", {})
    failures = []
    for tid, t in targets.items():
        if t.get("pass") is False:
            failures.append({
                "target": tid,
                "value": t.get("value"),
                "target_value": t.get("target"),
            })
    assert not failures, (
        f"RQ2 targets failed: {failures}. "
        f"Inspect results/rq2_metrics.json for details."
    )
    # Note: targets that are MISSING (pending tracks) do not fail this test.
    # That's intentional — pending != failed.


def test_rq2_failure_catalog_framing():
    """
    DEFENSE-CRITICAL: failure catalog must be framed as observation, not
    improvement. Per RQ2.d rescope to thesis §7.2.3.
    """
    import json
    from pathlib import Path

    p = Path("results/rq2_metrics.json")
    if not p.exists():
        return  # pending — handled by other tests

    m = json.loads(p.read_text())
    catalog_block = m.get("failure_catalog", {})
    if catalog_block.get("_status") == "pending":
        return  # nothing to check yet

    disclosure = catalog_block.get("disclosure", {})
    assert disclosure.get("framing") == "observation_not_improvement", (
        "Failure catalog framing must be 'observation_not_improvement'. "
        "See spec §11 — RQ2.d rescoped to future work."
    )
    assert disclosure.get("iteration_performed") is False, (
        "iteration_performed must be False — no iteration was done in scope."
    )
```

### 7.2 Verification

```bash
pytest tests/acceptance_tests.py::test_rq2_targets_met -v
pytest tests/acceptance_tests.py::test_rq2_failure_catalog_framing -v
```

---

## 8. Execution order

```bash
# ─── PHASE 0: PRE-RENAME AUDIT ─────────────────────────────────
python scripts/discover_compute_rq2_metrics_callers.py > /tmp/rename_audit.txt
# DEVELOPER CONFIRMS: file content, caller list, output path semantics.

# ─── PHASE 1: ATOMIC RENAME ────────────────────────────────────
# Single commit:
#   1. git mv module6_evaluation/compute_rq2_metrics.py compute_detection_metrics.py
#   2. Update docstring inside the renamed file
#   3. Change OUT path to results/detection_metrics.json
#   4. Update every caller from Phase 0
#   5. Update ARCHITECTURE.md line 39
pytest tests/                                     # all previously-green still green
python -m module6_evaluation.compute_detection_metrics
ls results/detection_metrics.json

# ─── PHASE 2: CANONICAL AGGREGATOR ─────────────────────────────
# Create module6_evaluation/compute_rq2_metrics.py
python -m module6_evaluation.compute_rq2_metrics
cat results/rq2_metrics.json | python -m json.tool | head -30

# ─── PHASE 3: FIGURES ──────────────────────────────────────────
python -m module6_evaluation.make_rq2_figures
ls results/figures/rq2_*.pdf

# ─── PHASE 4: CI GATE ──────────────────────────────────────────
# Append the two test functions to tests/acceptance_tests.py
pytest tests/acceptance_tests.py::test_rq2_targets_met -v
pytest tests/acceptance_tests.py::test_rq2_failure_catalog_framing -v

# ─── FINAL VERIFICATION ────────────────────────────────────────
pytest tests/ -v
ls results/rq2_metrics.json \
   results/detection_metrics.json \
   results/figures/rq2_*.pdf
```

---

## 9. Open questions to surface (DO NOT GUESS)

Claude Code must pause and ask:

1. **Phase 0 — existing file content.** Is the existing `compute_rq2_metrics.py` purely producing detection metrics, or has it drifted to also produce MVE metrics? If mixed, Phase 1 needs a split, not just a rename.
2. **Phase 0 — caller list.** What callers does the grep find? Each needs updating in the rename commit.
3. **Phase 0 — output path semantics.** Does `results/rq2_metrics.json` currently store detection metrics? If yes, the rename also relocates: detection → `results/detection_metrics.json`, MVE → `results/rq2_metrics.json` (new). If no, the path semantics are simpler.
4. **Phase 3 — `make_user_study` figure under Path C.** If Track 4 is resolved as Path C (LLM personas), `decision_time` is dropped from the per-role JSON. The renderer's `metrics_available` check handles this automatically, but the figure title should explicitly say "LLM-persona simulation" not "User study" in that case. Confirm wording.
5. **Phase 3 — figure aesthetics.** The current spec uses matplotlib defaults. If the paper has a specific style guide (font family, color palette, sizes), update the figure functions to apply it.

---

## 10. Coverage map — closing items → pipeline phase

| Item | Phase | Output |
|---|---|---|
| Senior engineer review naming-clash concern | 1 | File rename |
| `compute_rq2_metrics.py` canonical for MVE | 2 | New file created |
| `results/rq2_metrics.json` canonical for MVE | 2 | New JSON produced |
| Cross-track aggregation | 2 | `_status` per sub-block |
| Pass/fail target extraction | 2 | `targets` flat namespace |
| 4 paper PDFs + failure catalog PDF | 3 | `results/figures/rq2_*.pdf` |
| CI gate for RQ2 | 4 | `test_rq2_targets_met` |
| Defense-critical disclosure (catalog framing) | 4 | `test_rq2_failure_catalog_framing` |
| ARCHITECTURE.md update | 1 | line 39 reference updated |

---

## 11. Defense talking points this enables

When a reviewer asks closing questions about RQ2:

- **"What's the single source of truth for RQ2 results?"**
  *"`results/rq2_metrics.json`. It aggregates every Track 1-5 sub-file. The `headline` block gives status per sub-RQ; the `targets` block gives pass/fail per numeric target. Every claim in the paper's RQ2 chapter points to this file."*

- **"What about the naming clash you mentioned in the senior engineer review?"**
  *"Resolved. The old `compute_rq2_metrics.py` was renamed to `compute_detection_metrics.py` and now writes to `results/detection_metrics.json`. The new `compute_rq2_metrics.py` is the canonical MVE aggregator. ARCHITECTURE.md reflects the change."*

- **"How do you handle pending sub-tracks (e.g., user study not complete)?"**
  *"Each sub-block carries a `_status` field. CI tests assert that completed targets pass; pending tracks don't fail the CI. The `headline._overall_status` field tells you at a glance which sub-RQs are complete vs partial vs pending."*

- **"Can you regenerate the paper figures from the JSON?"**
  *"`python -m module6_evaluation.make_rq2_figures` produces all five PDFs from the canonical sub-files. `--only <id>` regenerates one. Figures cannot drift from the JSON since they're rendered from it deterministically."*

---

## 12. What this spec deliberately does NOT do

- **Compute any new metrics.** It only aggregates from existing sub-files.
- **Modify Track 1-5 sub-files.** Aggregator is read-only on them.
- **Generate the paper's prose.** Markdown rendering is each track's own concern.
- **Resolve the Track 4 strategic fork (Path A/B/C).** That's a separate decision; the aggregator handles whichever path is chosen via the `_status` field.

---

## End of spec

Implementation order: Phase 0 (audit) → Phase 1 (rename, atomic commit) → Phase 2 (aggregator) → Phase 3 (figures) → Phase 4 (CI gate). Phase 1 is the only step that touches existing code; everything else is additive. After this spec is implemented, RQ2 is complete and defense-ready (modulo Track 4 data collection and Path C resolution).