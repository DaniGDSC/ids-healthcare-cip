# RQ2 Failure Mode Catalog Pipeline — Observation Aggregator

**Project:** XAI-IDS-Healthcare
**Scope:** RQ2.d (RESCOPED) — Catalog of MVE failure modes systematically identified through structured evaluation.
**Purpose:** Single, self-contained spec for aggregating failure observations from Tracks 1, 2, 3, and 4 into a single catalog. Hand to Claude Code.
**Status of design:** All decisions locked. Three `DO NOT GUESS` checkpoints (Track 1/2 output paths, qualitative themes source provenance, impact-metric crosswalk).

---

## 0. How to use this spec

1. Track 5 is **dependent** — it cannot run until at least one of Tracks 1, 2, 3, or 4 has produced outputs. The catalog gracefully handles partial inputs.
2. Implementation order is Phase 0 → Phase 1 (manifest) → Phase 2 (aggregator) → Phase 3 (markdown renderer) → Phase 4 (tests).
3. Markers:
   - **DO NOT GUESS** — stop and ask the developer
   - **RESCOPED-NOTE** — this is observation work; framing must avoid implying "we fixed these"
   - **SOURCE-AGNOSTIC** — applies regardless of whether qualitative themes came from humans or LLM personas
4. Total expected size: one aggregator script, one renderer, two test files. No model inference. Runtime: sub-second.

---

## 1. Background: what this track produces

Track 5 is the *final integrative artifact* of the RQ2 pipeline. It takes the failure-shaped outputs of every other track and assembles them into a catalog the paper's §5.3 (Discussion) and §7.2.3 (Future Work) render from.

| Track | Failure source it contributes | Catalog category |
|---|---|---|
| Track 1 | `rq2_mve_shap_alignment.json::failure_examples` | TECHNICAL_LAYER_1 |
| Track 1 | `rq2_shap_stability.json::by_fusion_class.NOVEL_ANOMALY` (low-stability sub-pop) | TECHNICAL_LAYER_1 (known limitation, flagged) |
| Track 2 | `rq2_mitre_grounding.json::failure_examples` | MITRE_NOT_REFERENCED |
| Track 3 | `tests/test_step13_cross_role_consistency.py` test results | ROLE_VIEW_MISMATCH |
| Track 3 | `rq2_word_budget_audit.json::violations` | OTHER (word budget) |
| Track 4 | `survey/qualitative_themes.yaml::themes_per_role` (confusion_patterns) | Mapped to closest category; OTHER for unclassified |

**RESCOPED-NOTE:** Per thesis outline, RQ2.d is moved to §7.2.3 future work. The catalog produced here is framed as *observation*, not *improvement claim*. Every catalog entry's `recommended_iteration` field documents what future work would do — but no iteration has been performed.

---

## 2. Locked design decisions

| Decision | Resolution |
|---|---|
| Taxonomy | Fixed 4 categories (TECHNICAL_LAYER_1, MITRE_NOT_REFERENCED, DO_NOT_IGNORED, ROLE_VIEW_MISMATCH) + OTHER escape hatch |
| Quantification | Per-entry: frequency (count) + impact_metric (downstream metric affected if fixed) |
| Deliverable | JSON pipeline artifact + markdown render for paper + per-failure `recommended_iteration` field |
| Source-of-truth manifest | `config/rq2_failure_categories.yaml` defines the 5 categories with descriptions and impact-metric crosswalks |
| Defense framing | Every entry MUST carry a `_status: "observed_not_fixed"` field; catalog `_meta` carries a rescoping disclosure |
| Graceful degradation | Missing source files produce `_status: "source_unavailable"` per category, not a script crash |

---

## 3. Phase 0 — Source discovery (DO NOT GUESS)

Before writing the aggregator, Claude Code must verify which source files exist. The catalog must be runnable in *any* combination of tracks completed.

### 3.1 Discovery script

```python
# scripts/discover_failure_sources.py — TRANSIENT, delete after Phase 0
"""Inventory which Track 1/2/3/4 outputs exist."""
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

sources = {
    "track_1_alignment_failures": "results/rq2_mve_shap_alignment.json",
    "track_1_stability_by_class": "results/rq2_shap_stability.json",
    "track_2_grounding_failures": "results/rq2_mitre_grounding.json",
    "track_3_word_budget_violations": "results/rq2_word_budget_audit.json",
    "track_3_cross_role_pytest": "tests/test_step13_cross_role_consistency.py",
    "track_4_qualitative_themes": "survey/qualitative_themes.yaml",
}

found, missing = {}, []
for name, rel in sources.items():
    p = REPO_ROOT / rel
    if p.exists():
        found[name] = {"path": rel, "size_bytes": p.stat().st_size}
    else:
        missing.append({"name": name, "path": rel})

print(json.dumps({
    "found": found,
    "missing": missing,
    "n_found": len(found),
    "n_missing": len(missing),
}, indent=2))

print("\n" + "=" * 60)
print("Track 5 will degrade gracefully on missing inputs.")
print("Each missing source produces a 'source_unavailable' entry in the catalog.")
print("To produce a complete catalog, run prior tracks first.")
print("=" * 60)
```

### 3.2 What to confirm before Phase 1

1. **Track 1 outputs:** `rq2_mve_shap_alignment.json` has a `failure_examples` array per the faithfulness spec.
2. **Track 2 outputs:** `rq2_mitre_grounding.json` has a `failure_examples` array per the MITRE spec.
3. **Track 3 outputs:** `rq2_word_budget_audit.json` has a `violations` array per the compliance spec.
4. **Track 4 source provenance:** is `survey/qualitative_themes.yaml` themes coded from human participants or LLM personas? **SOURCE-AGNOSTIC** — the catalog handles either, but the markdown renderer needs to know which to attribute correctly.

### 3.3 Verification

```bash
python scripts/discover_failure_sources.py > /tmp/failure_sources.json
# Developer reviews; Track 5 will work even if some sources missing.
```

---

## 4. Phase 1 — Failure category manifest

### 4.1 Create `config/rq2_failure_categories.yaml`

This is the **pre-registered taxonomy.** It exists before any catalog is generated, which is what makes "fixed categories" defensible (no post-hoc cherry-picking).

```yaml
# config/rq2_failure_categories.yaml
# Pre-registered taxonomy of MVE failure modes for RQ2.d (rescoped to future work).
# Categories were defined in RQ2_expected_outputs.md §4.1 before any data collection.
# An OTHER bucket captures observations that don't fit the fixed categories.

schema_version: "1.0"
preregistered_date: "2026-02-01"   # Set to actual date when categories were locked
rescope_note: |
  RQ2.d (iteration on failure modes) was rescoped to future work in thesis
  Section 7.2.3 after the decision to use a single-round evaluation. The
  catalog below records observations only; no iteration has been performed.
  Each entry's recommended_iteration field describes what future work would do.

categories:
  - id: TECHNICAL_LAYER_1
    name: "Layer 1 too technical for non-technical roles"
    description: |
      MVE Layer 1 (the WHY layer) uses jargon or feature names that
      non-technical operators (nurses, biomed engineers) cannot parse.
    sources:
      - results/rq2_mve_shap_alignment.json
      - survey/qualitative_themes.yaml (confusion_patterns mentioning vocabulary/jargon)
    impact_metric: "User study confidence score (RQ2.c per-role confidence)"
    recommended_iteration: |
      Simplify vocabulary post-Mode-A generation; map raw feature names to
      role-appropriate phrasings (e.g., 'fwd_pkts_tot' → 'outbound data volume'
      for nurses). Re-run alignment metric to verify SHAP features still
      surface in simplified form. Round 2 evaluation required.

  - id: MITRE_NOT_REFERENCED
    name: "MITRE technique not referenced or not understood"
    description: |
      Layer 1 fails to reference the mapped MITRE technique (T-ID or human
      name), or references it without sufficient context for the reader to
      understand what it means.
    sources:
      - results/rq2_mitre_grounding.json (failure_examples)
      - survey/qualitative_themes.yaml (confusion_patterns mentioning MITRE/T-ID)
    impact_metric: "MITRE grounding rate (RQ2.e: layer1_grounding.headline.grounded_pct)"
    recommended_iteration: |
      Add plain-language gloss alongside T-IDs (e.g., 'T1071 (command-and-
      control communication)'). Strengthen Mode A prompt to require both
      T-ID and human name. Re-run grounding metric.

  - id: DO_NOT_IGNORED
    name: "DO_NOT clinical-safety actions ignored or under-emphasized"
    description: |
      Layer 3 DO_NOT constraints (e.g., 'do not disconnect ventilator')
      are present but not visually emphasized; operator overlooks them in
      study or skips the constraint when reporting action_taken.
    sources:
      - tests/test_step13_cross_role_consistency.py (Invariant 7 sanity test)
      - survey/qualitative_themes.yaml (themes mentioning DO_NOT skipping)
    impact_metric: "User study accuracy on CRITICAL+clinical-device alerts (RQ2.c per-role accuracy)"
    recommended_iteration: |
      Visual emphasis in UI for DO_NOT (red badge, separate block, required
      acknowledgement before submission). UI change, not generator change.
      Re-run study with new UI.

  - id: ROLE_VIEW_MISMATCH
    name: "Role view inappropriate for participant background"
    description: |
      A participant assigned an IT_GENERALIST view (per self-selection or
      assignment) reports the explanation didn't match their actual operational
      mental model — or two participants in different roles disagree about
      what the alert means despite a shared anchor.
    sources:
      - tests/test_step13_cross_role_consistency.py (Layer 3 differentiation test)
      - survey/qualitative_themes.yaml (themes mentioning role mismatch)
    impact_metric: "User study per-role differentiation (RQ2.c per-role decision_time variance)"
    recommended_iteration: |
      Better role inference at study signup (e.g., short skill survey instead
      of self-selection). For deployment: integrate with hospital identity
      provider to derive role from job title. Re-run study with better role
      assignment.

  - id: OTHER
    name: "Unclassified failure observations"
    description: |
      Observations that do not fit the four pre-registered categories.
      The size of this bucket is itself diagnostic — if it is large or
      growing across runs, the taxonomy needs revision.
    sources:
      - results/rq2_word_budget_audit.json (word budget violations)
      - results/rq2_shap_stability.json (NOVEL_ANOMALY low-stability cluster)
      - any source not mapped to the four fixed categories
    impact_metric: "Varies by observation; documented per-entry"
    recommended_iteration: |
      Triage each entry. If a pattern emerges, propose a new fixed
      category for the next iteration of the taxonomy.
```

### 4.2 Verification

```bash
python -c "
import yaml
from pathlib import Path
doc = yaml.safe_load(Path('config/rq2_failure_categories.yaml').read_text())
print(f'Categories: {len(doc[\"categories\"])}')
for c in doc['categories']:
    print(f'  {c[\"id\"]:30s} \"{c[\"name\"]}\"')
"
# Expected: 5 categories listed (4 fixed + OTHER)
```

---

## 5. Phase 2 — Aggregator script

### 5.1 Create `analysis/compile_failure_modes.py`

**Contract:**
- **Inputs:** all available outputs from Tracks 1, 2, 3, 4 + the manifest from Phase 1.
- **Output:** `results/rq2_failure_mode_catalog.json`.
- **Runtime:** sub-second.
- **Side effects:** writes one JSON file. Reads-only on all sources.
- **Graceful degradation:** missing sources produce `_status: "source_unavailable"` per category, not a crash.

### 5.2 Output schema

```json
{
  "_meta": {
    "schema_version": "1.0",
    "generated_at": "<ISO-8601>",
    "generated_by": "analysis/compile_failure_modes.py",
    "manifest_path": "config/rq2_failure_categories.yaml",
    "preregistered_date": "2026-02-01",
    "rescope_note": "RQ2.d rescoped to future work in thesis §7.2.3...",
    "sources_used": [
      "results/rq2_mve_shap_alignment.json",
      "results/rq2_mitre_grounding.json",
      "results/rq2_word_budget_audit.json"
    ],
    "sources_missing": [
      "survey/qualitative_themes.yaml"
    ],
    "_warning": "Catalog is INCOMPLETE — qualitative_themes.yaml unavailable."
  },
  "_disclosure": {
    "framing": "observation_not_improvement",
    "iteration_performed": false,
    "evaluation_rounds": 1,
    "intended_use": (
      "Failure observations are reported as evidence that the MVE evaluation "
      "framework can systematically surface problems. Per thesis Section "
      "7.2.3, addressing these is future work."
    )
  },
  "summary": {
    "total_observations": 47,
    "by_category": {
      "TECHNICAL_LAYER_1": 18,
      "MITRE_NOT_REFERENCED": 12,
      "DO_NOT_IGNORED": 3,
      "ROLE_VIEW_MISMATCH": 5,
      "OTHER": 9
    },
    "other_bucket_diagnostic": {
      "size": 9,
      "size_pct": 0.19,
      "_assessment": "Within acceptable range (<20%). Taxonomy stable."
    }
  },
  "catalog": {
    "TECHNICAL_LAYER_1": {
      "name": "Layer 1 too technical for non-technical roles",
      "description": "...",
      "impact_metric": "User study confidence score (RQ2.c per-role confidence)",
      "recommended_iteration": "Simplify vocabulary post-Mode-A generation...",
      "n_observations": 18,
      "_status": "observed_not_fixed",
      "evidence": [
        {
          "source": "results/rq2_mve_shap_alignment.json",
          "source_entry": "failure_examples[0]",
          "row_id": 1247,
          "mode": "A",
          "summary": "Top-3 SHAP features mapped to 'forward packet count', 'flow duration', 'forward packet rate' — none appeared in Layer 1.",
          "excerpt": "...this device shows abnormal outbound traffic patterns..."
        }
        // ... more evidence entries (capped at 10 per category for readability)
      ],
      "evidence_truncated_at": 10,
      "evidence_total_count": 18
    },
    "MITRE_NOT_REFERENCED": { ... },
    "DO_NOT_IGNORED": {
      "n_observations": 0,
      "_status": "no_observations_collected",
      "_note": "Source files exist but no failures of this type observed."
    },
    "ROLE_VIEW_MISMATCH": { ... },
    "OTHER": {
      "n_observations": 9,
      "evidence": [
        {
          "source": "results/rq2_word_budget_audit.json",
          "source_entry": "violations[3]",
          "summary": "Layer 1 word count 43 > budget 40 (Mode A output, post-truncation reached limit)",
          "_subtype": "word_budget"
        },
        {
          "source": "results/rq2_shap_stability.json",
          "source_entry": "by_fusion_class.NOVEL_ANOMALY",
          "summary": "NOVEL_ANOMALY alerts (n=73) show mean stability 0.62, below 0.90 threshold. Known limitation: XGBoost SHAP not faithful for DAE-driven alerts.",
          "_subtype": "known_limitation_novel_anomaly"
        }
      ]
    }
  }
}
```

### 5.3 Implementation outline

```python
"""
compile_failure_modes.py
Aggregate failure-shaped observations from Tracks 1, 2, 3, 4 into a single catalog.

Inputs (any subset; missing inputs produce graceful degradation):
  - results/rq2_mve_shap_alignment.json
  - results/rq2_shap_stability.json
  - results/rq2_mitre_grounding.json
  - results/rq2_word_budget_audit.json
  - survey/qualitative_themes.yaml
  + config/rq2_failure_categories.yaml (manifest, required)

Output:
  results/rq2_failure_mode_catalog.json
Runtime: sub-second.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "config/rq2_failure_categories.yaml"
OUT_PATH = REPO_ROOT / "results/rq2_failure_mode_catalog.json"

EVIDENCE_TRUNCATION = 10  # max evidence entries per category in JSON


def _load_manifest():
    if not MANIFEST_PATH.exists():
        raise SystemExit(
            f"Manifest missing: {MANIFEST_PATH}. "
            f"Create per Phase 1 spec before running."
        )
    return yaml.safe_load(MANIFEST_PATH.read_text())


def _try_load_json(rel_path: str):
    """Load a JSON file if it exists; return None otherwise."""
    p = REPO_ROOT / rel_path
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return None


def _try_load_yaml(rel_path: str):
    p = REPO_ROOT / rel_path
    if not p.exists():
        return None
    try:
        return yaml.safe_load(p.read_text())
    except yaml.YAMLError:
        return None


# ─── Per-source extractors ─────────────────────────────────────

def _extract_alignment_failures(data):
    """Track 1 alignment failure examples → TECHNICAL_LAYER_1."""
    if not data:
        return []
    out = []
    for i, ex in enumerate(data.get("failure_examples", [])):
        out.append({
            "source": "results/rq2_mve_shap_alignment.json",
            "source_entry": f"failure_examples[{i}]",
            "row_id": ex.get("row_id"),
            "mode": ex.get("mode"),
            "summary": (
                f"Top-3 SHAP features mapped to {ex.get('human_readable', [])} "
                f"— none appeared in Layer 1."
            ),
            "excerpt": (ex.get("layer1_excerpt") or "")[:200],
            "_category_assignment": "TECHNICAL_LAYER_1",
        })
    return out


def _extract_novel_anomaly_limitation(data):
    """Track 1 SHAP stability NOVEL_ANOMALY block → OTHER (known limitation)."""
    if not data:
        return []
    by_class = data.get("by_fusion_class", {})
    novel = by_class.get("NOVEL_ANOMALY")
    if not novel or novel.get("n", 0) == 0:
        return []
    return [{
        "source": "results/rq2_shap_stability.json",
        "source_entry": "by_fusion_class.NOVEL_ANOMALY",
        "summary": (
            f"NOVEL_ANOMALY alerts (n={novel['n']}) show mean stability "
            f"{novel.get('mean', 0):.2f}, below the 0.90 threshold. "
            "Known limitation: XGBoost SHAP not faithful for DAE-driven alerts."
        ),
        "_category_assignment": "OTHER",
        "_subtype": "known_limitation_novel_anomaly",
    }]


def _extract_mitre_failures(data):
    """Track 2 grounding failure examples → MITRE_NOT_REFERENCED."""
    if not data:
        return []
    out = []
    for i, ex in enumerate(data.get("failure_examples", [])):
        out.append({
            "source": "results/rq2_mitre_grounding.json",
            "source_entry": f"failure_examples[{i}]",
            "row_id": ex.get("row_id"),
            "mode": ex.get("mode"),
            "summary": (
                f"Expected MITRE terms {ex.get('expected_terms', [])} not found "
                f"in Layer 1 for category={ex.get('category')}."
            ),
            "excerpt": (ex.get("layer1_excerpt") or "")[:200],
            "_category_assignment": "MITRE_NOT_REFERENCED",
        })
    return out


def _extract_word_budget_violations(data):
    """Track 3 word budget violations → OTHER (word_budget)."""
    if not data:
        return []
    out = []
    for i, v in enumerate(data.get("violations", [])):
        out.append({
            "source": "results/rq2_word_budget_audit.json",
            "source_entry": f"violations[{i}]",
            "row_id": v.get("row_id"),
            "mode": v.get("mode"),
            "summary": (
                f"Word budget violation: {v.get('violations')}. "
                f"Total: {v.get('total')} words."
            ),
            "_category_assignment": "OTHER",
            "_subtype": "word_budget",
        })
    return out


def _extract_qualitative_themes(themes_doc):
    """
    Track 4 confusion_patterns → mapped to fixed categories or OTHER.

    Mapping heuristic: scan theme name for keywords. Defaults to OTHER for
    anything unmapped.
    """
    if not themes_doc:
        return []
    out = []
    keyword_map = [
        (["jargon", "technical", "vocabulary", "feature name"], "TECHNICAL_LAYER_1"),
        (["mitre", "t1", "technique", "att&ck", "attck"], "MITRE_NOT_REFERENCED"),
        (["do not", "do_not", "constraint", "safety", "isolation"], "DO_NOT_IGNORED"),
        (["role", "view mismatch", "irrelevant", "wrong audience"], "ROLE_VIEW_MISMATCH"),
    ]

    themes_per_role = themes_doc.get("themes_per_role", {})
    for role, blocks in themes_per_role.items():
        for theme in blocks.get("confusion_patterns", []):
            theme_text = (theme.get("theme") or "").lower()
            category = "OTHER"
            for keywords, cat in keyword_map:
                if any(k in theme_text for k in keywords):
                    category = cat
                    break
            out.append({
                "source": "survey/qualitative_themes.yaml",
                "source_entry": f"themes_per_role.{role}.confusion_patterns",
                "summary": (
                    f"[{role}] {theme.get('theme', '<unnamed>')} "
                    f"(frequency: {theme.get('frequency', 0)})"
                ),
                "excerpt": (theme.get("example_quote") or "")[:200],
                "_category_assignment": category,
                "_subtype": "qualitative_theme",
            })
    return out


# ─── Assembly ──────────────────────────────────────────────────

def main():
    manifest = _load_manifest()
    categories = manifest.get("categories", [])
    category_by_id = {c["id"]: c for c in categories}

    # Try every source
    src_alignment = _try_load_json("results/rq2_mve_shap_alignment.json")
    src_stability = _try_load_json("results/rq2_shap_stability.json")
    src_mitre = _try_load_json("results/rq2_mitre_grounding.json")
    src_word = _try_load_json("results/rq2_word_budget_audit.json")
    src_themes = _try_load_yaml("survey/qualitative_themes.yaml")

    sources_used, sources_missing = [], []
    for name, data in [
        ("results/rq2_mve_shap_alignment.json", src_alignment),
        ("results/rq2_shap_stability.json", src_stability),
        ("results/rq2_mitre_grounding.json", src_mitre),
        ("results/rq2_word_budget_audit.json", src_word),
        ("survey/qualitative_themes.yaml", src_themes),
    ]:
        (sources_used if data is not None else sources_missing).append(name)

    # Collect all observations
    all_evidence = []
    all_evidence += _extract_alignment_failures(src_alignment)
    all_evidence += _extract_novel_anomaly_limitation(src_stability)
    all_evidence += _extract_mitre_failures(src_mitre)
    all_evidence += _extract_word_budget_violations(src_word)
    all_evidence += _extract_qualitative_themes(src_themes)

    # Group by category
    catalog = {}
    for cat_id in [c["id"] for c in categories]:
        cat_meta = category_by_id[cat_id]
        cat_evidence = [
            e for e in all_evidence if e.get("_category_assignment") == cat_id
        ]
        catalog[cat_id] = {
            "name": cat_meta["name"],
            "description": cat_meta["description"],
            "impact_metric": cat_meta.get("impact_metric"),
            "recommended_iteration": cat_meta.get("recommended_iteration"),
            "n_observations": len(cat_evidence),
            "_status": (
                "observed_not_fixed" if cat_evidence else "no_observations_collected"
            ),
            "evidence": cat_evidence[:EVIDENCE_TRUNCATION],
            "evidence_truncated_at": EVIDENCE_TRUNCATION,
            "evidence_total_count": len(cat_evidence),
        }

    # Summary
    by_cat = {cid: len(catalog[cid]["evidence"]) for cid in catalog}
    by_cat = {cid: catalog[cid]["n_observations"] for cid in catalog}
    total = sum(by_cat.values())
    other_size = by_cat.get("OTHER", 0)
    other_pct = (other_size / total) if total else 0
    other_assessment = (
        "Within acceptable range (<20%). Taxonomy stable."
        if other_pct < 0.20 else
        "Above 20% of total — taxonomy may need revision (consider adding "
        "a new fixed category in next iteration)."
    )

    # Top-level disclosure block — defense-critical framing
    disclosure = {
        "framing": "observation_not_improvement",
        "iteration_performed": False,
        "evaluation_rounds": 1,
        "intended_use": (
            "Failure observations are reported as evidence that the MVE "
            "evaluation framework can systematically surface problems. "
            "Per thesis Section 7.2.3, addressing these is future work."
        ),
    }

    out = {
        "_meta": {
            "schema_version": "1.0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "generated_by": "analysis/compile_failure_modes.py",
            "manifest_path": "config/rq2_failure_categories.yaml",
            "preregistered_date": manifest.get("preregistered_date"),
            "rescope_note": manifest.get("rescope_note"),
            "sources_used": sources_used,
            "sources_missing": sources_missing,
            "_warning": (
                f"Catalog is INCOMPLETE — {len(sources_missing)} source(s) "
                f"unavailable: {sources_missing}"
            ) if sources_missing else None,
        },
        "_disclosure": disclosure,
        "summary": {
            "total_observations": total,
            "by_category": by_cat,
            "other_bucket_diagnostic": {
                "size": other_size,
                "size_pct": other_pct,
                "_assessment": other_assessment,
            },
        },
        "catalog": catalog,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2, default=str))

    print(f"Wrote {OUT_PATH.relative_to(REPO_ROOT)}")
    print(f"Total observations: {total}")
    for cid, n in by_cat.items():
        print(f"  {cid:25s} {n}")
    if sources_missing:
        print(f"WARNING: {len(sources_missing)} source(s) missing")


if __name__ == "__main__":
    main()
```

### 5.4 Verification

```bash
python -m analysis.compile_failure_modes
cat results/rq2_failure_mode_catalog.json | python -m json.tool | head -50
# Expected: counts per category, _warning if any source missing
```

---

## 6. Phase 3 — Markdown renderer

### 6.1 Create `analysis/render_failure_catalog_markdown.py`

**Purpose:** produce paper-ready markdown from the catalog JSON. Two outputs:

1. **Summary table** — for thesis §5.3 (Discussion)
2. **Per-category detail with recommended iteration** — for thesis §7.2.3 (Future Work)

```python
"""
render_failure_catalog_markdown.py
Render results/rq2_failure_mode_catalog.json into paper-ready markdown.

Outputs:
  - results/rq2_failure_mode_catalog.md         (for thesis §5.3 + §7.2.3)
Runtime: sub-second.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CATALOG_JSON = REPO_ROOT / "results/rq2_failure_mode_catalog.json"
OUT_MD = REPO_ROOT / "results/rq2_failure_mode_catalog.md"


def main():
    if not CATALOG_JSON.exists():
        raise SystemExit(
            f"{CATALOG_JSON} missing — run analysis/compile_failure_modes.py first"
        )
    cat = json.loads(CATALOG_JSON.read_text())

    lines = []
    lines.append("# RQ2 — Failure Mode Catalog")
    lines.append("")
    lines.append(f"*Generated by `analysis/compile_failure_modes.py` "
                 f"on {cat['_meta']['generated_at']}.*")
    lines.append("")
    lines.append(f"*Manifest pre-registered: "
                 f"{cat['_meta'].get('preregistered_date', 'unknown')}.*")
    lines.append("")

    # Disclosure block — defense-critical, always rendered first
    d = cat["_disclosure"]
    lines.append("## Disclosure")
    lines.append("")
    lines.append(f"- **Framing:** {d['framing']}")
    lines.append(f"- **Iteration performed:** {d['iteration_performed']}")
    lines.append(f"- **Evaluation rounds:** {d['evaluation_rounds']}")
    lines.append(f"- **Intended use:** {d['intended_use']}")
    lines.append("")
    lines.append("> This catalog records observations. Per thesis Section 7.2.3, "
                 "addressing the modes catalogued here is **future work**, not "
                 "a claim made in this thesis.")
    lines.append("")
    if cat["_meta"].get("_warning"):
        lines.append(f"> ⚠ **{cat['_meta']['_warning']}**")
        lines.append("")

    # Summary table
    lines.append("## 1. Summary — observations per category")
    lines.append("")
    lines.append("| Category | Observations | Impact metric (would affect if fixed) |")
    lines.append("|---|---:|---|")
    for cid, n in cat["summary"]["by_category"].items():
        impact = cat["catalog"][cid].get("impact_metric", "—")
        lines.append(f"| **{cid}** | {n} | {impact} |")
    total = cat["summary"]["total_observations"]
    lines.append(f"| **TOTAL** | **{total}** | |")
    lines.append("")

    # Other bucket diagnostic
    other = cat["summary"]["other_bucket_diagnostic"]
    lines.append(f"**OTHER bucket diagnostic:** {other['size']} of {total} "
                 f"observations ({other['size_pct']:.1%}) — {other['_assessment']}")
    lines.append("")

    # Per-category detail
    lines.append("## 2. Per-category observations and recommended iteration")
    lines.append("")
    for cid, entry in cat["catalog"].items():
        lines.append(f"### {cid} — {entry['name']}")
        lines.append("")
        lines.append(f"**Description.** {entry['description'].strip()}")
        lines.append("")
        lines.append(f"**Observations:** {entry['n_observations']}  ")
        lines.append(f"**Status:** `{entry['_status']}`  ")
        lines.append(f"**Impact metric:** {entry.get('impact_metric', '—')}")
        lines.append("")
        lines.append("**Recommended iteration (future work):**")
        lines.append("")
        lines.append("> " + (entry.get('recommended_iteration') or '').strip()
                     .replace("\n", "\n> "))
        lines.append("")

        if entry["evidence"]:
            lines.append("**Sample evidence:**")
            lines.append("")
            for ev in entry["evidence"][:5]:
                lines.append(f"- *{ev.get('source', '')}* — {ev.get('summary', '')}")
                excerpt = ev.get("excerpt") or ""
                if excerpt:
                    lines.append(f"  > {excerpt[:150]}")
            if entry["evidence_total_count"] > 5:
                lines.append(f"- *(+{entry['evidence_total_count'] - 5} more "
                             "in JSON catalog)*")
            lines.append("")
        else:
            lines.append("*No observations of this type collected.*")
            lines.append("")

    # Sources used / missing
    lines.append("## 3. Sources")
    lines.append("")
    lines.append("**Sources used:**")
    for s in cat["_meta"]["sources_used"]:
        lines.append(f"- `{s}`")
    lines.append("")
    if cat["_meta"].get("sources_missing"):
        lines.append("**Sources missing (catalog incomplete):**")
        for s in cat["_meta"]["sources_missing"]:
            lines.append(f"- `{s}`")
        lines.append("")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines))
    print(f"Wrote {OUT_MD.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
```

### 6.2 Verification

```bash
python -m analysis.render_failure_catalog_markdown
head -40 results/rq2_failure_mode_catalog.md
```

---

## 7. Phase 4 — Tests

### 7.1 Create `tests/test_failure_mode_catalog.py`

```python
"""Schema + invariant tests for the failure mode catalog."""
import json
from pathlib import Path

import pytest

CATALOG = Path("results/rq2_failure_mode_catalog.json")


@pytest.fixture(scope="module")
def catalog():
    if not CATALOG.exists():
        pytest.skip("Run analysis/compile_failure_modes.py first")
    return json.loads(CATALOG.read_text())


def test_schema_complete(catalog):
    for key in ["_meta", "_disclosure", "summary", "catalog"]:
        assert key in catalog


def test_all_five_categories_present(catalog):
    """The four fixed categories + OTHER must all be present, even if empty."""
    required = {"TECHNICAL_LAYER_1", "MITRE_NOT_REFERENCED",
                "DO_NOT_IGNORED", "ROLE_VIEW_MISMATCH", "OTHER"}
    assert required.issubset(set(catalog["catalog"].keys())), (
        f"Missing categories: {required - set(catalog['catalog'].keys())}"
    )


def test_disclosure_framing_is_observation(catalog):
    """Defense-critical: framing MUST NOT claim improvement."""
    d = catalog["_disclosure"]
    assert d["framing"] == "observation_not_improvement", (
        "Catalog framing must be 'observation_not_improvement' "
        "(RQ2.d rescoped to future work)."
    )
    assert d["iteration_performed"] is False, (
        "iteration_performed must be False — no iteration was done."
    )


def test_every_entry_has_status(catalog):
    """Every catalog entry must carry a _status field."""
    for cid, entry in catalog["catalog"].items():
        assert "_status" in entry, f"Missing _status for {cid}"
        assert entry["_status"] in {
            "observed_not_fixed", "no_observations_collected", "source_unavailable"
        }, f"Invalid _status for {cid}: {entry['_status']}"


def test_other_bucket_diagnostic_under_threshold(catalog):
    """Sanity: if OTHER >40% of observations, the taxonomy is broken."""
    other = catalog["summary"]["other_bucket_diagnostic"]
    if other["size"] > 0:
        assert other["size_pct"] < 0.40, (
            f"OTHER bucket is {other['size_pct']:.1%} of observations — "
            "taxonomy likely needs new fixed category."
        )


def test_every_entry_has_recommended_iteration(catalog):
    """Future-work column must be populated for every category."""
    for cid, entry in catalog["catalog"].items():
        ri = entry.get("recommended_iteration", "")
        assert ri and ri.strip(), (
            f"Category {cid} missing recommended_iteration. "
            "Update config/rq2_failure_categories.yaml."
        )
```

### 7.2 Verification

```bash
pytest tests/test_failure_mode_catalog.py -v
# Expected: 6 tests pass
```

---

## 8. Execution order

```bash
# ─── PHASE 0: SOURCE DISCOVERY ─────────────────────────────────
python scripts/discover_failure_sources.py > /tmp/failure_sources.json
# Developer reviews what's available.

# ─── PHASE 1: MANIFEST ─────────────────────────────────────────
# Create config/rq2_failure_categories.yaml
python -c "import yaml; yaml.safe_load(open('config/rq2_failure_categories.yaml'))"

# ─── PHASE 2: AGGREGATOR ───────────────────────────────────────
python -m analysis.compile_failure_modes
# Inspect output
cat results/rq2_failure_mode_catalog.json | python -m json.tool | head -50

# ─── PHASE 3: MARKDOWN RENDERER ────────────────────────────────
python -m analysis.render_failure_catalog_markdown
head -40 results/rq2_failure_mode_catalog.md

# ─── PHASE 4: TESTS ────────────────────────────────────────────
pytest tests/test_failure_mode_catalog.py -v

# ─── FINAL VERIFICATION ────────────────────────────────────────
ls config/rq2_failure_categories.yaml \
   results/rq2_failure_mode_catalog.json \
   results/rq2_failure_mode_catalog.md
```

---

## 9. Integration with `compute_rq2_metrics.py`

```python
def _load_failure_catalog_subfile():
    p = REPO_ROOT / "results/rq2_failure_mode_catalog.json"
    if not p.exists():
        return {"_status": "pending"}
    cat = json.loads(p.read_text())
    return {
        "_status": "complete" if not cat["_meta"].get("sources_missing")
                   else "partial",
        "_merged_at": datetime.now(timezone.utc).isoformat(),
        "summary": cat["summary"],
        "disclosure": cat["_disclosure"],
        "catalog_path": "results/rq2_failure_mode_catalog.json",
        "catalog_md_path": "results/rq2_failure_mode_catalog.md",
    }
```

In the aggregator: `out["failure_catalog"] = _load_failure_catalog_subfile()`.

Notably, the aggregator only carries the *summary* of the catalog — not the full evidence arrays. The full catalog stays in its own JSON for paper rendering.

---

## 10. Open questions to surface (DO NOT GUESS)

Claude Code must pause and ask:

1. **Phase 0 — qualitative theme provenance.** Were `survey/qualitative_themes.yaml` themes coded from human participants or LLM personas? (Track 4 strategic fork outcome.) The markdown renderer's attribution wording changes accordingly.
2. **Phase 1 — preregistered_date.** What's the actual date the 4-category taxonomy was decided? Put the truthful date in the manifest — it's defense-critical that this predates data collection.
3. **Phase 2 — qualitative theme keyword mapping.** The `keyword_map` in `_extract_qualitative_themes` is a heuristic. If themes turn out to use vocabulary the keywords don't capture, observations land in OTHER. Verify the mapping covers your actual coded themes after Track 4 completes.

---

## 11. Defense talking points this enables

When a reviewer asks about RQ2.d failures and iteration:

- **"Did you iterate on the failure modes?"**
  *"No — RQ2.d was rescoped to future work in §7.2.3 due to the single-round evaluation. The failure mode catalog in §5.3 documents what we observed; §7.2.3 documents what iteration would do."*

- **"How did you decide what counts as a 'failure mode'?"**
  *"Four categories were pre-registered in `config/rq2_failure_categories.yaml` on [date], before evaluation data was collected. An OTHER bucket captures observations that don't fit. The OTHER bucket size is reported as a diagnostic — if it grew large, that would signal taxonomy revision is needed."*

- **"How rigorous is the observation aggregation?"**
  *"All observations come from automated extraction of structured outputs (Tracks 1, 2, 3) plus a single-coder qualitative pass (Track 4). The aggregator script `analysis/compile_failure_modes.py` is deterministic — re-running produces identical output. Observation provenance is preserved per-entry."*

- **"Why is the catalog called 'observation_not_improvement'?"**
  *"Because no iteration was performed. The catalog establishes that the MVE evaluation framework can systematically identify failure modes — which is the rescoped contribution. The iteration cycle is future work."*

---

## 12. Coverage map — RQ2.d expected outputs → pipeline phase

| RQ2_expected_outputs.md §4 item | Track 5 phase | Output |
|---|---|---|
| Failure mode catalog (Layer 1 too technical) | 2 | `catalog.TECHNICAL_LAYER_1` |
| Failure mode catalog (MITRE technique not understood) | 2 | `catalog.MITRE_NOT_REFERENCED` |
| Failure mode catalog (DO_NOT actions ignored) | 2 | `catalog.DO_NOT_IGNORED` |
| Failure mode catalog (Role view mismatch) | 2 | `catalog.ROLE_VIEW_MISMATCH` |
| Frequency column | 2 | `n_observations` per category |
| Iteration applied column | 2 | All entries `_status: observed_not_fixed`; manifest's `recommended_iteration` describes what iteration WOULD do |
| Outcome metric column | 2 | `impact_metric` per category |
| Single-round limitation acknowledgement | 2 | `_disclosure.iteration_performed: false` + `_disclosure.evaluation_rounds: 1` |

Every numbered RQ2.d item is traceable to a phase. The original "Iteration applied" column is honestly populated as "none performed" via the `_status: observed_not_fixed` field.

---

## 13. What this track deliberately does NOT do

- **Run any new evaluation.** It only aggregates existing outputs.
- **Make improvement claims.** Every entry's framing is observation.
- **Categorize qualitative themes manually.** The keyword-map heuristic is automated; subjective re-coding is future work.
- **Compute new statistics.** All counts come from already-existing source data.

---

## End of spec

Implementation order: Phase 0 (discovery) → Phase 1 (manifest) → Phase 2 (aggregator) → Phase 3 (renderer) → Phase 4 (tests). Track 5 is **gated only on having at least one upstream track's output** — it degrades gracefully if some sources are missing. The honest expectation: Track 5 is the LAST thing run before paper writing.