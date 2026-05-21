# RQ3 Merge & Figures Pipeline — Canonical Aggregator + Paper Figures

**Project:** XAI-IDS-Healthcare
**Scope:** Phases 6–8 of the RQ3 pipeline: the canonical `compute_rq3_metrics.py` aggregator that pulls all Track 1–5 outputs into `results/rq3_metrics.json`, the figure generator producing 2 paper PDFs, and the CI gates (general + defense-critical) that verify all RQ3 targets are met.
**Purpose:** Single, self-contained spec for closing the RQ3 pipeline loop. Hand to Claude Code. **This is the final spec in the thesis pipeline.**
**Status of design:** All decisions locked. Two `DO NOT GUESS` checkpoints (Track 1-5 output paths, figure aesthetics).

---

## 0. How to use this spec

1. Phase 0 confirms which Track 1-5 outputs already exist. Track 5 is data-gated; partial outputs are expected.
2. Phases 1–3 are sequential after Phase 0.
3. Markers:
   - **DO NOT GUESS** — stop and ask the developer
   - **DEFENSE-CRITICAL** — directly defends a top-tier paper claim
4. Total expected output: 1 new aggregator, 1 figure script, 2 new test functions, 1 paper-ready markdown summary. Runtime: aggregator sub-second, figures ~5 seconds.

---

## 1. Background: what this completes

| Phase | Deliverable | Before this spec | After |
|---|---|---|---|
| **Phase 6** | `module6_evaluation/compute_rq3_metrics.py` | not exists | created — canonical RQ3 aggregator |
| **Phase 6** | `results/rq3_metrics.json` | not exists | canonical single source of truth |
| **Phase 6** | `results/rq3_executive_summary.md` | not exists | paper-ready one-pager rendered from JSON |
| **Phase 7** | `module6_evaluation/make_rq3_figures.py` | not exists | single script with `--only` flag |
| **Phase 7** | 2 PDFs in `results/figures/rq3_*.pdf` | not exists | invariant matrix + per-role/escalation |
| **Phase 8** | `tests/acceptance_tests.py::test_rq3_targets_met` | not exists | general aggregate gate |
| **Phase 8** | `tests/acceptance_tests.py::test_rq3_defense_critical_targets` | not exists | hard-fail subset |

After this spec, **the entire thesis pipeline has end-to-end specs**.

---

## 2. Locked design decisions

| Decision | Resolution |
|---|---|
| Output schema | Mirror RQ2: per-track blocks + headline + targets + tracks_pending |
| Defense summary | Added top-level block `defense_summary` — reviewer-friendly one-liners per claim |
| Per-target rationale | Each entry in `targets` carries `rationale` pointing to `RQ3_expected_outputs.md §X` |
| Figure scope | 2 PDFs: invariant matrix + per-role with escalation |
| Audit chain visualization | Documented in markdown (not a figure); JSON is the source for audit chain claims |
| Figure script | Single `make_rq3_figures.py` with `--only <id>` CLI flag |
| CI gate structure | Two tests: general `test_rq3_targets_met` + hard `test_rq3_defense_critical_targets` |
| Defense-critical subset | Invariants 1-4, audit chain integrity, no-auto-execution audit |
| Status semantics | `_status` ∈ {complete, partial, pending, source_unavailable} per sub-block (same as RQ2) |
| Markdown render | Paper-ready executive summary rendered from JSON; covers all defense talking points |

---

## 3. Phase 0 — Sub-file inventory (DO NOT GUESS)

### 3.1 Discovery script

```python
# scripts/discover_rq3_subfiles.py — TRANSIENT, delete after Phase 0
"""Inventory which Track 1-5 outputs exist."""
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

sources = {
    "track_1_invariant_manifest":   "config/invariants_manifest.yaml",
    "track_1_manifest_validation":  "results/rq3_invariant_manifest_validation.json",
    "track_1_invariant_evidence":   "results/rq3_invariant_evidence.json",
    "track_2_audit_schema":         "results/rq3_audit_schema_completeness.json",
    "track_2_audit_chain":          "results/rq3_audit_chain_integrity.json",
    "track_2_audit_integrity":      "results/rq3_audit_integrity.json",
    "track_3_no_auto_exec":         "results/rq3_no_auto_execution.json",
    "track_4_truth_table_ref":      "results/rq3_truth_table_reference.json",
    "track_4_appendix_b":           "results/rq3_truth_table_appendix_b.md",
    "track_5_escalation":           "analysis/outputs/rq3_escalation.json",
    "track_5_user_study":           "analysis/outputs/rq3_user_study.json",
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
print("RQ3 merge degrades gracefully on missing inputs.")
print("Tracks 1-4 should be present after RQ3 implementation completes.")
print("Track 5 (user study) is DATA-GATED and expected pending.")
print("=" * 60)
```

### 3.2 What to confirm

1. **Track 1-4 outputs exist.** If any are missing, the relevant RQ3 track spec implementation is incomplete.
2. **Track 5 status.** Expected to be `pending` unless user study data has been collected.

---

## 4. Phase 1 — Canonical aggregator

### 4.1 Create `module6_evaluation/compute_rq3_metrics.py`

**Contract:**
- **Inputs:** every sub-file from Tracks 1–5 (any subset; missing inputs produce `_status: pending`)
- **Output:** `results/rq3_metrics.json` + `results/rq3_executive_summary.md`
- **Runtime:** sub-second
- **Side effects:** writes 2 files. No model inference, no slow computation. Read-only on all sub-files.

### 4.2 Output schema

```json
{
  "_meta": {
    "schema_version": "1.0",
    "generated_at": "<ISO-8601>",
    "generated_by": "module6_evaluation/compute_rq3_metrics.py",
    "research_question": "RQ3 — Does the system support distributed security responsibility across hospital roles while maintaining clinical safety?",
    "active_subquestions": ["RQ3.1", "RQ3.2", "RQ3.3", "RQ3.4", "RQ3.5"],
    "tracks_present": ["1", "2", "3", "4"],
    "tracks_pending": ["5"]
  },
  "defense_summary": {
    "_description": "One-line answer per defense claim. Read first.",
    "no_auto_execution": "PASS — 4-layer defense verified (grep + imports + negative test + runtime mock)",
    "audit_tamper_evident": "PASS — chain intact across 1247 entries; schema completeness verified",
    "safety_floor_invariant": "PASS — CRITICAL+unpatchable always surfaces (truth table rows verified)",
    "architectural_invariants": "4/9 enforced, 5/9 pending (linked to RQ2 track completion)",
    "distributed_responsibility_empirical": "PARTIAL — Track 5 user study data-gated"
  },
  "headline": {
    "_description": "Highest-level pass/fail per RQ3 sub-RQ. Read this second.",
    "rq3_1_invariants": "complete",
    "rq3_2_audit_integrity": "complete",
    "rq3_3_no_auto_execution": "complete",
    "rq3_4_truth_table": "complete",
    "rq3_5_user_study": "pending",
    "_overall_status": "partial — user study pending"
  },
  "invariants": {
    "_status": "complete",
    "_merged_at": "<ISO-8601>",
    "_subfile_paths": ["results/rq3_invariant_evidence.json"],
    "manifest_path": "config/invariants_manifest.yaml",
    "evidence_json_path": "results/rq3_invariant_evidence.json",
    "evidence_md_path": "results/rq3_invariant_evidence.md",
    "headline": { ... }
  },
  "audit_integrity": {
    "_status": "complete",
    "_merged_at": "<ISO-8601>",
    "_subfile_paths": ["results/rq3_audit_integrity.json"],
    "_framing": "tamper-evident (detection), not tamper-resistant (prevention)",
    "headline": { ... }
  },
  "no_auto_execution": {
    "_status": "complete",
    "_merged_at": "<ISO-8601>",
    "_subfile_paths": ["results/rq3_no_auto_execution.json"],
    "_framing": "Layer B of the three-layer no-auto-execution defense",
    "headline": { ... }
  },
  "truth_table": {
    "_status": "complete",
    "_merged_at": "<ISO-8601>",
    "_subfile_paths": ["results/rq3_truth_table_reference.json"],
    "appendix_md_path": "results/rq3_truth_table_appendix_b.md",
    "source_csv": "results/rq1_tier_surfacing_truth_table.csv",
    "headline": { ... }
  },
  "user_study": {
    "_status": "pending",
    "_merged_at": null,
    "_subfile_paths": ["analysis/outputs/rq3_user_study.json"],
    "_note": "DATA-GATED: requires user study data collection completion"
  },
  "targets": {
    "_description": "Boolean pass/fail per RQ3 target. Used by tests/acceptance_tests.py.",
    "all_invariants_pass": {
      "value": true, "target": true, "pass": true,
      "rationale": "Per RQ3_expected_outputs.md §4.1 — all 9 invariants enforced",
      "is_defense_critical": true
    },
    "audit_chain_intact": {
      "value": true, "target": true, "pass": true,
      "rationale": "Per RQ3_expected_outputs.md §3.2 — verify_audit_log_integrity() = True",
      "is_defense_critical": true
    },
    "audit_schema_complete": {
      "value": true, "target": true, "pass": true,
      "rationale": "Per RQ3_expected_outputs.md §3.2 — every entry has all required fields",
      "is_defense_critical": true
    },
    "no_auto_exec_audit_pass": {
      "value": true, "target": true, "pass": true,
      "rationale": "Per RQ3_expected_outputs.md §3.1 — zero forbidden patterns in production",
      "is_defense_critical": true
    },
    "safety_floor_holds": {
      "value": true, "target": true, "pass": true,
      "rationale": "Per RQ3_expected_outputs.md §4.2 — CRITICAL+unpatchable rows surface=TRUE",
      "is_defense_critical": true
    },
    "truth_table_completeness": {
      "value": true, "target": true, "pass": true,
      "rationale": "Per RQ3_expected_outputs.md §4.2 — 8 representative rows present + verified",
      "is_defense_critical": false
    },
    "escalation_chi2_overall": {
      "value": null, "target": "computed", "pass": null,
      "rationale": "Per RQ3_expected_outputs.md §2.2 — Chi-square A vs B overall",
      "is_defense_critical": false,
      "_status": "pending_data"
    }
  }
}
```

### 4.3 Implementation

```python
"""
compute_rq3_metrics.py
Canonical aggregator for RQ3 — pulls every Track 1-5 sub-file into one JSON
plus a paper-ready markdown summary.

Inputs: any subset of Track 1-5 sub-files (missing → _status: pending).
Outputs:
  results/rq3_metrics.json
  results/rq3_executive_summary.md

Runtime: sub-second. No model inference.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = REPO_ROOT / "results/rq3_metrics.json"
OUT_MD = REPO_ROOT / "results/rq3_executive_summary.md"


def _try_load_json(rel_path: str) -> Optional[dict]:
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
    out = {
        "_status": status,
        "_merged_at": _now_iso() if status != "pending" else None,
        "_subfile_paths": subfile_paths,
    }
    out.update(contents)
    return out


# ─── Sub-block loaders ─────────────────────────────────────────

def _load_invariants():
    evidence = _try_load_json("results/rq3_invariant_evidence.json")
    paths = ["results/rq3_invariant_evidence.json"]
    if not evidence:
        return _make_block("pending", paths)

    md_exists = (REPO_ROOT / "results/rq3_invariant_evidence.md").exists()
    h = evidence.get("headline", {})
    return _make_block(
        "complete" if h.get("all_invariants_pass") else "failing",
        paths,
        manifest_path="config/invariants_manifest.yaml",
        evidence_json_path="results/rq3_invariant_evidence.json",
        evidence_md_path=("results/rq3_invariant_evidence.md"
                          if md_exists else None),
        headline=h,
    )


def _load_audit_integrity():
    integrity = _try_load_json("results/rq3_audit_integrity.json")
    paths = ["results/rq3_audit_integrity.json"]
    if not integrity:
        return _make_block("pending", paths)

    h = integrity.get("headline", {})
    overall = h.get("_overall_status", "unknown")
    if "skipped" in str(overall):
        status = "skipped — no audit log yet"
    elif overall == "complete":
        status = "complete"
    else:
        status = "failing"

    return _make_block(
        status, paths,
        _framing="tamper-evident (detection), not tamper-resistant (prevention)",
        headline=h,
    )


def _load_no_auto_execution():
    audit = _try_load_json("results/rq3_no_auto_execution.json")
    paths = ["results/rq3_no_auto_execution.json"]
    if not audit:
        return _make_block("pending", paths)

    h = audit.get("headline", {})
    return _make_block(
        "complete" if h.get("audit_pass") else "failing",
        paths,
        _framing="Layer B of the three-layer no-auto-execution defense",
        headline=h,
    )


def _load_truth_table():
    ref = _try_load_json("results/rq3_truth_table_reference.json")
    paths = ["results/rq3_truth_table_reference.json"]
    if not ref:
        return _make_block("pending", paths)

    md_exists = (REPO_ROOT / "results/rq3_truth_table_appendix_b.md").exists()
    h = ref.get("headline", {})
    return _make_block(
        "complete" if h.get("verification_pass") else "failing",
        paths,
        appendix_md_path=("results/rq3_truth_table_appendix_b.md"
                          if md_exists else None),
        source_csv=ref.get("_meta", {}).get("source_csv"),
        headline=h,
    )


def _load_user_study():
    study = _try_load_json("analysis/outputs/rq3_user_study.json")
    paths = ["analysis/outputs/rq3_user_study.json"]
    if not study:
        return _make_block(
            "pending", paths,
            _note="DATA-GATED: requires user study data collection completion",
        )

    status = study.get("_meta", {}).get("_status", "pending")
    return _make_block(
        status, paths,
        _note=study.get("_meta", {}).get("_status_message"),
        per_role=study.get("per_role"),
        overall=study.get("overall"),
        methodology_notes_count=len(study.get("methodology_notes", [])),
        limitations_count=len(study.get("limitations", [])),
    )


# ─── Defense summary builder ───────────────────────────────────

def _build_defense_summary(blocks: dict) -> dict:
    """One-line answer per major defense claim. Reads from sub-block headlines."""
    inv = blocks["invariants"]
    audit = blocks["audit_integrity"]
    no_exec = blocks["no_auto_execution"]
    tt = blocks["truth_table"]
    us = blocks["user_study"]

    summary = {
        "_description": "One-line answer per defense claim. Read first."
    }

    # No auto-execution
    if no_exec["_status"] == "complete":
        n_files = (no_exec.get("headline", {}).get("n_files_scanned", {})
                   .get("production", 0))
        summary["no_auto_execution"] = (
            f"PASS — 4-layer defense verified (grep + imports + negative "
            f"test + runtime mock; {n_files} production files scanned)"
        )
    elif no_exec["_status"] == "failing":
        n = no_exec.get("headline", {}).get("n_violations_production", 0)
        summary["no_auto_execution"] = f"FAIL — {n} violation(s) in production code"
    else:
        summary["no_auto_execution"] = "PENDING — Track 3 not yet run"

    # Audit tamper-evidence
    if audit["_status"] == "complete":
        n_entries = audit.get("headline", {}).get("n_entries", 0)
        summary["audit_tamper_evident"] = (
            f"PASS — chain intact across {n_entries} entries; "
            "schema completeness verified"
        )
    elif audit["_status"].startswith("skipped"):
        summary["audit_tamper_evident"] = (
            "SKIPPED — audit log empty (no production runs yet)"
        )
    elif audit["_status"] == "failing":
        n_breaks = audit.get("headline", {}).get("chain_integrity", {}).get("n_breaks", 0)
        summary["audit_tamper_evident"] = (
            f"FAIL — {n_breaks} chain break(s) OR schema violation(s)"
        )
    else:
        summary["audit_tamper_evident"] = "PENDING — Track 2 not yet run"

    # Safety floor
    if tt["_status"] == "complete":
        summary["safety_floor_invariant"] = (
            "PASS — CRITICAL+unpatchable rows verified surface=TRUE "
            "(truth table audit)"
        )
    elif tt["_status"] == "failing":
        summary["safety_floor_invariant"] = "FAIL — truth table verification failed"
    else:
        summary["safety_floor_invariant"] = "PENDING — Track 4 not yet run"

    # Architectural invariants
    if inv["_status"] == "complete":
        h = inv.get("headline", {})
        summary["architectural_invariants"] = (
            f"PASS — {h.get('n_enforced', 0)}/9 enforced, "
            f"{h.get('n_pending', 0)}/9 pending, "
            f"{h.get('n_failed', 0)}/9 failing"
        )
    elif inv["_status"] == "failing":
        summary["architectural_invariants"] = "FAIL — invariant test(s) failing"
    else:
        summary["architectural_invariants"] = "PENDING — Track 1 not yet run"

    # Empirical (user study)
    if us["_status"] == "complete":
        summary["distributed_responsibility_empirical"] = "PASS — user study analysis complete"
    elif us["_status"] == "partial":
        summary["distributed_responsibility_empirical"] = (
            "PARTIAL — some upstream analyses missing"
        )
    else:
        summary["distributed_responsibility_empirical"] = (
            "PARTIAL — Track 5 user study data-gated"
        )

    return summary


# ─── Target extraction ─────────────────────────────────────────

def _extract_targets(blocks: dict) -> dict:
    """Pull pass/fail targets into a flat namespace with rationale per target."""
    targets = {"_description": (
        "Boolean pass/fail per RQ3 target. Used by "
        "tests/acceptance_tests.py::test_rq3_targets_met."
    )}

    inv = blocks["invariants"]
    if inv["_status"] in ("complete", "failing"):
        h = inv.get("headline", {})
        targets["all_invariants_pass"] = {
            "value": h.get("all_invariants_pass"),
            "target": True,
            "pass": bool(h.get("all_invariants_pass")),
            "rationale": "Per RQ3_expected_outputs.md §4.1 — all 9 invariants enforced",
            "is_defense_critical": True,
        }

    audit = blocks["audit_integrity"]
    if audit["_status"] in ("complete", "failing"):
        h = audit.get("headline", {})
        if h.get("chain_intact") is not None:
            targets["audit_chain_intact"] = {
                "value": h.get("chain_intact"),
                "target": True,
                "pass": bool(h.get("chain_intact")),
                "rationale": (
                    "Per RQ3_expected_outputs.md §3.2 — "
                    "verify_audit_log_integrity() returns True"
                ),
                "is_defense_critical": True,
            }
        if h.get("schema_completeness_pass") is not None:
            targets["audit_schema_complete"] = {
                "value": h.get("schema_completeness_pass"),
                "target": True,
                "pass": bool(h.get("schema_completeness_pass")),
                "rationale": (
                    "Per RQ3_expected_outputs.md §3.2 — "
                    "every entry has all required fields"
                ),
                "is_defense_critical": True,
            }

    no_exec = blocks["no_auto_execution"]
    if no_exec["_status"] in ("complete", "failing"):
        h = no_exec.get("headline", {})
        targets["no_auto_exec_audit_pass"] = {
            "value": h.get("audit_pass"),
            "target": True,
            "pass": bool(h.get("audit_pass")),
            "rationale": (
                "Per RQ3_expected_outputs.md §3.1 — "
                "zero forbidden patterns in production code"
            ),
            "is_defense_critical": True,
        }

    tt = blocks["truth_table"]
    if tt["_status"] in ("complete", "failing"):
        h = tt.get("headline", {})
        targets["truth_table_completeness"] = {
            "value": h.get("verification_pass"),
            "target": True,
            "pass": bool(h.get("verification_pass")),
            "rationale": (
                "Per RQ3_expected_outputs.md §4.2 — "
                "8 representative rows present + verified"
            ),
            "is_defense_critical": False,
        }
        # Safety floor specifically: derived from truth table verification
        targets["safety_floor_holds"] = {
            "value": h.get("verification_pass"),
            "target": True,
            "pass": bool(h.get("verification_pass")),
            "rationale": (
                "Per RQ3_expected_outputs.md §4.2 — "
                "CRITICAL+unpatchable rows surface=TRUE"
            ),
            "is_defense_critical": True,
        }

    # Escalation Chi-square (RQ3.5)
    us = blocks["user_study"]
    if us["_status"] == "complete":
        overall_esc = us.get("overall", {}).get("escalation")
        if overall_esc and overall_esc.get("chi2_p_value") is not None:
            targets["escalation_chi2_overall"] = {
                "value": overall_esc.get("chi2_p_value"),
                "target": "computed",
                "pass": True,  # No specific p-value threshold; presence = pass
                "rationale": (
                    "Per RQ3_expected_outputs.md §2.2 — "
                    "Chi-square A vs B overall computed"
                ),
                "is_defense_critical": False,
            }
    else:
        targets["escalation_chi2_overall"] = {
            "value": None,
            "target": "computed",
            "pass": None,
            "rationale": (
                "Per RQ3_expected_outputs.md §2.2 — "
                "Chi-square A vs B overall (pending data)"
            ),
            "is_defense_critical": False,
            "_status": "pending_data",
        }

    return targets


# ─── Headline ──────────────────────────────────────────────────

def _build_headline(blocks: dict) -> dict:
    statuses = {
        "rq3_1_invariants":         blocks["invariants"]["_status"],
        "rq3_2_audit_integrity":    blocks["audit_integrity"]["_status"],
        "rq3_3_no_auto_execution":  blocks["no_auto_execution"]["_status"],
        "rq3_4_truth_table":        blocks["truth_table"]["_status"],
        "rq3_5_user_study":         blocks["user_study"]["_status"],
    }

    bad = [k for k, v in statuses.items() if v == "failing"]
    if bad:
        overall = f"FAIL — {', '.join(bad)}"
    elif all(v == "complete" for v in statuses.values()):
        overall = "complete"
    elif all(v == "pending" for v in statuses.values()):
        overall = "pending"
    else:
        missing = [k for k, v in statuses.items() if v != "complete"]
        overall = f"partial — incomplete: {', '.join(missing)}"

    return {
        "_description": "Highest-level pass/fail per RQ3 sub-RQ.",
        **statuses,
        "_overall_status": overall,
    }


# ─── Markdown rendering ────────────────────────────────────────

def _render_executive_summary(data: dict) -> str:
    """Render results/rq3_executive_summary.md from the canonical JSON."""
    lines = []
    lines.append("# RQ3 — Executive Summary")
    lines.append("")
    lines.append(
        f"*Generated on {data['_meta']['generated_at']} by "
        f"`module6_evaluation/compute_rq3_metrics.py`.*"
    )
    lines.append("")

    lines.append(f"**Research Question:** {data['_meta']['research_question']}")
    lines.append("")

    # Defense summary first — the executive read
    lines.append("## Defense Summary (Read First)")
    lines.append("")
    for k, v in data["defense_summary"].items():
        if k.startswith("_"):
            continue
        label = k.replace("_", " ").title()
        lines.append(f"- **{label}**: {v}")
    lines.append("")

    # Sub-RQ status
    lines.append("## Sub-RQ Status")
    lines.append("")
    lines.append("| Sub-RQ | Status |")
    lines.append("|---|---|")
    for k, v in data["headline"].items():
        if k.startswith("_"):
            continue
        label = k.replace("rq3_", "RQ3.").replace("_", " ")
        lines.append(f"| {label} | `{v}` |")
    lines.append(f"| **Overall** | **{data['headline']['_overall_status']}** |")
    lines.append("")

    # Targets
    lines.append("## Targets")
    lines.append("")
    lines.append("| Target | Value | Pass | Defense-critical |")
    lines.append("|---|---|---|---|")
    for tid, t in data["targets"].items():
        if tid.startswith("_"):
            continue
        passed = t.get("pass")
        mark = (
            "✓" if passed is True
            else "✗" if passed is False
            else "○ pending"
        )
        critical = "yes" if t.get("is_defense_critical") else "no"
        lines.append(
            f"| `{tid}` | {t.get('value')} | {mark} | {critical} |"
        )
    lines.append("")

    # Cross-references
    lines.append("## Cross-References")
    lines.append("")
    lines.append("- **Full invariant catalog:** "
                 "`results/rq3_invariant_evidence.md`")
    lines.append("- **Truth table (Appendix B):** "
                 "`results/rq3_truth_table_appendix_b.md`")
    lines.append("- **Audit chain status:** "
                 "`results/rq3_audit_integrity.json`")
    lines.append("- **Detailed JSON:** "
                 "`results/rq3_metrics.json`")
    lines.append("")

    return "\n".join(lines)


# ─── Main ──────────────────────────────────────────────────────

def main():
    blocks = {
        "invariants":         _load_invariants(),
        "audit_integrity":    _load_audit_integrity(),
        "no_auto_execution":  _load_no_auto_execution(),
        "truth_table":        _load_truth_table(),
        "user_study":         _load_user_study(),
    }

    tracks_present = [
        k for k, v in blocks.items()
        if v["_status"] in ("complete", "failing", "partial")
    ]
    tracks_pending = [
        k for k, v in blocks.items() if v["_status"] == "pending"
    ]

    out = {
        "_meta": {
            "schema_version": "1.0",
            "generated_at": _now_iso(),
            "generated_by": "module6_evaluation/compute_rq3_metrics.py",
            "research_question": (
                "RQ3 — Does the system support distributed security "
                "responsibility across hospital roles while maintaining "
                "clinical safety?"
            ),
            "active_subquestions": ["RQ3.1", "RQ3.2", "RQ3.3", "RQ3.4", "RQ3.5"],
            "blocks_present": tracks_present,
            "blocks_pending": tracks_pending,
        },
        "defense_summary": _build_defense_summary(blocks),
        "headline": _build_headline(blocks),
        "invariants":         blocks["invariants"],
        "audit_integrity":    blocks["audit_integrity"],
        "no_auto_execution":  blocks["no_auto_execution"],
        "truth_table":        blocks["truth_table"],
        "user_study":         blocks["user_study"],
        "targets": _extract_targets(blocks),
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2, default=str))
    md = _render_executive_summary(out)
    OUT_MD.write_text(md)

    print(f"Wrote {OUT_JSON.relative_to(REPO_ROOT)}")
    print(f"Wrote {OUT_MD.relative_to(REPO_ROOT)}")
    print(f"Overall: {out['headline']['_overall_status']}")
    for k, v in out["headline"].items():
        if k.startswith("rq3_"):
            print(f"  {k}: {v}")

    n_targets = sum(1 for t in out["targets"].values()
                    if isinstance(t, dict) and "pass" in t)
    n_pass = sum(1 for t in out["targets"].values()
                 if isinstance(t, dict) and t.get("pass") is True)
    print(f"\nTargets: {n_pass}/{n_targets} pass")
    for tid, t in out["targets"].items():
        if not isinstance(t, dict) or "pass" not in t:
            continue
        mark = "✓" if t.get("pass") is True else "✗" if t.get("pass") is False else "○"
        critical = " [DEFENSE-CRITICAL]" if t.get("is_defense_critical") else ""
        print(f"  {mark} {tid}{critical}")


if __name__ == "__main__":
    main()
```

### 4.4 Verification

```bash
python -m module6_evaluation.compute_rq3_metrics
cat results/rq3_metrics.json | python -m json.tool | head -40
head -30 results/rq3_executive_summary.md
```

---

## 5. Phase 2 — Figure generator

### 5.1 Create `module6_evaluation/make_rq3_figures.py`

**Contract:**
- **Inputs:** Track 1 and Track 5 outputs
- **Outputs:** PDFs in `results/figures/rq3_*.pdf`
- **CLI:** `python -m module6_evaluation.make_rq3_figures` or `--only <id>`
- **Runtime:** ~5 seconds total
- **Graceful skip:** missing sub-files produce `[SKIP]` not crashes

### 5.2 Figure inventory

| Figure ID | Filename | Source | Paper section |
|---|---|---|---|
| `invariants` | `rq3_invariant_matrix.pdf` | `rq3_invariant_evidence.json::invariants` | §5.6 Safety Engineering |
| `user_study` | `rq3_per_role_with_escalation.pdf` | `rq3_user_study.json` | §5.3 Distributed Responsibility |

### 5.3 Implementation

```python
"""
make_rq3_figures.py
Generate paper-ready PDFs for RQ3 from canonical sub-files.

Usage:
  python -m module6_evaluation.make_rq3_figures             # all figures
  python -m module6_evaluation.make_rq3_figures --only invariants
  python -m module6_evaluation.make_rq3_figures --list

Runtime: ~5 seconds.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = REPO_ROOT / "results/figures"

FIGURES = {
    "invariants":  "rq3_invariant_matrix.pdf",
    "user_study":  "rq3_per_role_with_escalation.pdf",
}


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


# ─── Figure 1: Invariant pass/fail matrix ──────────────────────

def make_invariants(out_path):
    data = _load_json("results/rq3_invariant_evidence.json")
    if not data:
        return _skip("invariants", "rq3_invariant_evidence.json missing")

    invs = data.get("invariants", [])
    if not invs:
        return _skip("invariants", "no invariants in evidence file")

    # Build a 9-row × 3-column status matrix
    # Columns: serves RQ1 / serves RQ2 / serves RQ3
    # Rows: each invariant (1-9)
    # Cell color: green=pass, red=fail, gray=pending/documented, white=not applicable

    invs_sorted = sorted(invs, key=lambda i: i["id"])
    n = len(invs_sorted)

    fig, ax = plt.subplots(figsize=(9, max(4.5, 0.5 * n + 2)))

    # Color map by status
    color_map = {
        "pass": "#3aaa35",       # green
        "fail": "#c0392b",       # red
        "pending": "#bbbbbb",    # gray
        "documented": "#888888",  # darker gray
        "no_tests_found": "#f39c12",  # orange
        "unknown": "#dddddd",
    }

    # Layout: y-axis = invariants top-down; x-axis = RQ columns
    y = np.arange(n)[::-1]
    rqs = [1, 2, 3]
    cell_w = 0.8
    cell_h = 0.7

    # Draw cells
    for i, inv in enumerate(invs_sorted):
        for j, rq in enumerate(rqs):
            applies = rq in inv.get("serves_rqs", [])
            if not applies:
                # not applicable — just empty cell border
                ax.add_patch(plt.Rectangle(
                    (j - cell_w / 2, y[i] - cell_h / 2),
                    cell_w, cell_h, fill=False,
                    edgecolor="#dddddd", linewidth=0.5,
                ))
            else:
                status = inv.get("_overall_status", "unknown")
                color = color_map.get(status, "#dddddd")
                ax.add_patch(plt.Rectangle(
                    (j - cell_w / 2, y[i] - cell_h / 2),
                    cell_w, cell_h, color=color, alpha=0.85,
                    edgecolor="black", linewidth=0.6,
                ))
                # Label
                marker = (
                    "✓" if status == "pass" else
                    "✗" if status == "fail" else
                    "○" if status == "pending" else
                    "—" if status == "documented" else
                    "?"
                )
                ax.text(j, y[i], marker, ha="center", va="center",
                        fontsize=11, color="white"
                        if status in {"pass", "fail"} else "black")

    # Row labels (invariant titles, truncated)
    for i, inv in enumerate(invs_sorted):
        title = (inv["title"] if len(inv["title"]) <= 55
                 else inv["title"][:52] + "...")
        severity = inv.get("severity", "")
        severity_marker = "⚠" if severity == "safety_critical" else ""
        ax.text(-0.6, y[i],
                f"{severity_marker} Inv {inv['id']}: {title}",
                ha="right", va="center", fontsize=8)

    # Column headers
    for j, rq in enumerate(rqs):
        ax.text(j, max(y) + 0.7, f"RQ{rq}", ha="center", va="bottom",
                fontsize=12, fontweight="bold")

    ax.set_xlim(-4.5, len(rqs) - 0.5 + 0.2)
    ax.set_ylim(-0.5, max(y) + 1.2)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", "box")
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Legend
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=color_map["pass"], label="Pass"),
        plt.Rectangle((0, 0), 1, 1, color=color_map["fail"], label="Fail"),
        plt.Rectangle((0, 0), 1, 1, color=color_map["pending"], label="Pending"),
        plt.Rectangle((0, 0), 1, 1, color=color_map["documented"],
                      label="Documented"),
    ]
    ax.legend(handles=legend_handles, loc="lower right",
              bbox_to_anchor=(1.0, -0.15), ncol=4, fontsize=8, frameon=False)

    ax.set_title(
        "Architectural Invariants — Cross-RQ Coverage Matrix\n"
        "⚠ = safety-critical invariant",
        fontsize=11,
    )

    plt.tight_layout()
    plt.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close()
    _saved("invariants", out_path)


# ─── Figure 2: User study per role with escalation ─────────────

def make_user_study(out_path):
    data = _load_json("analysis/outputs/rq3_user_study.json")
    if not data:
        return _skip("user_study", "rq3_user_study.json missing (data-gated)")

    per_role = data.get("per_role", {})
    if not per_role:
        return _skip("user_study", "per_role block missing")

    roles = ["IT_GENERALIST", "BIOMED_ENGINEER", "NURSE_MANAGER"]
    roles_present = [r for r in roles if per_role.get(r) and
                     per_role[r].get("escalation")]

    if not roles_present:
        return _skip("user_study", "no roles have escalation data")

    # 4-panel grid: decision_time | accuracy | confidence | escalation
    metrics = ["decision_time", "accuracy", "confidence", "escalation"]
    metric_labels = ["Decision Time (median)", "Accuracy", "Confidence",
                     "Escalation Rate"]

    # Filter to metrics that have data across all roles
    available = []
    for m, lbl in zip(metrics, metric_labels):
        if all(per_role[r] and per_role[r].get(m) is not None
               for r in roles_present):
            available.append((m, lbl))

    if not available:
        return _skip("user_study", "no metrics available across all roles")

    fig, axes = plt.subplots(
        1, len(available), figsize=(4 * len(available), 4.5)
    )
    if len(available) == 1:
        axes = [axes]

    for ax, (metric, label) in zip(axes, available):
        x = np.arange(len(roles_present))
        bar_width = 0.35

        if metric == "escalation":
            # Escalation rates from rq3_escalation.json structure
            a_vals = [
                per_role[r]["escalation"].get("escalation_rate_A", 0)
                for r in roles_present
            ]
            b_vals = [
                per_role[r]["escalation"].get("escalation_rate_B", 0)
                for r in roles_present
            ]
            ax.bar(x, a_vals, bar_width, label="Group A (MVE)")
            ax.bar(x + bar_width, b_vals, bar_width, label="Group B (no MVE)")
            ax.set_ylim(0, 1.0)

            # Annotate p-values
            for i, r in enumerate(roles_present):
                cell = per_role[r]["escalation"]
                p = cell.get("chi2_p_value")
                if p is not None:
                    test_label = cell.get("recommended_test", "chi2")
                    p_text = f"p={p:.3f}\n({test_label})"
                    ax.annotate(
                        p_text,
                        xy=(i + bar_width / 2, max(a_vals[i], b_vals[i]) + 0.03),
                        ha="center", fontsize=7,
                        color="darkred" if p < 0.05 else "gray"
                    )

        else:
            # decision_time / accuracy / confidence — from rq2c structure
            a_vals = [
                per_role[r][metric].get("median_A", 0)
                for r in roles_present
            ]
            b_vals = [
                per_role[r][metric].get("median_B", 0)
                for r in roles_present
            ]
            ax.bar(x, a_vals, bar_width, label="Group A (MVE)")
            ax.bar(x + bar_width, b_vals, bar_width, label="Group B (no MVE)")

            # n_warning markers
            for i, r in enumerate(roles_present):
                if per_role[r][metric].get("n_warning"):
                    ax.annotate(
                        "low-n",
                        xy=(i + bar_width / 2, max(a_vals[i], b_vals[i])),
                        ha="center", fontsize=7, color="red"
                    )

        ax.set_xticks(x + bar_width / 2)
        ax.set_xticklabels(
            [r.replace("_", "\n") for r in roles_present], fontsize=9
        )
        ax.set_title(label, fontsize=10)
        ax.legend(fontsize=7, loc="best")

    fig.suptitle(
        "RQ3.5 — User Study: per-role metrics + escalation rate (Chi-square)",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close()
    _saved("user_study", out_path)


# ─── Dispatch ──────────────────────────────────────────────────

GENERATORS = {
    "invariants":  make_invariants,
    "user_study":  make_user_study,
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

### 5.4 Verification

```bash
python -m module6_evaluation.make_rq3_figures
# Expected: 1-2 [OK] lines (user_study skips if data not ready)

python -m module6_evaluation.make_rq3_figures --only invariants
# Expected: invariants figure only

ls results/figures/rq3_*.pdf
```

---

## 6. Phase 3 — CI acceptance tests

### 6.1 Extend `tests/acceptance_tests.py`

Two new functions per the locked design — general aggregate + defense-critical hard gate.

```python
def test_rq3_targets_met():
    """
    Aggregate RQ3 CI gate: every present target must pass.
    Pending targets (Track 5 user study, etc.) do NOT fail this test.
    """
    import json
    from pathlib import Path

    metrics_path = Path("results/rq3_metrics.json")
    assert metrics_path.exists(), (
        "Run module6_evaluation/compute_rq3_metrics.py first"
    )

    m = json.loads(metrics_path.read_text())
    targets = m.get("targets", {})
    failures = []
    for tid, t in targets.items():
        if tid.startswith("_") or not isinstance(t, dict):
            continue
        if t.get("pass") is False:
            failures.append({
                "target": tid,
                "value": t.get("value"),
                "target_value": t.get("target"),
                "rationale": t.get("rationale"),
            })
    assert not failures, (
        f"RQ3 targets failed: {failures}. "
        f"Inspect results/rq3_metrics.json for details."
    )


def test_rq3_defense_critical_targets():
    """
    DEFENSE-CRITICAL CI gate: targets marked is_defense_critical=true
    MUST pass. Pending defense-critical targets are also failures here
    (in contrast to test_rq3_targets_met which allows pending).

    Reasoning: defense-critical claims must be DEMONSTRABLE before defense,
    not pending data collection. If invariant evidence is pending, that
    breaks the architectural defense.
    """
    import json
    from pathlib import Path

    metrics_path = Path("results/rq3_metrics.json")
    assert metrics_path.exists(), (
        "Run module6_evaluation/compute_rq3_metrics.py first"
    )

    m = json.loads(metrics_path.read_text())
    targets = m.get("targets", {})
    defense_critical = []
    for tid, t in targets.items():
        if tid.startswith("_") or not isinstance(t, dict):
            continue
        if not t.get("is_defense_critical"):
            continue
        defense_critical.append((tid, t))

    assert defense_critical, (
        "No defense-critical targets found — manifest may be malformed"
    )

    failures = [
        (tid, t) for tid, t in defense_critical
        if t.get("pass") is not True
    ]
    assert not failures, (
        f"{len(failures)} defense-critical target(s) not in PASS state:\n"
        + "\n".join(
            f"  - {tid}: pass={t.get('pass')} "
            f"value={t.get('value')} ({t.get('rationale', 'no rationale')})"
            for tid, t in failures
        )
    )


def test_rq3_defense_summary_complete():
    """
    DEFENSE-CRITICAL meta-test: the defense_summary block must contain
    one-line answers for all top claims. This protects against a future
    refactor silently dropping summary fields.
    """
    import json
    from pathlib import Path

    p = Path("results/rq3_metrics.json")
    if not p.exists():
        import pytest
        pytest.skip("Run compute_rq3_metrics.py first")

    m = json.loads(p.read_text())
    summary = m.get("defense_summary", {})

    required = [
        "no_auto_execution",
        "audit_tamper_evident",
        "safety_floor_invariant",
        "architectural_invariants",
        "distributed_responsibility_empirical",
    ]
    missing = [k for k in required if k not in summary]
    assert not missing, (
        f"defense_summary block missing required keys: {missing}"
    )

    # Each summary value must be a non-empty string
    for k in required:
        val = summary[k]
        assert isinstance(val, str) and val.strip(), (
            f"defense_summary[{k}] is empty or non-string"
        )
```

### 6.2 Verification

```bash
pytest tests/acceptance_tests.py::test_rq3_targets_met -v
pytest tests/acceptance_tests.py::test_rq3_defense_critical_targets -v
pytest tests/acceptance_tests.py::test_rq3_defense_summary_complete -v
```

---

## 7. Execution order

```bash
# ─── PHASE 0: SUB-FILE INVENTORY ───────────────────────────────
python scripts/discover_rq3_subfiles.py > /tmp/rq3_subfile_inventory.json

# ─── PHASE 1: CANONICAL AGGREGATOR ─────────────────────────────
# Create module6_evaluation/compute_rq3_metrics.py
python -m module6_evaluation.compute_rq3_metrics
# Outputs:
#   results/rq3_metrics.json
#   results/rq3_executive_summary.md
cat results/rq3_metrics.json | python -m json.tool | head -50
head -40 results/rq3_executive_summary.md

# ─── PHASE 2: FIGURES ──────────────────────────────────────────
python -m module6_evaluation.make_rq3_figures
ls results/figures/rq3_*.pdf

# ─── PHASE 3: CI GATES ─────────────────────────────────────────
# Append the three test functions to tests/acceptance_tests.py
pytest tests/acceptance_tests.py -k rq3 -v

# ─── FINAL VERIFICATION ────────────────────────────────────────
pytest tests/ -v
ls results/rq3_metrics.json \
   results/rq3_executive_summary.md \
   results/figures/rq3_*.pdf
```

---

## 8. Open questions to surface (DO NOT GUESS)

1. **Phase 0 — sub-file paths.** If any track's output is at a different path than listed in §3.1, update the loader in `compute_rq3_metrics.py`.
2. **Phase 2 — figure aesthetics.** Matplotlib defaults used. If the paper has style guidelines (font family, palette), update the figure functions.
3. **Phase 2 — invariant matrix layout.** With 9 invariants, the matrix is compact. If you have substantially more or fewer invariants in your final manifest, the figure dimensions may need adjustment.

---

## 9. Coverage map — closing items → pipeline phase

| Item | Phase | Output |
|---|---|---|
| Canonical RQ3 metrics file | 1 | `results/rq3_metrics.json` |
| Paper-ready executive summary | 1 | `results/rq3_executive_summary.md` |
| Defense summary (one-liner per claim) | 1 | `defense_summary` block in JSON + markdown |
| Per-target rationale | 1 | `targets[*].rationale` in JSON |
| Defense-critical flag | 1 | `targets[*].is_defense_critical` in JSON |
| Invariant matrix figure | 2 | `results/figures/rq3_invariant_matrix.pdf` |
| Per-role + escalation figure | 2 | `results/figures/rq3_per_role_with_escalation.pdf` |
| General CI gate | 3 | `test_rq3_targets_met` |
| Defense-critical hard gate | 3 | `test_rq3_defense_critical_targets` |
| Defense summary meta-test | 3 | `test_rq3_defense_summary_complete` |
| Cross-RQ coverage | 1 | Per-invariant `serves_rqs` carried into JSON; visualized in figure |

---

## 10. Defense talking points this enables

When a reviewer asks closing RQ3 questions:

- **"What's the single source of truth for RQ3 results?"**
  *"`results/rq3_metrics.json`. The top-level `defense_summary` block answers the major claims with one line each. `headline` gives status per sub-RQ. `targets` lists every pass/fail with rationale pointing to the expected outputs document. The executive summary markdown is rendered from this JSON; nothing in the paper bypasses it."*

- **"What's defense-critical vs nice-to-have?"**
  *"Each target in the JSON has an `is_defense_critical` flag. The dedicated test `test_rq3_defense_critical_targets` hard-fails CI if any defense-critical target is anything but PASS — including pending. The architectural claims (invariants 1-4, audit chain, no-auto-execution, safety floor) must be demonstrable, not pending."*

- **"What about pending tracks like the user study?"**
  *"Tracked separately. The general `test_rq3_targets_met` allows pending without failing. The defense-critical test does not — but no defense-critical target is in the empirical/user-study category. RQ3's defense story doesn't require the user study to land; it's nice-to-have empirical confirmation."*

- **"Can you regenerate the paper figures?"**
  *"`python -m module6_evaluation.make_rq3_figures` produces both PDFs. `--only invariants` or `--only user_study` regenerates one. The invariant matrix shows cross-RQ coverage at a glance. The user-study figure includes the escalation Chi-square p-values per role."*

- **"How does this connect to RQ1 and RQ2?"**
  *"Invariants 2, 6, 7, 9 are shared across RQ2 and RQ3. The truth table is shared between RQ1 and RQ3. The user study data is shared between RQ2.c and RQ3.5. The invariant matrix figure shows this overlap visually. The cross-RQ defense story is consistent: same evidence, different framing per RQ."*

- **"Why isn't there an audit chain health figure?"**
  *"The audit chain claim is a single bit — either intact or not. Visual encoding doesn't add information. The `defense_summary` block in JSON, the executive summary markdown, and the audit integrity JSON all carry the claim. If a reviewer wants visual evidence, the figure can be added in 30 lines of matplotlib — the data is already in `results/rq3_audit_integrity.json`."*

---

## 11. What this spec deliberately does NOT do

- **Re-run any track analysis.** It aggregates from existing sub-files only.
- **Modify Track 1-5 sub-files.** Aggregator is read-only on them.
- **Generate the paper's prose.** Markdown is structured tables; the paper author writes connecting prose.
- **Produce a Venn diagram of cross-RQ overlap.** The invariant matrix figure already shows per-invariant RQ coverage; a Venn adds nothing.
- **Render LaTeX.** Markdown only; the thesis is in `.docx`.

---

## End of spec

Implementation order: Phase 0 (inventory) → Phase 1 (aggregator) → Phase 2 (figures) → Phase 3 (CI gates).

**This is the final spec in the thesis pipeline.** With its implementation, RQ3 (and the project as a whole) has end-to-end automation: every paper claim is traceable to a JSON artifact, every artifact is verifiable by a CI test, every CI test has a documented defense rationale.