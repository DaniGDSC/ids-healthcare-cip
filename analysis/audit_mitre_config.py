"""RQ2.e — MITRE config audit (RQ2_Mitre.md Phase 1).

Audits ``configs/attack_to_mitre_mapping.yaml`` for structural
completeness and emits ``results/rq2_mitre_audit.json``.

Schema (Phase 0 discovery, RQ2_Mitre.md §3):
  * Pattern D (not in the spec's A/B/C): ``mappings`` is a LIST of
    dicts, each with ``attack_category``, ``mitre_techniques`` (list),
    and per-entry ``last_validated``.
  * Top-level ``mitre_framework_version`` present.
  * No top-level ``last_validated``.
  * The benign sentinel ``attack_category: "normal"`` is intentionally
    mapped with ``mitre_techniques: []`` and must be excluded from A4 /
    A5 (treating it as a missing mapping or a structural defect would
    misrepresent intent).

Checks (RQ2_Mitre.md §4.2):
  A1 YAML parses                                                  (FAIL)
  A2 mitre_framework_version present + non-empty                  (FAIL)
  A3 last_validated present (top-level OR per-entry)              (FAIL)
  A4 every in-data attack_category has a mapping                  (FAIL)
  A5 every non-"normal" mapped entry has >= 1 technique           (FAIL)
  A6 T-ID matches ^T\\d{4}(\\.\\d{3})?$                            (WARN)
  A7 every technique has a human name                             (WARN)
  A8 confidence in {HIGH, MEDIUM, LOW}                            (WARN)
  A9 mappings exist but unused in data                            (INFO)

``headline.audit_pass`` is True iff zero FAIL findings.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
YAML_PATH = REPO_ROOT / "configs/attack_to_mitre_mapping.yaml"
NPZ_PATH = REPO_ROOT / "results/reports/risk_scores.npz"
OUT = REPO_ROOT / "results/rq2_mitre_audit.json"

TID_RE = re.compile(r"^T\d{4}(\.\d{3})?$")
VALID_CONFIDENCE = {"HIGH", "MEDIUM", "LOW"}
BENIGN_SENTINEL = "normal"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_yaml() -> tuple[dict | None, dict]:
    try:
        doc = yaml.safe_load(YAML_PATH.read_text(encoding="utf-8"))
        return doc, {
            "check_id": "A1", "severity": "PASS",
            "description": "YAML parsed successfully",
            "details": None,
        }
    except Exception as e:  # noqa: BLE001
        return None, {
            "check_id": "A1", "severity": "FAIL",
            "description": "YAML failed to parse",
            "details": {"error": str(e)},
        }


def _mappings_list(doc: dict) -> list[dict]:
    """Pattern D: top-level ``mappings:`` is a list of entry dicts."""
    block = doc.get("mappings", [])
    if not isinstance(block, list):
        raise TypeError(
            "Expected 'mappings' to be a list per Pattern D "
            f"(see RQ2_Mitre.md §3); got {type(block).__name__}"
        )
    return block


def _extract_techniques(entry: dict) -> list[tuple[str | None, str | None, str | None]]:
    """Pattern D: each entry has ``mitre_techniques: [{id, name, confidence}, ...]``."""
    techs = entry.get("mitre_techniques") or []
    return [
        (t.get("id"), t.get("name"), t.get("confidence"))
        for t in techs
        if isinstance(t, dict)
    ]


def _attack_categories_in_data() -> set[str]:
    """Non-benign attack categories present in test data (per npz v1.1)."""
    data = np.load(NPZ_PATH, allow_pickle=False)
    cats = {str(c) for c in np.unique(data["attack_category"])}
    return cats - {BENIGN_SENTINEL, ""}


def _finalize(out: dict, findings: list[dict]) -> None:
    out["findings"] = findings
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {OUT.relative_to(REPO_ROOT)}")
    h = out.get("headline", {})
    print(
        f"Audit: {'PASS' if h.get('audit_pass') else 'FAIL'} "
        f"(fail={h.get('n_fail', '?')}, warn={h.get('n_warn', '?')}, "
        f"info={h.get('n_info', '?')})"
    )


def main() -> None:
    findings: list[dict] = []
    out: dict[str, Any] = {
        "_meta": {
            "schema_version": "1.0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "generated_by": "analysis/audit_mitre_config.py",
            "inputs": {
                "yaml_path": str(YAML_PATH.relative_to(REPO_ROOT)),
                "yaml_sha256": (
                    _sha256(YAML_PATH) if YAML_PATH.exists() else None
                ),
                "risk_scores_npz": str(NPZ_PATH.relative_to(REPO_ROOT)),
                "risk_scores_sha256": (
                    _sha256(NPZ_PATH) if NPZ_PATH.exists() else None
                ),
            },
            "config": {
                "required_top_level_fields": [
                    "mitre_framework_version", "last_validated_anywhere",
                ],
                "tid_regex": TID_RE.pattern,
                "valid_confidence_levels": sorted(VALID_CONFIDENCE),
                "benign_sentinel": BENIGN_SENTINEL,
                "phase0_pattern": "D — mappings is a list of entry dicts",
            },
        },
        "headline": {},
        "findings": [],
        "mappings_summary": {},
    }

    # ── A1 — parse ─────────────────────────────────────────────────
    doc, a1 = _load_yaml()
    findings.append(a1)
    if doc is None:
        out["headline"] = {"audit_pass": False, "n_fail": 1, "n_warn": 0, "n_info": 0}
        _finalize(out, findings)
        return

    # ── A2 — framework version ─────────────────────────────────────
    framework_version = doc.get("mitre_framework_version")
    findings.append({
        "check_id": "A2",
        "severity": "PASS" if framework_version else "FAIL",
        "description": "mitre_framework_version present and non-empty",
        "details": {"value": framework_version},
    })

    # Walk the mappings list defensively.
    try:
        mappings = _mappings_list(doc)
    except TypeError as e:
        findings.append({
            "check_id": "A1", "severity": "FAIL",
            "description": "mappings block structural error",
            "details": {"error": str(e)},
        })
        out["headline"] = {
            "audit_pass": False, "n_fail": 2, "n_warn": 0, "n_info": 0,
        }
        _finalize(out, findings)
        return

    # Build category-keyed view so subsequent checks are O(1).
    mappings_by_cat: dict[str, dict] = {}
    duplicate_cats: list[str] = []
    for entry in mappings:
        cat = entry.get("attack_category") if isinstance(entry, dict) else None
        if cat is None:
            findings.append({
                "check_id": "A1", "severity": "FAIL",
                "description": "Mapping entry missing 'attack_category' key",
                "details": {"entry": entry},
            })
            continue
        if cat in mappings_by_cat:
            duplicate_cats.append(cat)
        mappings_by_cat[cat] = entry
    if duplicate_cats:
        findings.append({
            "check_id": "A1", "severity": "FAIL",
            "description": "Duplicate attack_category keys in mappings list",
            "details": {"duplicates": duplicate_cats},
        })

    # ── A3 — last_validated present (top-level OR per-entry) ───────
    top_lv = doc.get("last_validated")
    per_entry_lvs = {
        cat: e.get("last_validated")
        for cat, e in mappings_by_cat.items()
        if e.get("last_validated")
    }
    findings.append({
        "check_id": "A3",
        "severity": "PASS" if (top_lv or per_entry_lvs) else "FAIL",
        "description": "last_validated present (top-level or per-entry)",
        "details": {
            "top_level": top_lv,
            "per_entry_count": len(per_entry_lvs),
            "per_entry_keys": sorted(per_entry_lvs.keys()),
        },
    })

    # ── A4 + A9 — orphan / unused mapping check ───────────────────
    # Benign sentinel excluded from both sides per Phase 0 decision D3.
    in_data = _attack_categories_in_data()
    in_yaml = set(mappings_by_cat.keys()) - {BENIGN_SENTINEL}
    orphans = sorted(in_data - in_yaml)
    unused = sorted(in_yaml - in_data)
    findings.append({
        "check_id": "A4",
        "severity": "FAIL" if orphans else "PASS",
        "description": "Every in-data attack_category has a mapping",
        "details": {
            "orphans": orphans,
            "categories_in_data": sorted(in_data),
            "categories_in_yaml": sorted(in_yaml),
        },
    })
    if unused:
        findings.append({
            "check_id": "A9",
            "severity": "INFO",
            "description": "Mappings exist but no test-split data uses them",
            "details": {"unused_mappings": unused},
        })

    # ── A5–A8 — per-entry technique validation ────────────────────
    mappings_summary: dict[str, dict] = {}
    for category, entry in mappings_by_cat.items():
        try:
            techniques = _extract_techniques(entry)
        except Exception as e:  # noqa: BLE001
            findings.append({
                "check_id": "A5", "severity": "FAIL",
                "description": "Failed to extract techniques from entry",
                "details": {"category": category, "error": str(e)},
            })
            continue

        if not techniques:
            # The benign sentinel is intentionally empty — that's not a defect.
            if category != BENIGN_SENTINEL:
                findings.append({
                    "check_id": "A5", "severity": "FAIL",
                    "description": "Entry has no techniques",
                    "details": {"category": category},
                })
            mappings_summary[category] = {
                "n_techniques": 0,
                "technique_ids": [],
                "technique_names": [],
                "confidence_set": [],
                "last_validated": entry.get("last_validated"),
                "is_benign_sentinel": category == BENIGN_SENTINEL,
            }
            continue

        for tid, name, conf in techniques:
            if tid and not TID_RE.match(str(tid)):
                findings.append({
                    "check_id": "A6", "severity": "WARN",
                    "description": "Technique ID does not match MITRE pattern",
                    "details": {"category": category, "tid": tid},
                })
            if not name:
                findings.append({
                    "check_id": "A7", "severity": "WARN",
                    "description": "Technique missing human name",
                    "details": {"category": category, "tid": tid},
                })
            if conf and str(conf).upper() not in VALID_CONFIDENCE:
                findings.append({
                    "check_id": "A8", "severity": "WARN",
                    "description": "Confidence level outside enum",
                    "details": {
                        "category": category, "tid": tid, "confidence": conf,
                    },
                })

        tids = [t[0] for t in techniques]
        names = [t[1] for t in techniques]
        confs = [t[2] for t in techniques]
        mappings_summary[category] = {
            "n_techniques": len(techniques),
            "technique_ids": tids,
            "technique_names": names,
            "confidence_set": sorted({
                str(c).upper() for c in confs if c
            }),
            "confidence_counts": dict(Counter(
                str(c).upper() for c in confs if c
            )),
            "last_validated": entry.get("last_validated"),
            "is_benign_sentinel": category == BENIGN_SENTINEL,
        }

    out["mappings_summary"] = mappings_summary
    out["headline"] = {
        "audit_pass": not any(f["severity"] == "FAIL" for f in findings),
        "n_fail": sum(1 for f in findings if f["severity"] == "FAIL"),
        "n_warn": sum(1 for f in findings if f["severity"] == "WARN"),
        "n_info": sum(1 for f in findings if f["severity"] == "INFO"),
        "mitre_framework_version": framework_version,
        "last_validated_top_level": top_lv,
        "last_validated_per_entry_count": len(per_entry_lvs),
        "n_categories_mapped": len(in_yaml),
        "n_categories_in_data": len(in_data),
        "orphan_categories": orphans,
        "unused_mappings": unused,
        "benign_sentinel_mapped_empty": (
            BENIGN_SENTINEL in mappings_by_cat
            and not _extract_techniques(mappings_by_cat[BENIGN_SENTINEL])
        ),
    }

    _finalize(out, findings)


if __name__ == "__main__":
    main()
