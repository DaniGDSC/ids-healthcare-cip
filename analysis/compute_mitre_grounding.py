"""RQ2.e — MVE Layer 1 MITRE-grounding rate (RQ2_Mitre.md Phase 2).

For every alert with an MVE in ``results/reports/mve_outputs.jsonl``,
check whether ``layer_1_why_anomalous`` references the MITRE technique
the alert's ``attack_category`` is mapped to (per the audit).

Match rule (RQ2_Mitre.md §2 locked decision):
    Case-insensitive substring; either the T-ID or the human technique
    name counts as a hit.

Three sample sets reported:
  * Headline — surfaced alerts (``fusion_class != BENIGN``)
  * Per-attack-category — paired against the mapped technique terms
  * Appendix — all MVE records (in case --include-benign was used in
    the Module 5 batch)

Plus by-mode breakdown (Mode A LLM vs Mode B rule-based) and a strict
variant: ``T-ID AND human name both present`` reported alongside the
permissive metric at zero extra cost.

Inputs (Phase 0 confirmed):
  * results/rq2_mitre_audit.json     — category → match terms
  * results/reports/mve_outputs.jsonl — produced by Module 5 batch
  * results/reports/risk_scores.npz  — surfaced mask + alignment check

Output: results/rq2_mitre_grounding.json
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
AUDIT = REPO_ROOT / "results/rq2_mitre_audit.json"
NPZ = REPO_ROOT / "results/reports/risk_scores.npz"
MVE_OUTPUTS = REPO_ROOT / "results/reports/mve_outputs.jsonl"
OUT = REPO_ROOT / "results/rq2_mitre_grounding.json"

GROUNDED_TARGET = 0.90
BENIGN_SENTINEL = "normal"
TID_RE = re.compile(r"^t\d{4}(\.\d{3})?$")


def _sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_match_terms_from_audit() -> dict[str, dict]:
    """category -> {'tids': set[str], 'names': set[str]}, all lowercase."""
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    out: dict[str, dict] = {}
    for cat, entry in audit.get("mappings_summary", {}).items():
        tids: set[str] = set()
        names: set[str] = set()
        for tid in entry.get("technique_ids", []) or []:
            if tid:
                tids.add(str(tid).lower())
        for name in entry.get("technique_names", []) or []:
            if name:
                names.add(str(name).lower())
        out[cat] = {"tids": tids, "names": names}
    return out


def _load_mve(path: Path) -> dict[int, dict]:
    """Load MVE JSONL keyed by ``row_id``.

    Each record is the union of ``MVEOutput.to_dict()`` and the per-row
    surfacing context the Module 5 batch attached.  We trust those
    fields rather than re-asserting their presence.
    """
    out: dict[int, dict] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            out[int(rec["row_id"])] = rec
    return out


def _grounding_for_alert(
    layer1_text: str,
    tids: set[str],
    names: set[str],
) -> tuple[bool, bool, int, int, list[str]]:
    """Return (grounded, strict_grounded, tid_hits, name_hits, hits_list)."""
    if not layer1_text:
        return False, False, 0, 0, []
    txt = layer1_text.lower()
    matched: list[str] = []
    tid_hit_count = 0
    for t in tids:
        if t in txt:
            tid_hit_count += 1
            matched.append(t)
    name_hit_count = 0
    for n in names:
        if n in txt:
            name_hit_count += 1
            matched.append(n)
    grounded = (tid_hit_count + name_hit_count) > 0
    strict = (tid_hit_count > 0) and (name_hit_count > 0)
    return grounded, strict, tid_hit_count, name_hit_count, matched


def _aggregate(records: list[dict], scope_name: str) -> dict:
    n = len(records)
    if n == 0:
        return {
            "_scope": scope_name,
            "n_evaluated": 0,
            "grounded_pct": None,
            "strict_grounded_pct": None,
        }
    grounded = sum(1 for r in records if r["grounded"])
    strict = sum(1 for r in records if r["strict_grounded"])
    return {
        "_scope": scope_name,
        "n_evaluated": n,
        "n_grounded": grounded,
        "n_strict_grounded": strict,
        "grounded_pct": round(grounded / n, 4),
        "strict_grounded_pct": round(strict / n, 4),
    }


def main() -> None:
    if not AUDIT.exists():
        raise SystemExit(
            f"{AUDIT} missing — run "
            "`python -m analysis.audit_mitre_config` first."
        )
    if not MVE_OUTPUTS.exists():
        raise SystemExit(
            f"{MVE_OUTPUTS} missing — run "
            "`python -m module5_responses.module5_mve_batch` first."
        )

    category_to_terms = _load_match_terms_from_audit()
    mve = _load_mve(MVE_OUTPUTS)
    data = np.load(NPZ, allow_pickle=True)

    row_ids = np.asarray(data["row_id"]).astype(int)
    attack_cats = np.asarray(data["attack_category"]).astype(str)
    fusion = np.asarray(data["fusion_class"]).astype(str)

    records: list[dict] = []
    failure_examples: list[dict] = []
    skipped_no_mve = 0
    skipped_benign = 0

    for i in range(len(row_ids)):
        rid = int(row_ids[i])
        cat = str(attack_cats[i])

        if cat == BENIGN_SENTINEL:
            skipped_benign += 1
            continue
        if rid not in mve:
            skipped_no_mve += 1
            continue

        mve_rec = mve[rid]
        layer1 = mve_rec.get("layer_1_why_anomalous", "") or ""
        terms = category_to_terms.get(cat, {"tids": set(), "names": set()})
        grounded, strict, tid_h, name_h, matched = _grounding_for_alert(
            layer1, terms["tids"], terms["names"],
        )

        rec = {
            "row_id": rid,
            "category": cat,
            # The Module 5 batch uses MVEOutput.mode_used (A_llm / B_rule).
            "mode": mve_rec.get("mode_used", "unknown"),
            "fusion_class": str(fusion[i]),
            "surfaced": str(fusion[i]) != "BENIGN",
            "grounded": grounded,
            "strict_grounded": strict,
            "tid_hits": tid_h,
            "name_hits": name_h,
            "matched_terms": matched,
        }
        records.append(rec)

        if not grounded and len(failure_examples) < 10:
            failure_examples.append({
                "row_id": rid,
                "category": cat,
                "mode": rec["mode"],
                "fusion_class": rec["fusion_class"],
                "expected_tids": sorted(terms["tids"]),
                "expected_names": sorted(terms["names"]),
                "layer1_excerpt": layer1[:200],
            })

    surfaced = [r for r in records if r["surfaced"]]
    headline = _aggregate(surfaced, "surfaced alerts (fusion_class != BENIGN)")
    headline["target"] = GROUNDED_TARGET
    headline["pass"] = (
        headline.get("grounded_pct") is not None
        and headline["grounded_pct"] >= GROUNDED_TARGET
    )

    by_cat: dict[str, dict] = {}
    for cat in sorted({r["category"] for r in surfaced}):
        cat_records = [r for r in surfaced if r["category"] == cat]
        agg = _aggregate(cat_records, f"surfaced alerts of category={cat}")
        terms = category_to_terms.get(cat, {"tids": set(), "names": set()})
        agg["expected_tids"] = sorted(terms["tids"])
        agg["expected_names"] = sorted(terms["names"])
        agg["_pair_validity"] = (
            "paired against mapped technique for this category"
        )
        by_cat[cat] = agg

    by_mode: dict[str, dict] = {}
    for m in sorted({r["mode"] for r in surfaced}):
        m_records = [r for r in surfaced if r["mode"] == m]
        agg = _aggregate(m_records, f"surfaced alerts, mode={m}")
        if m == "B_rule":
            agg["_note"] = (
                "Rule-based MVE template — MITRE term injection depends "
                "on whether src.mve_generator includes T-IDs/names in "
                "layer_1_why_anomalous strings; this number is the truth."
            )
        elif m == "A_llm":
            agg["_note"] = (
                "LLM-generated MVE; grounding depends on prompt design."
            )
        by_mode[m] = agg

    appendix = _aggregate(records, "all alerts with an MVE output (excludes benign-sentinel rows)")

    out: dict[str, Any] = {
        "_meta": {
            "schema_version": "1.0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "generated_by": "analysis/compute_mitre_grounding.py",
            "inputs": {
                "audit_json": str(AUDIT.relative_to(REPO_ROOT)),
                "audit_sha256": _sha256(AUDIT),
                "mve_outputs": str(MVE_OUTPUTS.relative_to(REPO_ROOT)),
                "mve_outputs_sha256": _sha256(MVE_OUTPUTS),
                "risk_scores_npz": str(NPZ.relative_to(REPO_ROOT)),
                "n_records_in_mve_jsonl": len(mve),
                "n_evaluated": len(records),
                "n_surfaced": len(surfaced),
                "n_skipped_benign_sentinel": skipped_benign,
                "n_skipped_no_mve_for_row": skipped_no_mve,
            },
            "config": {
                "match_rule": (
                    "case-insensitive substring; T-ID OR human name accepted"
                ),
                "search_scope": "layer_1_why_anomalous field only",
                "strict_appendix_metric": "T-ID AND human name both present",
                "grounded_target": GROUNDED_TARGET,
                "benign_sentinel_excluded": BENIGN_SENTINEL,
            },
        },
        "headline": headline,
        "by_attack_category": by_cat,
        "by_mode": by_mode,
        "appendix_all_mve": appendix,
        "failure_examples": failure_examples,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {OUT.relative_to(REPO_ROOT)}")
    gp = headline.get("grounded_pct")
    sp = headline.get("strict_grounded_pct")
    gp_str = f"{gp:.3f}" if gp is not None else "n/a"
    sp_str = f"{sp:.3f}" if sp is not None else "n/a"
    print(
        f"Grounding (surfaced):  {gp_str} (target >= {GROUNDED_TARGET})  "
        f"{'PASS' if headline.get('pass') else 'FAIL'}"
    )
    print(f"Strict (both T-ID + name): {sp_str}")
    print(
        f"Evaluated: {len(records)} (surfaced={len(surfaced)}, "
        f"skipped benign={skipped_benign}, no-mve={skipped_no_mve})"
    )


if __name__ == "__main__":
    main()
