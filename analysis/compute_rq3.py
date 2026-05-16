"""RQ3: Architectural safety verification + HITL user study.

Sub-tasks:
  RQ3.1 — pytest test summary
  RQ3.2 — No-auto-execution grep verification on module5_responses/
  RQ3.3 — Audit log hash chain verification
  RQ3.4 — Cross-role consistency (shared anchor, severity, action authorization)
  RQ3.5 — HITL user-study analysis (escalation, confidence, decision distribution)
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from analysis._common import (
    RESULTS_DIR,
    build_provenance,
    file_hashes,
    log,
    section_begin,
    section_end,
    write_json,
)

CONFIGS = REPO / "configs"
REPORTS = RESULTS_DIR / "reports"
SURVEY = REPO / "survey"


# --------------------------------------------------------------------------
# RQ3.1 — Test suite summary
# --------------------------------------------------------------------------
def compute_rq3_1() -> dict[str, Any]:
    section = "RQ3.1"
    start = section_begin(section, "pytest summary")

    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-v", "--no-cov", "--tb=no",
         "--no-header", "-q", "--disable-warnings"],
        cwd=str(REPO), capture_output=True, text=True, timeout=900,
    )
    out_text = proc.stdout + "\n" + proc.stderr
    # Save raw log
    (RESULTS_DIR / "rq3_pytest_raw.log").write_text(out_text)
    log(section, f"pytest exit code: {proc.returncode}")

    # Parse summary
    # Look for lines like "tests/test_xxx.py::test_yyy PASSED|FAILED|SKIPPED"
    per_file = defaultdict(lambda: {"passed": 0, "failed": 0, "skipped": 0, "tests": []})
    pattern = re.compile(r"^(tests/[^:]+\.py)::([^\s]+)\s+(PASSED|FAILED|SKIPPED|ERROR|XFAIL|XPASS)")
    for line in out_text.splitlines():
        m = pattern.match(line.strip())
        if m:
            fn, test, status = m.group(1), m.group(2), m.group(3)
            entry = per_file[fn]
            entry["tests"].append({"name": test, "status": status})
            if status == "PASSED":
                entry["passed"] += 1
            elif status in ("FAILED", "ERROR"):
                entry["failed"] += 1
            elif status == "SKIPPED":
                entry["skipped"] += 1
    files_summary: list[dict[str, Any]] = []
    total_passed = total_failed = total_skipped = 0
    for fn, info in sorted(per_file.items()):
        files_summary.append({
            "file": fn,
            "test_count": len(info["tests"]),
            "passed": info["passed"],
            "failed": info["failed"],
            "skipped": info["skipped"],
            "status": "passed" if info["failed"] == 0 else "failed",
        })
        total_passed += info["passed"]
        total_failed += info["failed"]
        total_skipped += info["skipped"]

    payload = {
        "provenance": build_provenance(input_files=file_hashes()),
        "results": {
            "test_files": files_summary,
            "total_tests": total_passed + total_failed + total_skipped,
            "passed": total_passed,
            "failed": total_failed,
            "skipped": total_skipped,
            "pytest_exit_code": proc.returncode,
            "overall_status": "PASSED" if total_failed == 0 else "FAILED",
        },
    }
    write_json(RESULTS_DIR / "rq3_test_summary.json", payload)
    log(section, f"OUTPUT: rq3_test_summary.json (passed={total_passed} failed={total_failed} skipped={total_skipped})")
    section_end(section, start, f"passed={total_passed} failed={total_failed}")
    return payload


# --------------------------------------------------------------------------
# RQ3.2 — No-auto-execution grep verification
# --------------------------------------------------------------------------
def compute_rq3_2() -> dict[str, Any]:
    section = "RQ3.2"
    start = section_begin(section, "no-auto-execution grep on module5_responses/")

    m5_dir = REPO / "module5_responses"
    grep_cmd_1 = [
        "grep", "-rnE",
        r"subprocess|os\.system|iptables|netcat|\bnc\b|\bcurl\b|\bwget\b|\bssh\b|\bsudo\b|\beval\(|\bexec\(",
        str(m5_dir),
    ]
    grep_cmd_2 = [
        "grep", "-rnE",
        r"^import subprocess|^from subprocess",
        str(m5_dir),
    ]
    p1 = subprocess.run(grep_cmd_1, capture_output=True, text=True)
    p2 = subprocess.run(grep_cmd_2, capture_output=True, text=True)

    matches_1 = [m for m in p1.stdout.splitlines() if m.strip()]
    matches_2 = [m for m in p2.stdout.splitlines() if m.strip()]

    grep_status = "PASSED" if not matches_1 else "FAILED"
    import_status = "PASSED" if not matches_2 else "FAILED"

    payload = {
        "provenance": build_provenance(input_files=file_hashes()),
        "results": {
            "grep_check": {
                "command": " ".join(grep_cmd_1),
                "matches": matches_1,
                "n_matches": len(matches_1),
                "status": grep_status,
            },
            "import_check": {
                "command": " ".join(grep_cmd_2),
                "matches": matches_2,
                "n_matches": len(matches_2),
                "status": import_status,
            },
            "overall_verdict": (
                "PASSED — No auto-execution verified"
                if (grep_status == "PASSED" and import_status == "PASSED")
                else "FAILED — auto-execution code paths detected"
            ),
        },
    }
    write_json(RESULTS_DIR / "rq3_no_auto_execution.json", payload)
    log(section, f"OUTPUT: rq3_no_auto_execution.json ({payload['results']['overall_verdict']})")
    section_end(section, start, payload["results"]["overall_verdict"])
    return payload


# --------------------------------------------------------------------------
# RQ3.3 — Audit log hash chain verification
# --------------------------------------------------------------------------
def compute_rq3_3() -> dict[str, Any]:
    section = "RQ3.3"
    start = section_begin(section, "audit hash chain verification")

    log_files: list[Path] = []
    al = REPORTS / "alert_responses.json"
    audit = REPORTS / "audit_log.jsonl"
    if audit.exists():
        log_files.append(audit)
    if al.exists():
        log_files.append(al)
    # survey log files
    survey_logs = sorted(SURVEY.glob("study_responses_*.json"))

    results: list[dict[str, Any]] = []

    for lf in log_files:
        try:
            entries = _load_log(lf)
            verdict = _verify_chain(entries)
            results.append({"file": str(lf.relative_to(REPO)), **verdict})
        except Exception as exc:
            results.append({"file": str(lf.relative_to(REPO)), "verification_status": "ERROR",
                            "error": str(exc), "chain_intact": False, "n_entries": 0})
    # Aggregate survey files: just count entries (their structure isn't hash-chained)
    n_survey_files = len(survey_logs)
    if n_survey_files:
        # Count rows for completeness reporting
        total_rows = 0
        for sl in survey_logs:
            try:
                doc = json.loads(sl.read_text())
                total_rows += len(doc.get("rows", []))
            except Exception:
                continue
        results.append({
            "file": f"survey/study_responses_*.json (n={n_survey_files})",
            "n_entries": total_rows,
            "chain_intact": None,  # survey JSON has no hash chain — N/A
            "verification_status": "N/A — no hash chain in survey JSON",
        })

    total_entries = sum(r["n_entries"] for r in results if isinstance(r.get("n_entries"), int))
    chains = [r for r in results if r.get("chain_intact") is not None]
    all_intact = (
        bool(chains) and all(r["chain_intact"] for r in chains)
    )

    payload = {
        "provenance": build_provenance(input_files=file_hashes()),
        "results": {
            "logs_checked": results,
            "total_logs": len(results),
            "total_entries": total_entries,
            "all_chains_intact": all_intact,
            "overall_status": "PASSED" if all_intact else "FAILED",
        },
    }
    write_json(RESULTS_DIR / "rq3_audit_integrity.json", payload)
    log(section, f"OUTPUT: rq3_audit_integrity.json ({payload['results']['overall_status']})")
    section_end(section, start, f"all_intact={all_intact}")
    return payload


def _load_log(path: Path) -> list[dict[str, Any]]:
    """Load audit log — supports both .jsonl and JSON-array files."""
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return json.loads(path.read_text())


def _verify_chain(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Verify hash chain. Supports two field-name conventions:
    'prev_hash' + 'integrity_hash' (alert_responses.json) and
    'previous_hash' + 'entry_hash' (canonical task prompt schema)."""
    if not entries:
        return {"n_entries": 0, "chain_intact": True, "verification_status": "PASSED (empty)"}
    # Auto-detect field names
    sample = entries[0]
    if "integrity_hash" in sample and "prev_hash" in sample:
        prev_key, hash_key = "prev_hash", "integrity_hash"
    elif "entry_hash" in sample and "previous_hash" in sample:
        prev_key, hash_key = "previous_hash", "entry_hash"
    else:
        # No hash chain present
        return {"n_entries": len(entries), "chain_intact": None,
                "verification_status": "N/A — no hash chain fields"}

    GENESIS = "0" * 64
    expected_prev = GENESIS
    chain_restarts: list[int] = []
    for i, e in enumerate(entries):
        # Check previous hash
        actual_prev = e.get(prev_key)
        if actual_prev != expected_prev:
            # Chain restart: per module5_pipeline.py AuditLogger._recover_prev_hash,
            # a fresh chain may begin with prev_hash=GENESIS (e.g. after archival).
            if actual_prev == GENESIS:
                chain_restarts.append(i)
                expected_prev = GENESIS  # accept restart
            else:
                return {
                    "n_entries": len(entries),
                    "chain_intact": False,
                    "verification_status": "FAILED",
                    "broken_at": i,
                    "reason": f"prev_hash mismatch at entry {i}: expected {expected_prev[:16]}... got {(actual_prev or '')[:16]}...",
                }
        # Recompute integrity hash. The codebase supports two encodings:
        #   1. canonical compact:   sort_keys=True, separators=(",",":")
        #   2. legacy default-sep:  sort_keys=True only (default separators)
        # AuditLogger.verify() in module5_pipeline.py accepts either as a
        # migration-window legacy fallback. We replicate that here.
        # Signature-related fields (signature, signing_key_id, signature_alg)
        # are added AFTER integrity_hash is computed, so they must be
        # excluded when recomputing the hash on signed records.
        SIG_FIELDS = {hash_key, "signature", "signing_key_id", "signature_alg"}
        e_for_hash = {k: v for k, v in e.items() if k not in SIG_FIELDS}
        compact = hashlib.sha256(
            json.dumps(e_for_hash, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        legacy = hashlib.sha256(
            json.dumps(e_for_hash, sort_keys=True).encode()
        ).hexdigest()
        expected_hash = compact if e.get(hash_key) == compact else legacy
        if e.get(hash_key) not in (compact, legacy):
            return {
                "n_entries": len(entries),
                "chain_intact": False,
                "verification_status": "FAILED",
                "broken_at": i,
                "reason": (
                    f"hash mismatch at entry {i}: "
                    f"computed {expected_hash[:16]}... stored {(e.get(hash_key) or '')[:16]}..."
                ),
            }
        expected_prev = e[hash_key]

    return {
        "n_entries": len(entries),
        "chain_intact": True,
        "verification_status": "PASSED",
        "chain_restarts": chain_restarts,
        "n_chain_restarts": len(chain_restarts),
    }


# --------------------------------------------------------------------------
# RQ3.4 — Cross-role consistency
# --------------------------------------------------------------------------
def compute_rq3_4() -> dict[str, Any]:
    section = "RQ3.4"
    start = section_begin(section, "cross-role consistency checks")

    # Source: evaluation_alerts.json. Each alert carries one mve_structured —
    # to fully validate Invariants 6/9 we need all 3 role views. If only a
    # single rendering is present, we verify the shared_anchor exists and
    # the structure is valid; full per-role comparison is best-effort.
    eval_path = REPORTS / "evaluation_alerts.json"
    role_auth_path = CONFIGS / "role_action_authorization.yaml"

    if not eval_path.exists():
        payload = {
            "provenance": build_provenance(input_files=file_hashes()),
            "results": {"status": "pending", "reason": "evaluation_alerts.json missing"},
        }
        write_json(RESULTS_DIR / "rq3_cross_role_consistency.json", payload)
        section_end(section, start, "pending")
        return payload

    alerts = json.loads(eval_path.read_text())
    role_auth_doc = yaml.safe_load(role_auth_path.read_text()) if role_auth_path.exists() else {}
    role_forbidden = role_auth_doc.get("forbidden_terms") or role_auth_doc

    n_alerts = len(alerts)
    anchor_violations: list[dict[str, Any]] = []
    severity_violations: list[dict[str, Any]] = []
    auth_violations: list[dict[str, Any]] = []
    n_anchors_present = 0

    for a in alerts:
        anchor = a.get("shared_anchor")
        # Invariant 9 — shared anchor presence and key fields
        required = ["alert_id", "risk_tier", "device_id", "one_line_summary", "timestamp"]
        if not isinstance(anchor, dict) or not all(k in anchor for k in required):
            anchor_violations.append({
                "alert_id": a.get("alert_id"),
                "missing": [k for k in required if not (anchor or {}).get(k)],
            })
            continue
        n_anchors_present += 1

        # Invariant 6 — severity consistency: anchor.risk_tier matches risk_level field
        if anchor["risk_tier"] != a.get("risk_level"):
            severity_violations.append({
                "alert_id": a.get("alert_id"),
                "anchor_tier": anchor["risk_tier"],
                "risk_level": a.get("risk_level"),
            })

        # Invariant 6 — Layer 3 action authorization (single rendering check)
        mve = a.get("mve_structured", {})
        l3 = mve.get("layer_3", {})
        action_text = (l3.get("immediate_action") or "").lower()
        # Without per-role text we cannot enforce per-role authorization. We
        # check that the unified action is *legitimate* (not in any role's
        # forbidden list under the most-restrictive role, e.g. nurse_manager).
        if isinstance(role_forbidden, dict):
            most_restrictive = role_forbidden.get("nurse_manager") or role_forbidden.get("Nurse_Manager") or []
            if isinstance(most_restrictive, list):
                for term in most_restrictive:
                    if isinstance(term, str) and term.lower() in action_text:
                        auth_violations.append({
                            "alert_id": a.get("alert_id"),
                            "forbidden_term": term,
                            "matched_in_action": action_text[:80],
                            "note": "Layer 3 action contains a term forbidden for nurse_manager",
                        })
                        break

    inv9_ok = len(anchor_violations) == 0
    inv6_sev_ok = len(severity_violations) == 0
    inv6_auth_ok = len(auth_violations) == 0
    overall_ok = inv9_ok and inv6_sev_ok and inv6_auth_ok

    payload = {
        "provenance": build_provenance(input_files=file_hashes()),
        "results": {
            "alerts_checked": n_alerts,
            "anchors_present": n_anchors_present,
            "invariant_9_shared_anchor": {
                "all_identical": inv9_ok,
                "violations": anchor_violations,
                "n_violations": len(anchor_violations),
            },
            "invariant_6_severity": {
                "all_identical": inv6_sev_ok,
                "violations": severity_violations,
                "n_violations": len(severity_violations),
            },
            "invariant_6_action_authorization": {
                "all_authorized": inv6_auth_ok,
                "violations": auth_violations,
                "n_violations": len(auth_violations),
                "note": (
                    "Checked against the most-restrictive role (nurse_manager). "
                    "Full per-role comparison requires per-role MVE renderings, "
                    "which evaluation_alerts.json does not include (each alert "
                    "has one mve_structured for the active study condition)."
                ),
            },
            "overall_status": "PASSED" if overall_ok else "FAILED",
        },
    }
    write_json(RESULTS_DIR / "rq3_cross_role_consistency.json", payload)
    log(section, f"OUTPUT: rq3_cross_role_consistency.json ({payload['results']['overall_status']})")
    section_end(section, start, payload["results"]["overall_status"])
    return payload


# --------------------------------------------------------------------------
# RQ3.5 — HITL user study analysis
# --------------------------------------------------------------------------
def compute_rq3_5() -> dict[str, Any]:
    section = "RQ3.5"
    start = section_begin(section, "HITL user-study analysis")

    files = sorted(SURVEY.glob("study_responses_*.json"))
    if not files:
        payload = {
            "provenance": build_provenance(input_files=file_hashes()),
            "results": {"status": "pending", "reason": "no survey/study_responses_*.json files"},
        }
        write_json(RESULTS_DIR / "rq3_user_study.json", payload)
        section_end(section, start, "pending")
        return payload

    # Aggregate per-role distributions
    per_role: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for f in files:
        try:
            doc = json.loads(f.read_text())
        except Exception:
            continue
        pid = doc.get("persona_id", f.stem)
        # Persona_id form like "biomed_engineer_P01" — first underscore-prefixed
        # token chain is the role.
        role = _role_from_pid(pid)
        rows = doc.get("rows", [])
        per_role[role].extend(rows)

    role_summary: dict[str, dict[str, Any]] = {}
    for role, rows in per_role.items():
        actions = Counter()
        conditions_actions: dict[str, Counter] = defaultdict(Counter)
        confidence_by_cond: dict[str, list[int]] = defaultdict(list)
        correct_by_cond: dict[str, list[int]] = defaultdict(list)
        escalation_by_cond: dict[str, int] = defaultdict(int)
        n_by_cond: dict[str, int] = defaultdict(int)
        for r in rows:
            action = (r.get("response") or {}).get("action")
            actions[action] += 1
            cond = r.get("condition", "?")
            conditions_actions[cond][action] += 1
            confidence = (r.get("response") or {}).get("confidence")
            if isinstance(confidence, (int, float)):
                confidence_by_cond[cond].append(int(confidence))
            n_by_cond[cond] += 1
            correct = (r.get("response") or {}).get("action") == r.get("correct_action")
            correct_by_cond[cond].append(int(correct))
            if action == "escalate":
                escalation_by_cond[cond] += 1
        summary = {
            "n_responses": sum(actions.values()),
            "action_distribution": dict(actions),
            "by_condition": {},
        }
        for cond in sorted(conditions_actions.keys()):
            n = n_by_cond[cond]
            confs = confidence_by_cond.get(cond, [])
            acc = correct_by_cond.get(cond, [])
            summary["by_condition"][cond] = {
                "n": n,
                "action_distribution": dict(conditions_actions[cond]),
                "escalation_rate": round(escalation_by_cond[cond] / n, 4) if n else None,
                "mean_confidence": round(float(sum(confs) / len(confs)), 4) if confs else None,
                "accuracy": round(float(sum(acc) / len(acc)), 4) if acc else None,
            }
        role_summary[role] = summary

    # Chi-square escalation comparison between condition A and B per role
    chi2_results = {}
    try:
        from scipy.stats import chi2_contingency
        for role, s in role_summary.items():
            byc = s["by_condition"]
            if "A" in byc and "B" in byc:
                a_esc = (byc["A"]["escalation_rate"] or 0) * byc["A"]["n"]
                a_not = byc["A"]["n"] - a_esc
                b_esc = (byc["B"]["escalation_rate"] or 0) * byc["B"]["n"]
                b_not = byc["B"]["n"] - b_esc
                table = [[a_esc, a_not], [b_esc, b_not]]
                try:
                    chi2, p, _, _ = chi2_contingency(table)
                    chi2_results[role] = {
                        "chi2": round(float(chi2), 4),
                        "p_value": round(float(p), 4),
                        "table": [[int(round(x)) for x in row] for row in table],
                    }
                except Exception as exc:
                    chi2_results[role] = {"error": str(exc)}
    except Exception:
        pass

    payload = {
        "provenance": build_provenance(input_files=file_hashes(),
                                       extra={"n_survey_files": len(files)}),
        "results": {
            "per_role": role_summary,
            "escalation_chi_square_A_vs_B": chi2_results,
            "n_files_loaded": len(files),
        },
    }
    write_json(RESULTS_DIR / "rq3_user_study.json", payload)
    log(section, f"OUTPUT: rq3_user_study.json (roles={list(role_summary.keys())})")
    section_end(section, start, f"roles={len(role_summary)}")
    return payload


def _role_from_pid(pid: str) -> str:
    # Examples: "biomed_engineer_P01", "IT_P14", "nurse_manager_P05"
    parts = pid.split("_")
    # Strip the "PNN" terminal token
    if parts and re.match(r"^P\d+$", parts[-1]):
        parts = parts[:-1]
    return "_".join(parts) if parts else pid


def main() -> None:
    compute_rq3_2()
    compute_rq3_3()
    compute_rq3_4()
    compute_rq3_5()
    # Run tests LAST (heaviest)
    compute_rq3_1()


if __name__ == "__main__":
    main()
