from __future__ import annotations

import subprocess
import sys
import yaml
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent


# ================================================================
# CORE: Run Claude Code subprocess
# ================================================================


def run_agent(role: str, prompt: str) -> str:
    """Spawn Claude Code subprocess cho từng role."""
    model = "claude-opus-4-5" if role in ("pm", "arch") else "claude-sonnet-4-5"
    result = subprocess.run(
        ["claude", "--model", model, "--print", prompt],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )
    return result.stdout


def gate_check(required_files: list[str], gate_name: str) -> bool:
    """Verify required files exist before proceeding."""
    missing = [f for f in required_files if not Path(f).exists()]
    if missing:
        print(f"[GATE {gate_name}] BLOCKED — missing: {missing}")
        return False
    print(f"[GATE {gate_name}] PASS")
    return True


def save_output(filename: str, content: str) -> None:
    Path(filename).write_text(content, encoding="utf-8")
    print(f"  → Saved: {filename}")


# ================================================================
# BA AGENT — Clarify spec, break down tasks
# ================================================================


def ba_agent() -> bool:
    print("\n[BA AGENT] Clarifying spec and breaking down tasks...")

    if not gate_check(["research_spec.yaml", "CLAUDE.md"], "BA-input"):
        return False

    prompt = """
Read research_spec.yaml and CLAUDE.md.

Identify any ambiguous requirements then break down
into concrete implementation tasks.

Output ONLY valid YAML as task_breakdown.yaml:

tasks:
  - id: T1
    component: mve_generator | risk_scorer | harness
    description: "specific what to build"
    file: "src/filename.py"
    acceptance_test: "which test validates this"
    estimated_complexity: low | medium | high
    dependencies: []
    ambiguities_resolved: []

gates:
  proceed: true | false
  reason: "..."
"""
    output = run_agent("ba", prompt)
    save_output("task_breakdown.yaml", output)
    return True


# ================================================================
# ARCHITECT AGENT — Review design before DEV builds
# ================================================================


def architect_agent() -> bool:
    print("\n[ARCHITECT AGENT] Reviewing architecture...")

    if not gate_check(["research_spec.yaml", "task_breakdown.yaml", "CLAUDE.md"], "ARCH-input"):
        return False

    prompt = """
Read research_spec.yaml, task_breakdown.yaml, CLAUDE.md.
Also read existing pipeline/ structure to understand
what already exists before any new code is written.

Review the proposed architecture for these problems:

PROBLEM 1 — Boundary violations:
- Does any task in task_breakdown.yaml touch pipeline/ files?
  (CLAUDE.md says: do NOT touch pipeline/)
- Does any component exceed its single responsibility?

PROBLEM 2 — Interface integrity:
- Do the 3 components (mve_generator, risk_scorer, harness)
  have clean interfaces with no circular dependencies?
- Does src/ correctly WRAP pipeline/ without reimplementing?

PROBLEM 3 — Scope creep:
- Does any task implement something in out_of_scope list?
  (device discovery, automated blocking, RF detection,
   UI/frontend, database, authentication)

PROBLEM 4 — Test coverage:
- Does each component map to at least 1 acceptance test?
- Are all negative tests from research_spec.yaml covered?

Output ONLY valid YAML as architecture_review.yaml:

verdict: APPROVED | BLOCKED
issues:
  - type: boundary_violation | interface_problem | scope_creep | missing_test
    task_id: "T1"
    description: "specific problem"
    fix: "specific fix required"
approved_tasks: [T1, T2, ...]
blocked_tasks: []
notes: "..."
"""
    output = run_agent("arch", prompt)
    save_output("architecture_review.yaml", output)

    # Check verdict
    try:
        review = yaml.safe_load(output)
        verdict = review.get("verdict", "BLOCKED")
        blocked = review.get("blocked_tasks", [])
        issues = review.get("issues", [])

        if verdict == "BLOCKED":
            print(f"  ⚠️  BLOCKED — {len(blocked)} tasks rejected")
            for issue in issues:
                print(f"     [{issue.get('type')}] {issue.get('description')}")
            return False

        print(f"  ✅ APPROVED — {len(review.get('approved_tasks', []))} tasks cleared")
        return True

    except yaml.YAMLError:
        print("  ⚠️  Could not parse architecture_review.yaml")
        return False


# ================================================================
# DEV AGENT — Implement components
# ================================================================


def dev_agent(task_id: str = None) -> bool:
    print(f"\n[DEV AGENT] Implementing {'task ' + task_id if task_id else 'all tasks'}...")

    if not gate_check(
        ["CLAUDE.md", "research_spec.yaml", "task_breakdown.yaml", "architecture_review.yaml"],
        "DEV-input",
    ):
        return False

    # Only proceed with architect-approved tasks
    try:
        review = yaml.safe_load(Path("architecture_review.yaml").read_text())
        if review.get("verdict") != "APPROVED":
            print("  ⚠️  Architecture not approved — DEV blocked")
            return False
        approved = review.get("approved_tasks", [])
    except Exception:
        approved = []

    task_filter = (
        f"Focus ONLY on task {task_id}."
        if task_id
        else f"Implement these approved tasks: {approved}."
    )

    prompt = f"""
Read CLAUDE.md first — follow every instruction strictly.
Read research_spec.yaml, task_breakdown.yaml, architecture_review.yaml.

{task_filter}

Rules:
1. Do NOT touch any file in pipeline/ directory
2. Write tests BEFORE implementation (TDD)
3. Follow file structure in CLAUDE.md exactly
4. Implement Option B (rule-based) fallback first,
   Option A (LLM-based) second
5. After implementation run: python run_tests.py
6. Report each test result: PASS / WARN / FAIL

Build order (from CLAUDE.md):
1. src/data_models.py
2. tests/fixtures/
3. tests/acceptance_tests.py + negative_tests.py
4. src/risk_scorer.py
5. src/mve_generator.py
6. src/harness.py
7. run_tests.py
"""
    output = run_agent("dev", prompt)
    print(f"  DEV output preview: {output[:300]}...")
    return True


# ================================================================
# SECURITY AGENT — HIPAA + clinical safety review
# ================================================================


def security_agent() -> bool:
    print("\n[SECURITY AGENT] Running HIPAA + clinical safety audit...")

    required = [
        "src/mve_generator.py",
        "tests/negative_tests.py",
        "mve_specification.yaml",
    ]
    if not gate_check(required, "SEC-input"):
        print("  Skipping — implementation not complete yet")
        return True  # Non-blocking if files don't exist yet

    prompt = """
You are a healthcare IT security auditor with HIPAA expertise.
Read src/mve_generator.py, tests/negative_tests.py,
mve_specification.yaml, research_spec.yaml.

Audit for 4 categories:

CATEGORY 1 — PHI Exposure in Logs:
- Does Layer 1 (WHY anomalous) log raw patient data?
- Does Layer 2 (Clinical severity) expose identifiable PHI?
- Are patient counts generalized (e.g. "8 patients")
  or specific (e.g. "John Smith, DOB 1980-01-01")?
- PASS condition: no specific patient identifiers in output

CATEGORY 2 — Clinical Safety of Recommendations:
- Does Layer 3 ever recommend disconnecting
  life-sustaining devices without clinical coordination?
- Is "DO NOT power-cycle during active infusion"
  or equivalent present for CRITICAL device alerts?
- PASS condition: every CRITICAL alert has
  explicit clinical constraint

CATEGORY 3 — Scope Boundary Enforcement:
- Does any output claim RF/Bluetooth detection capability?
- Does any output recommend automated blocking
  without human confirmation?
- PASS condition: all negative tests in
  tests/negative_tests.py cover these cases

CATEGORY 4 — Severity Calibration:
- Does CRITICAL severity require life-sustaining
  device involvement?
- Could LOW severity be incorrectly assigned
  to a patient safety event?
- PASS condition: severity criteria match
  mve_specification.yaml severity_criteria field

Output ONLY valid YAML as security_review.yaml:

verdict: PASS | FAIL
phi_exposure: PASS | FAIL
clinical_safety: PASS | FAIL
scope_boundaries: PASS | FAIL
severity_calibration: PASS | FAIL
violations:
  - category: "..."
    severity: critical | warning | info
    location: "file:line or function name"
    description: "specific violation"
    fix: "specific fix required"
cleared_for_user_study: true | false
notes: "..."
"""
    output = run_agent("sec", prompt)
    save_output("security_review.yaml", output)

    try:
        review = yaml.safe_load(output)
        verdict = review.get("verdict", "FAIL")
        violations = review.get("violations", [])
        critical = [v for v in violations if v.get("severity") == "critical"]

        if verdict == "FAIL" or critical:
            print(f"  🚫 FAIL — {len(critical)} critical violations")
            for v in critical:
                print(f"     [{v.get('category')}] {v.get('description')}")
            return False

        warnings = [v for v in violations if v.get("severity") == "warning"]
        print(f"  ✅ PASS — {len(warnings)} warnings (non-blocking)")
        return True

    except yaml.YAMLError:
        print("  ⚠️  Could not parse security_review.yaml")
        return False


# ================================================================
# QA AGENT — Run tests, generate alignment report
# ================================================================


def qa_agent() -> bool:
    print("\n[QA AGENT] Running acceptance tests...")

    if not gate_check(["run_tests.py", "research_claims.yaml"], "QA-input"):
        return False

    prompt = """
Read research_claims.yaml.
Run: python run_tests.py
Read the full output carefully.

Generate alignment_report.yaml strictly:
- result_value must be the ACTUAL number from test output
- Do NOT round up to meet targets
- pass_fail: PASS only if result_value >= minimum
- pass_fail: WARN if result_value >= minimum but < target
- pass_fail: FAIL if result_value < minimum

For recommendation:
- SHIP_TO_USER_STUDY: ALL tests PASS or WARN,
  ALL negative tests PASS (0 violations),
  >= 4/5 claims SUPPORTED
- ITERATE: any test WARN or 1-2 claims PARTIAL
- BLOCKED: any test FAIL or any negative test violation
"""
    output = run_agent("qa", prompt)
    save_output("alignment_report.yaml", output)
    return True


# ================================================================
# PM AGENT — Read report, decide next action
# ================================================================


def pm_agent() -> str:
    print("\n[PM AGENT] Reading alignment report...")

    if not gate_check(["alignment_report.yaml"], "PM-input"):
        return "BLOCKED"

    report_text = Path("alignment_report.yaml").read_text()

    # Read security review if exists
    sec_text = ""
    if Path("security_review.yaml").exists():
        sec_review = yaml.safe_load(Path("security_review.yaml").read_text())
        sec_cleared = sec_review.get("cleared_for_user_study", False)
        sec_text = f"\nSecurity review: {'CLEARED' if sec_cleared else 'NOT CLEARED'}"

    prompt = f"""
Read alignment_report.yaml and security_review.yaml.

alignment_report content:
{report_text}
{sec_text}

Make a final decision considering BOTH technical tests
AND security clearance.

Output ONLY valid YAML as next_action.yaml:

decision: SHIP_TO_USER_STUDY | ITERATE | BLOCKED
reason: "specific reason referencing actual test results"
security_cleared: true | false
next_steps:
  - action: "specific action"
    agent: ba | arch | dev | sec | qa
    priority: high | medium | low
    details: "what exactly to fix"
blocked_claims: []
ship_conditions_met: true | false
message_to_researcher: "1-2 sentences for human PM"
"""
    output = run_agent("pm", prompt)
    save_output("next_action.yaml", output)

    try:
        action = yaml.safe_load(output)
        decision = action.get("decision", "BLOCKED")
        message = action.get("message_to_researcher", "")
        print(f"\n  Decision: {decision}")
        print(f"  Message:  {message}")
        return decision
    except yaml.YAMLError:
        return "BLOCKED"


# ================================================================
# TECH WRITER AGENT — Generate paper sections
# ================================================================


def tech_writer_agent() -> bool:
    print("\n[TECH WRITER AGENT] Generating paper sections...")

    if not gate_check(
        ["alignment_report.yaml", "research_claims.yaml", "problem_evidence_v2.yaml"], "WRITE-input"
    ):
        return False

    prompt = """
Read alignment_report.yaml, research_claims.yaml,
problem_evidence_v2.yaml, mve_specification.yaml.

Write IEEE-format paper sections in paper_sections.md:

## Section 1: Implementation Results (300 words)
- Report each metric with actual values
- Cite test names and result_values
- Do not overclaim — use hedged language for WARN results

## Table 1: Acceptance Test Results
| Metric | Target | Result | Status |
Format all 8 metrics from alignment_report.yaml

## Section 2: Limitations (150 words)
- List claims_not_tested with reasons
- Domain expert validation gap
- AI simulation vs human participants
- Single dataset (WUSTL-EHMS-2020)

## Section 3: Future Work (100 words)
- Phase 2: User study (C4 validation)
- Phase 3: Field deployment (C5 dwell time)
- Domain expert validation
- Multi-site validation

Output: paper_sections.md
Use IEEE passive voice. Be precise with numbers.
"""
    output = run_agent("write", prompt)
    save_output("paper_sections.md", output)
    return True


# ================================================================
# DEVOPS AGENT — Deployment readiness for user study
# ================================================================


def devops_agent() -> bool:
    print("\n[DEVOPS AGENT] Checking deployment readiness...")

    if not gate_check(
        [
            "pipeline/module6_evaluation/module6_app.py",
            "tests/fixtures/user_study_alert_scenarios.yaml",
        ],
        "DEVOPS-input",
    ):
        return False

    prompt = """
Read DEPLOY.md, pipeline/module6_evaluation/module6_app.py,
pipeline/module6_evaluation/study_loader.py,
pipeline/module6_evaluation/study_analysis.py.

Check deployment readiness for Streamlit Cloud:

CHECK 1 — requirements.txt:
- Is streamlit listed?
- Is pyyaml listed?
- Is scipy listed?
- Is numpy listed?
- Any import in study files not in requirements?

CHECK 2 — Hardcoded paths:
- Any absolute paths that break on cloud?
- All paths use PROJECT_ROOT / relative?

CHECK 3 — Secrets:
- Any API keys hardcoded?
- Any local file paths to sensitive data?

CHECK 4 — Smoke test:
- Does study_mode() load without errors
  when no response files exist?
- Does study_analysis.py handle empty
  response directory gracefully?

CHECK 5 — Participant experience:
- Is registration form complete?
- Are Q21 + Q22 proxy questions included?
- Does completion redirect correctly?

Output ONLY valid YAML as deployment_checklist.yaml:

verdict: READY | ISSUES
checks:
  requirements: PASS | FAIL
  paths: PASS | FAIL
  secrets: PASS | FAIL
  smoke_test: PASS | FAIL
  participant_ux: PASS | FAIL
issues:
  - check: "..."
    description: "..."
    fix: "..."
deploy_command: "streamlit run pipeline/module6_evaluation/module6_app.py"
estimated_setup_minutes: 0
"""
    output = run_agent("devops", prompt)
    save_output("deployment_checklist.yaml", output)

    try:
        checklist = yaml.safe_load(output)
        verdict = checklist.get("verdict", "ISSUES")
        issues = checklist.get("issues", [])
        print(f"  Deployment: {verdict} — {len(issues)} issues")
        return verdict == "READY"
    except yaml.YAMLError:
        return False


# ================================================================
# FULL PIPELINE
# ================================================================


def full_pipeline():
    print("=" * 60)
    print("MULTI-AGENT PIPELINE — XAI-IDS-Healthcare")
    print("=" * 60)

    results = {}

    # ── Phase 1: Spec clarification ──
    print("\n── PHASE 1: SPEC ──")
    if not Path("task_breakdown.yaml").exists():
        results["ba"] = ba_agent()
    else:
        print("[BA AGENT] Skipped — task_breakdown.yaml exists")
        results["ba"] = True

    # ── Phase 2: Architecture review ──
    print("\n── PHASE 2: ARCHITECTURE ──")
    if not Path("architecture_review.yaml").exists():
        results["arch"] = architect_agent()
    else:
        print("[ARCH AGENT] Skipped — architecture_review.yaml exists")
        results["arch"] = True

    if not results.get("arch"):
        print("\n🚫 PIPELINE BLOCKED at Architecture Review")
        print("Fix architecture issues then re-run: python orchestrate.py full")
        return

    # ── Phase 3: Implementation ──
    print("\n── PHASE 3: IMPLEMENTATION ──")
    results["dev"] = dev_agent()

    # ── Phase 4: Security + QA (parallel intent, sequential execution) ──
    print("\n── PHASE 4: SECURITY + QA ──")
    results["sec"] = security_agent()
    results["qa"] = qa_agent()

    # ── Phase 5: PM Decision ──
    print("\n── PHASE 5: PM DECISION ──")
    decision = pm_agent()
    results["pm"] = decision

    # ── Phase 6: Loop or Ship ──
    print("\n── PHASE 6: ACTION ──")

    if decision == "ITERATE":
        print("\n🔄 ITERATING — fixing issues and re-running...")
        action = yaml.safe_load(Path("next_action.yaml").read_text())
        for step in action.get("next_steps", []):
            if step["agent"] == "dev":
                dev_agent()
            elif step["agent"] == "sec":
                security_agent()
        qa_agent()
        decision = pm_agent()

    if decision == "SHIP_TO_USER_STUDY":
        print("\n── PHASE 7: SHIP ──")
        results["write"] = tech_writer_agent()
        results["devops"] = devops_agent()

        print("\n" + "=" * 60)
        print("✅ PIPELINE COMPLETE — READY FOR PHASE 2 USER STUDY")
        print("=" * 60)
        print("\nArtifacts generated:")
        for f in [
            "alignment_report.yaml",
            "security_review.yaml",
            "paper_sections.md",
            "deployment_checklist.yaml",
        ]:
            status = "✓" if Path(f).exists() else "✗"
            print(f"  {status} {f}")

    elif decision == "BLOCKED":
        print("\n🚫 PIPELINE BLOCKED — manual intervention required")
        if Path("next_action.yaml").exists():
            action = yaml.safe_load(Path("next_action.yaml").read_text())
            print("\nRequired actions:")
            for step in action.get("next_steps", []):
                print(f"  [{step['agent'].upper()}] {step['details']}")

    # ── Summary ──
    print("\n── PIPELINE SUMMARY ──")
    for agent, result in results.items():
        status = "✅" if result else "❌"
        print(f"  {status} {agent.upper()}")


# ================================================================
# ENTRY POINT
# ================================================================

AGENTS = {
    "ba": ba_agent,
    "arch": architect_agent,
    "dev": dev_agent,
    "sec": security_agent,
    "qa": qa_agent,
    "pm": pm_agent,
    "write": tech_writer_agent,
    "deploy": devops_agent,
    "full": full_pipeline,
}

if __name__ == "__main__":
    task = sys.argv[1] if len(sys.argv) > 1 else "full"

    if task not in AGENTS:
        print(f"Unknown task: {task}")
        print(f"Available: {', '.join(AGENTS.keys())}")
        sys.exit(1)

    AGENTS[task]()
