.PHONY: phase0-baseline phase0-check phase0-update-floors test-phase0 \
        phase1-baselines phase1-regen phase1-verify test-phase1 \
        phase2-regen phase2-verify test-phase2 \
        phase4-gate phase4-round2-sim test-phase4 help \
        lock security-scan

PYTHON ?= python3

help:
	@echo "Phase-0 instrumentation targets:"
	@echo "  make phase0-baseline       Recompute metrics, write results/phase0_baseline.json"
	@echo "  make phase0-check          Recompute and fail if any metric regressed below recorded floor"
	@echo "  make phase0-update-floors  Re-baseline floors to current values (after a phase that raises them)"
	@echo "  make test-phase0           Run pytest for the new phase0_metrics module"
	@echo ""
	@echo "Phase-1 plumbing targets:"
	@echo "  make phase1-baselines      Build artifacts/feature_baselines.json from train_phase1.parquet"
	@echo "  make phase1-regen          Offline regen of clinician/analyst artifacts with Phase 1.1 enrichment"
	@echo "  make phase1-verify         baselines → regen → phase0-baseline (one-shot verification)"
	@echo "  make test-phase1           Run pytest for Phase 1 upgrades"

phase0-baseline:
	$(PYTHON) -m tools.phase0_baseline

phase0-check:
	$(PYTHON) -m tools.phase0_baseline --check

phase0-update-floors:
	$(PYTHON) -m tools.phase0_baseline --update-floors

test-phase0:
	$(PYTHON) -m pytest tests/test_phase0_metrics.py -v

phase1-baselines:
	$(PYTHON) -m tools.build_feature_baselines

phase1-regen:
	$(PYTHON) -m tools.phase1_regen_module4 test

phase1-verify: phase1-baselines phase1-regen phase0-baseline

test-phase1:
	$(PYTHON) -m pytest tests/test_phase1_upgrades.py -v

phase2-regen:
	$(PYTHON) -m tools.phase1_regen_module4 test
	$(PYTHON) -m tools.phase1_regen_module5 test

phase2-verify: phase1-baselines phase2-regen phase0-baseline

test-phase2:
	$(PYTHON) -m pytest tests/test_phase2_counterfactual.py -v

phase4-gate:
	$(PYTHON) -m tools.faithfulness_gate --check

phase4-round2-sim:
	$(PYTHON) -m tools.phase4_round2_simulator

test-phase4:
	$(PYTHON) -m pytest tests/test_phase4_stability.py -v

# ── Supply chain / security ────────────────────────────────────────
# `make lock` regenerates requirements.lock with --generate-hashes.
# MUST run in a Python 3.11+ environment because keras>=3.13.2 (required
# to close CVE-2026-1462 and PYSEC-2026-73) only ships wheels for 3.11+.
lock:
	@PY_MINOR=$$($(PYTHON) -c "import sys;print(sys.version_info.minor)"); \
	if [ "$$PY_MINOR" -lt 11 ]; then \
	  echo "ERROR: make lock requires Python 3.11+ (current = 3.$$PY_MINOR)."; \
	  echo "  Use a 3.11+ venv: python3.11 -m venv .venv && . .venv/bin/activate"; \
	  exit 1; \
	fi
	$(PYTHON) -m pip install --quiet pip-tools
	$(PYTHON) -m piptools compile --generate-hashes --resolver=backtracking \
	  --output-file=requirements.lock requirements.in

# Local security scan mirroring the CI matrix. Uses an isolated venv
# so sec-tool CVEs do not pollute the report.
security-scan:
	@test -d /tmp/sec-tools-local || \
	  (python3 -m venv /tmp/sec-tools-local && \
	   /tmp/sec-tools-local/bin/pip install --quiet --upgrade pip bandit[toml] pip-audit cyclonedx-bom)
	@echo "── Bandit ──"
	/tmp/sec-tools-local/bin/bandit -r module0_analysis/ common/ -c pyproject.toml -ll || true
	@echo "── pip-audit (requirements.in) ──"
	/tmp/sec-tools-local/bin/pip-audit -r requirements.in --strict --desc || true
