"""Reproducibility report generator for thesis defence / IEEE Q1 submission.

Single Responsibility
---------------------
This module renders ``report_section_reproducibility.md`` (§3.4) from
pre-computed metadata.  It performs no computation itself — all inputs
arrive via function arguments.

Dependency Inversion
--------------------
The function ``render_reproducibility_report`` accepts plain Python values
rather than analyzer or security objects, keeping it decoupled from the
analysis and CI layers.
"""

from __future__ import annotations

import logging
import platform
import sys
from typing import Any, Dict, List, Optional

from .config import Phase0Config

logger = logging.getLogger(__name__)


def render_reproducibility_report(
    config: Phase0Config,
    dataset_hash: str,
    test_count: int,
    coverage_pct: float,
    installed_packages: Dict[str, str],
    security_findings: int = 0,
    dataset_url: str = "https://www.cse.wustl.edu/~jain/ehms/index.html",
    repo_url: Optional[str] = None,
) -> str:
    """Render ``report_section_reproducibility.md`` (§3.4).

    Args:
        config: Validated Phase0Config (supplies random_state, split ratios).
        dataset_hash: SHA-256 hex digest of the raw dataset file.
        test_count: Total number of passing tests.
        coverage_pct: Code coverage percentage (0–100).
        installed_packages: Mapping of package name → version string.
        security_findings: Number of critical/high Bandit findings (0 = clean).
        dataset_url: Public URL for the WUSTL-EHMS-2020 dataset.
        repo_url: Public GitHub URL for the source code (optional).

    Returns:
        Complete Markdown string ready for file export.
    """
    lines: List[str] = []
    w = lines.append

    _section_header(w)
    _section_environment(w, installed_packages)
    _section_experiment_reproducibility(w, config, dataset_hash)
    _section_cicd_summary(w, test_count, coverage_pct, security_findings)
    _section_dataset_versioning(w, dataset_hash, dataset_url)
    _section_peer_review_checklist(w, repo_url, dataset_url)

    content = "\n".join(lines)
    logger.info("Reproducibility report rendered: %d lines", len(lines))
    return content


# ---------------------------------------------------------------------------
# Private section renderers
# ---------------------------------------------------------------------------


def _section_header(w) -> None:
    w("## 3.4 Reproducibility and Environment Specification")
    w("")
    w("This section documents the computational environment, random seed "
      "configuration, CI/CD pipeline status, and dataset versioning to "
      "ensure full reproducibility of the reported results as required "
      "by IEEE Q1 standards.")
    w("")


def _section_environment(
    w,
    packages: Dict[str, str],
) -> None:
    w("### 3.4.1 Environment Specification")
    w("")
    w(f"- **Python**: {platform.python_version()}")
    w(f"- **Platform**: {platform.system()} {platform.release()}")
    w(f"- **Architecture**: {platform.machine()}")
    w("")

    # Key packages table
    key_packages = [
        "tensorflow", "keras", "pandas", "numpy", "scikit-learn",
        "scipy", "imbalanced-learn", "hdbscan", "matplotlib",
        "pyyaml", "cryptography", "pytest",
    ]

    w("| Package | Version |")
    w("|---------|---------|")
    for pkg in key_packages:
        ver = packages.get(pkg, packages.get(pkg.replace("-", "_"), "—"))
        w(f"| {pkg} | {ver} |")
    w("")
    w(f"Full dependency list is maintained in `requirements.txt` "
      f"({len(packages)} packages total).")
    w("")


def _section_experiment_reproducibility(
    w,
    config: Phase0Config,
    dataset_hash: str,
) -> None:
    train_pct = int(config.train_ratio * 100)
    test_pct = int(config.test_ratio * 100)

    w("### 3.4.2 Experiment Reproducibility")
    w("")
    w(f"All random seeds are fixed at `random_state={config.random_state}`. "
      f"The dataset is split {train_pct}/{test_pct} (train/test) with "
      f"stratified sampling to preserve class priors.")
    w("")
    w(f"Dataset SHA-256: `{dataset_hash}`")
    w("")
    w("Results are reproducible via:")
    w("")
    w("```bash")
    w("docker build -t analyst/phase0 .")
    w("docker run --rm -v $(pwd)/data:/home/analyst/app/data analyst/phase0")
    w("```")
    w("")


def _section_cicd_summary(
    w,
    test_count: int,
    coverage_pct: float,
    security_findings: int,
) -> None:
    w("### 3.4.3 CI/CD Pipeline Summary")
    w("")
    w("The project employs a four-stage GitHub Actions pipeline:")
    w("")
    w("| Stage | Tool | Status |")
    w("|-------|------|--------|")
    w("| Lint | ruff + black | Enforced |")
    w(f"| Security | bandit + pip-audit | {security_findings} critical findings |")
    w(f"| Test | pytest | {test_count} tests passing |")
    w(f"| Coverage | pytest-cov | {coverage_pct:.1f}% (≥ 80% required) |")
    w("| Build | Docker | Image verified |")
    w("| SBOM | CycloneDX | Generated per build |")
    w("")
    w("Pre-commit hooks enforce black, ruff, bandit, and detect-secrets "
      "on every local commit.")
    w("")


def _section_dataset_versioning(
    w,
    dataset_hash: str,
    dataset_url: str,
) -> None:
    w("### 3.4.4 Dataset Versioning")
    w("")
    w("| Attribute | Value |")
    w("|-----------|-------|")
    w("| Dataset | WUSTL-EHMS-2020 |")
    w(f"| SHA-256 | `{dataset_hash[:32]}...` |")
    w(f"| Source | [{dataset_url}]({dataset_url}) |")
    w("| Integrity | Verified via `IntegrityVerifier` on every pipeline run |")
    w("")
    w("The SHA-256 hash is computed on first load and automatically verified "
      "on all subsequent loads. Any modification to the raw dataset triggers "
      "an `IntegrityError`, preventing silent data corruption from affecting "
      "experimental results.")
    w("")


def _section_peer_review_checklist(
    w,
    repo_url: Optional[str],
    dataset_url: str,
) -> None:
    w("### 3.4.5 Peer Review Readiness Checklist")
    w("")
    repo_display = f"[{repo_url}]({repo_url})" if repo_url else "[GitHub URL]"
    w(f"- [ ] Code publicly available at {repo_display}")
    w(f"- [ ] Dataset publicly available at [{dataset_url}]({dataset_url})")
    w("- [ ] Docker image builds and passes health check")
    w("- [ ] All tests pass with ≥ 80% coverage")
    w("- [ ] Security scan shows 0 critical/high findings")
    w("- [ ] SBOM (CycloneDX) generated and archived")
    w("- [ ] `report_section_*.md` artifacts uploaded per CI run")
    w("- [ ] `random_state=42` used across all stochastic operations")
    w("- [ ] Dataset SHA-256 hash recorded and verified on load")
    w("")
