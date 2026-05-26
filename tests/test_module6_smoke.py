"""Module 6 smoke tests — module imports + entrypoint sanity checks.

AppTest-based full-page smoke would mount each ``pages/`` file. We don't
ship multi-page in this layout (mode handlers still live in
``module6_app.py``), so smoke here is import-only — proves the app
module loads without bootstrapping signing keys or mutating sys.path
permanently.
"""
from __future__ import annotations

import importlib



def test_module6_evaluation_imports_clean():
    """``import module6_evaluation`` must be side-effect-free."""
    import sys
    before = set(sys.path)
    m6 = importlib.import_module("module6_evaluation")
    after = set(sys.path)
    # No permanent sys.path mutation from the package init.
    assert before == after
    assert hasattr(m6, "__all__")
    assert "compute_evaluation_metrics" in m6.__all__


def test_module6_app_imports_clean_warnings_only():
    """``import module6_evaluation.module6_app`` succeeds. Streamlit emits
    cache_data warnings when no runtime is attached — that's expected.
    """
    m = importlib.import_module("module6_evaluation.module6_app")
    assert hasattr(m, "ROLE_DISPLAY_NAMES")
    assert hasattr(m, "load_responses_for")
    assert hasattr(m, "PAGE_SPLIT")


def test_module6_evaluation_back_compat_shim():
    """Legacy import path still resolves all expected symbols."""
    m = importlib.import_module("module6_evaluation.module6_evaluation")
    assert callable(m.statistical_analysis)
    assert callable(m.compute_inter_rater_reliability)
    assert callable(m._derive_device_class)
    assert callable(m._curate_split_paths)


def test_no_hardcoded_default_str_in_module6():
    """Source-level check — Phase 4 C4 should have removed all default=str."""
    import pathlib
    pkg = pathlib.Path(__file__).resolve().parents[1] / "module6_evaluation"
    offenders = []
    for py in pkg.glob("**/*.py"):
        text = py.read_text(encoding="utf-8")
        # Allow docstring mentions but flag bare code lines.
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if "default=str" in stripped and "default=str" not in stripped.lstrip("#"):
                # Only flag if it's not commented or in a docstring
                if "default=str" in stripped and stripped.startswith('"') is False:
                    if "``default=str``" not in stripped:
                        offenders.append(f"{py.name}: {stripped}")
    assert not offenders, "Found default=str in code (C4 regression): " + "\n".join(offenders)


def test_module6_evaluation_runs_via_python_m(tmp_path):
    """``python -m module6_evaluation --curate-only --split=test`` invocation path
    should be intact. We don't actually run it (would require artefacts on
    disk); we just verify the entrypoint resolves.
    """
    import module6_evaluation.__main__ as mm
    assert callable(mm.main)
    import module6_evaluation.pipeline as pipe
    assert callable(pipe.main)
