"""Module 4 example_explanations — Y3 fix verification.

Y3 latent bug: ``generate_example_explanations`` previously hardcoded
``data/phase2/risk_scores/risk_scores.npz`` which doesn't exist at that
path. Module 3 produces the file at ``results/reports/risk_scores.npz``.
Tests here verify the path resolution now goes through ``common.split_paths``.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from module4_explanations.example_explanations import (
    _load_risk_scores,
    generate_example_explanations,
)


# ── Y3 fix: canonical path resolution ───────────────────────────────


def test_load_risk_scores_uses_split_paths_resolver(monkeypatch, tmp_path):
    """_load_risk_scores must consult common.split_paths.risk_scores()."""
    fake_path = tmp_path / "fake_risk_scores.npz"
    np.savez(fake_path, R=np.array([0.1, 0.5, 0.9]))

    from common import split_paths as sp
    monkeypatch.setattr(sp, "risk_scores", lambda split: fake_path)

    out = _load_risk_scores("test")
    assert "R" in out
    np.testing.assert_array_almost_equal(out["R"], [0.1, 0.5, 0.9])


def test_load_risk_scores_returns_empty_when_missing(monkeypatch, tmp_path):
    """Missing file → empty dict with WARNING (not silent FileNotFoundError)."""
    missing = tmp_path / "nope.npz"
    from common import split_paths as sp
    monkeypatch.setattr(sp, "risk_scores", lambda split: missing)
    out = _load_risk_scores("test")
    assert out == {}


def test_load_risk_scores_no_hardcoded_legacy_path():
    """Module no longer references the old hardcoded path in code (the
    docstring still mentions it for historical context, but the code
    must not use it). We grep the AST for any string literal containing
    the legacy substring outside of module/function docstrings.
    """
    import ast
    src_path = Path("module4_explanations/example_explanations.py")
    tree = ast.parse(src_path.read_text())
    # Collect docstring node ids so we can exclude them
    docstring_ids = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            ds = ast.get_docstring(node, clean=False)
            if ds and node.body and isinstance(node.body[0], ast.Expr):
                docstring_ids.add(id(node.body[0].value))
    # Now walk Constant string nodes that aren't docstrings
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) in docstring_ids:
                continue
            assert "data/phase2/risk_scores" not in node.value, (
                f"Y3 regression: legacy path appears as a non-docstring "
                f"string literal: {node.value!r}"
            )


# ── generate_example_explanations smoke ─────────────────────────────


def test_generate_example_explanations_picks_attack_samples(monkeypatch, tmp_path):
    """Picks 5 alerts: top-2 confidence + 2 categories + 1 borderline."""
    from common import split_paths as sp
    monkeypatch.setattr(sp, "risk_scores", lambda split: tmp_path / "missing.npz")

    n = 20
    rng = np.random.default_rng(0)
    all_shap = {"xgboost": rng.standard_normal((n, 5))}
    all_preds = {
        "xgboost": {
            "y_pred": np.array([1] * 10 + [0] * 10),
            "y_proba": np.concatenate([np.linspace(0.5, 0.95, 10), np.zeros(10)]),
        },
        "random_forest": {
            "y_pred": np.zeros(n, dtype=int),
            "y_proba": np.zeros(n),
        },
        "decision_tree": {
            "y_pred": np.zeros(n, dtype=int),
            "y_proba": np.zeros(n),
        },
    }
    dae_preds = {"y_pred": np.zeros(n, dtype=int), "reconstruction_error": np.zeros(n)}
    weighted_err = rng.random((n, 5))
    feat_names = ["DIntPkt", "SrcLoad", "Pulse_Rate", "TotBytes", "ST"]
    y_test = np.array([1] * 10 + [0] * 10)
    attack_cats = np.array(["Spoofing"] * 5 + ["Data Alteration"] * 5 + ["normal"] * 10)

    risk_levels = np.array(["LOW"] * n)
    examples = generate_example_explanations(
        all_shap, all_preds, dae_preds, weighted_err,
        feat_names, y_test, attack_cats, risk_levels, output_dir=tmp_path,
    )
    # At most 5 examples
    assert len(examples) <= 5
    # All examples have the expected shape
    for ex in examples:
        assert "sample_index" in ex
        assert "ground_truth" in ex
        assert "attack_category" in ex
        assert "views" in ex
        assert set(ex["views"].keys()) == {"clinician", "analyst", "administrator"}


def test_generate_example_explanations_includes_risk_score_when_available(
    monkeypatch, tmp_path,
):
    """Y3 fix: when risk_scores file exists, examples carry non-zero R."""
    n = 12
    # Stage a synthetic risk_scores file
    fake_risk_path = tmp_path / "risk_scores.npz"
    np.savez(
        fake_risk_path,
        R=np.linspace(0.1, 0.9, n),
        c_detect=np.full(n, 0.5),
        d_crit=np.full(n, 0.4),
        s_data=np.full(n, 0.3),
        d_clinical_tier=np.full(n, 0.6),
    )

    from common import split_paths as sp
    monkeypatch.setattr(sp, "risk_scores", lambda split: fake_risk_path)

    rng = np.random.default_rng(0)
    all_shap = {"xgboost": rng.standard_normal((n, 5))}
    all_preds = {
        "xgboost": {
            "y_pred": np.array([1] * 6 + [0] * 6),
            "y_proba": np.concatenate([np.linspace(0.6, 0.95, 6), np.zeros(6)]),
        },
        "random_forest": {"y_pred": np.zeros(n, dtype=int), "y_proba": np.zeros(n)},
        "decision_tree": {"y_pred": np.zeros(n, dtype=int), "y_proba": np.zeros(n)},
    }
    dae_preds = {"y_pred": np.zeros(n, dtype=int), "reconstruction_error": np.zeros(n)}
    weighted_err = rng.random((n, 5))
    feat_names = ["DIntPkt", "SrcLoad", "Pulse_Rate", "TotBytes", "ST"]
    y_test = np.array([1] * 6 + [0] * 6)
    attack_cats = np.array(["Spoofing"] * 6 + ["normal"] * 6)

    risk_levels = np.array(["HIGH"] * n)
    examples = generate_example_explanations(
        all_shap, all_preds, dae_preds, weighted_err,
        feat_names, y_test, attack_cats, risk_levels, output_dir=tmp_path,
    )

    # At least one example should have non-zero risk_score in the
    # administrator view (proves Y3 fix loaded the file)
    admin_risk_scores = [
        ex["views"]["administrator"]["content"]["risk_score"]
        for ex in examples
    ]
    assert any(r > 0 for r in admin_risk_scores), (
        "Y3 fix: at least one example should carry a non-zero risk_score from "
        "the loaded npz, but all were zero — risk_scores path may be broken."
    )


def test_generate_example_explanations_writes_json(monkeypatch, tmp_path):
    """example_explanations.json is produced + JSON-parseable."""
    from common import split_paths as sp
    monkeypatch.setattr(sp, "risk_scores", lambda split: tmp_path / "missing.npz")

    n = 8
    rng = np.random.default_rng(0)
    all_shap = {"xgboost": rng.standard_normal((n, 5))}
    all_preds = {
        "xgboost": {
            "y_pred": np.array([1] * 4 + [0] * 4),
            "y_proba": np.concatenate([np.linspace(0.5, 0.9, 4), np.zeros(4)]),
        },
        "random_forest": {"y_pred": np.zeros(n, dtype=int), "y_proba": np.zeros(n)},
        "decision_tree": {"y_pred": np.zeros(n, dtype=int), "y_proba": np.zeros(n)},
    }
    dae_preds = {"y_pred": np.zeros(n, dtype=int), "reconstruction_error": np.zeros(n)}
    weighted_err = rng.random((n, 5))
    feat_names = ["DIntPkt", "SrcLoad", "Pulse_Rate", "TotBytes", "ST"]
    y_test = np.array([1] * 4 + [0] * 4)
    attack_cats = np.array(["Spoofing"] * 4 + ["normal"] * 4)

    risk_levels = np.array(["LOW"] * n)
    generate_example_explanations(
        all_shap, all_preds, dae_preds, weighted_err,
        feat_names, y_test, attack_cats, risk_levels, output_dir=tmp_path,
    )

    out_path = tmp_path / "example_explanations.json"
    assert out_path.exists()
    # JSON must be valid (no default=str silent coercion)
    data = json.loads(out_path.read_text())
    assert isinstance(data, list)
