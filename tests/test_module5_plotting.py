"""Module 5 plotting smoke tests — assert PNGs are produced + non-empty."""
from __future__ import annotations

import matplotlib
import pytest

matplotlib.use("Agg")

from module5_responses.plotting import (
    plot_effectiveness_by_action,
    plot_escalation_funnel,
    plot_precision_by_level,
    plot_response_distribution,
    plot_response_sankey,
)


@pytest.fixture(autouse=True)
def _charts_dir(monkeypatch, tmp_path):
    from module5_responses import plotting
    monkeypatch.setattr(plotting, "CHARTS_DIR", tmp_path)
    return tmp_path


def _records():
    return [
        {"risk_level": "HIGH", "ground_truth": "attack",
         "response": {"actions": ["log_event", "isolate_device"]}},
        {"risk_level": "HIGH", "ground_truth": "benign",
         "response": {"actions": ["log_event"]}},
        {"risk_level": "LOW", "ground_truth": "benign",
         "response": {"actions": ["log_event"]}},
    ]


def _stats():
    return {
        "alerts_by_level": {"LOW": 1, "MEDIUM": 0, "HIGH": 2, "CRITICAL": 0},
        "true_positives_by_level": {"HIGH": 1},
        "false_positives_by_level": {"HIGH": 1, "LOW": 1},
        "precision_by_level": {"HIGH": 0.5, "LOW": 0.0},
    }


def _effectiveness():
    return {
        "proportionality_analysis": [
            {"action": "isolate_device", "cost": 0.8, "precision": 0.9, "total": 10},
            {"action": "log_event", "cost": 0.1, "precision": 0.3, "total": 30},
        ],
    }


def _audit_records():
    return [
        {"risk_level": "HIGH", "recommended_actions": ["log_event", "isolate_device"],
         "simulated_outcome": {"outcome": "threat_contained", "ground_truth": "attack"}},
        {"risk_level": "LOW", "recommended_actions": ["log_event"],
         "simulated_outcome": {"outcome": "benign_logged", "ground_truth": "benign"}},
    ]


def test_plot_response_distribution_writes_png(_charts_dir):
    plot_response_distribution(_records())
    out = _charts_dir / "response_actions_by_level.png"
    assert out.exists() and out.stat().st_size > 0


def test_plot_precision_by_level_writes_png(_charts_dir):
    plot_precision_by_level(_stats())
    assert (_charts_dir / "precision_by_level.png").stat().st_size > 0


def test_plot_escalation_funnel_writes_png(_charts_dir):
    plot_escalation_funnel(_stats())
    assert (_charts_dir / "response_escalation_funnel.png").stat().st_size > 0


def test_plot_effectiveness_by_action_writes_png(_charts_dir):
    plot_effectiveness_by_action(_effectiveness())
    assert (_charts_dir / "effectiveness_by_action.png").stat().st_size > 0


def test_plot_effectiveness_empty_skips(_charts_dir):
    plot_effectiveness_by_action({"proportionality_analysis": []})
    # No file produced when there's nothing to plot.
    assert not (_charts_dir / "effectiveness_by_action.png").exists()


def test_plot_response_sankey_writes_png(_charts_dir):
    plot_response_sankey(_audit_records())
    assert (_charts_dir / "response_sankey.png").stat().st_size > 0
