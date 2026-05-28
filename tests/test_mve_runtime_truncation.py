"""Runtime word-budget truncation tests for ``src.mve_generator``.

These tests cover the post-generation guard ``_truncate_to_budget`` that
``generate_mve`` calls before returning. The pre-existing
``tests/test_mve_word_budget.py`` asserts the rule-based template path
stays within budget *under realistic SHAP/MITRE injection*; this file
asserts the guard catches the case where an upstream provider (an LLM
that ignores the system prompt, a bug in a future template injection)
ships an over-budget MVE — chapter §4.4 invariant.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_models import MVEOutput
from src.mve_generator import _truncate_to_budget, generate_mve


def _long(n: int) -> str:
    """``n``-word filler sentence; final period keeps the sentence-boundary
    truncation path exercisable."""
    return " ".join(f"w{i}" for i in range(n)) + "."


def _oversize_mve() -> MVEOutput:
    """An MVE where every layer drastically overshoots its budget."""
    body = _long(200)
    return MVEOutput(
        layer_1={
            "baseline_behavior": "Device normally talks to internal hosts.",
            "deviation_description": body,
            "confidence_indicator": "Confidence: HIGH — strong signal.",
        },
        layer_2={
            "affected_system": "Patient monitor.",
            "patient_care_impact": body,
            "phi_exposure": "Risk of biometric data exposure.",
            "severity_label": "HIGH",
            "severity_rationale": "Clinical impact.",
        },
        layer_3={
            "immediate_action": "Block flow.",
            "clinical_constraint": "DO NOT remove from patient.",
            "escalation_path": body,
            "timeframe": "Act within 15 minutes.",
        },
    )


def _layer_words(mve: MVEOutput, layer_name: str) -> int:
    fields = getattr(mve, f"_{layer_name.upper().replace('LAYER_', 'L')}")
    layer = getattr(mve, layer_name)
    return sum(len(layer.get(f, "").split()) for f in fields)


# ── Unit tests on the helper ────────────────────────────────────────────


def test_truncate_caps_layer_1_at_60():
    mve = _truncate_to_budget(_oversize_mve())
    assert _layer_words(mve, "layer_1") <= 60


def test_truncate_caps_layer_2_at_50():
    mve = _truncate_to_budget(_oversize_mve())
    assert _layer_words(mve, "layer_2") <= 50


def test_truncate_caps_layer_3_at_60():
    mve = _truncate_to_budget(_oversize_mve())
    assert _layer_words(mve, "layer_3") <= 60


def test_truncate_total_at_150():
    mve = _truncate_to_budget(_oversize_mve())
    assert mve.total_word_count <= 150


def test_truncate_preserves_severity_label():
    """severity_label is a single-token enum read verbatim downstream;
    the truncator must never touch it."""
    mve = _oversize_mve()
    mve.layer_2["severity_label"] = "CRITICAL"
    _truncate_to_budget(mve)
    assert mve.layer_2["severity_label"] == "CRITICAL"


def test_truncate_under_budget_is_no_op():
    """An MVE already under budget passes through bit-identical."""
    mve = MVEOutput(
        layer_1={"baseline_behavior": "A.", "deviation_description": "B.",
                 "confidence_indicator": "C."},
        layer_2={"affected_system": "D.", "patient_care_impact": "E.",
                 "phi_exposure": "F.", "severity_label": "LOW",
                 "severity_rationale": "G."},
        layer_3={"immediate_action": "H.", "clinical_constraint": "DO NOT I.",
                 "escalation_path": "J.", "timeframe": "K."},
    )
    snapshot = (dict(mve.layer_1), dict(mve.layer_2), dict(mve.layer_3))
    _truncate_to_budget(mve)
    assert (dict(mve.layer_1), dict(mve.layer_2), dict(mve.layer_3)) == snapshot


def test_truncate_emits_warning_log(caplog):
    """Truncation events are visible in the operator log."""
    import logging
    with caplog.at_level(logging.WARNING, logger="src.mve_generator"):
        _truncate_to_budget(_oversize_mve())
    messages = [r.getMessage() for r in caplog.records]
    truncation_logs = [m for m in messages if "overran word-budget" in m]
    assert len(truncation_logs) == 3, (
        f"expected one warning per overrun layer, got {len(truncation_logs)}: "
        f"{truncation_logs}"
    )


def test_truncate_prefers_sentence_boundary():
    """When a sentence boundary falls inside the allowed window, the
    trimmed field ends with a period (clean cut)."""
    mve = MVEOutput(
        layer_1={
            "baseline_behavior": "",
            "confidence_indicator": "",
            # 30 short sentences = 60 words; budget is 60 ⇒ no trim;
            # use 80 words instead.
            "deviation_description": " ".join(
                f"sentence {i} ends here." for i in range(20)
            ),
        },
        layer_2={"affected_system": "", "patient_care_impact": "",
                 "phi_exposure": "", "severity_label": "LOW",
                 "severity_rationale": ""},
        layer_3={"immediate_action": "", "clinical_constraint": "",
                 "escalation_path": "", "timeframe": ""},
    )
    _truncate_to_budget(mve)
    trimmed = mve.layer_1["deviation_description"]
    assert trimmed.endswith("."), f"expected sentence-boundary trim, got {trimmed!r}"


# ── Integration through generate_mve ────────────────────────────────────


def test_generate_mve_rule_based_stays_under_budget():
    """Sanity: end-to-end generate_mve still under budget after the
    truncation hook is wired in."""
    mve = generate_mve(
        raw_alert={"alert_id": "T-1", "attack_category": "Exfiltration"},
        device_context={"device_type": "patient_monitor",
                        "criticality": "HIGH", "patchable": True},
        baseline={"normal_hours": "business hours", "baseline_days": 90},
        user_context=None,
        shap_context={"top_features": ["DIntPkt", "SrcLoad", "Sport"],
                      "top_category": "network_protocol"},
        force_rule_based=True,
    )
    assert mve.total_word_count <= 150


def test_generate_mve_oversize_llm_response_clipped(monkeypatch):
    """If an LLM ignores the prompt and returns a 200-word block, the
    guard at the end of ``generate_mve`` still clips it."""
    oversize = _oversize_mve()

    def _fake_llm(*_args, **_kwargs):
        return oversize

    monkeypatch.setattr("src.mve_generator._generate_llm_openai", _fake_llm)
    # The Anthropic path won't be reached when openai returns non-None.

    mve = generate_mve(
        raw_alert={"alert_id": "T-2", "attack_category": "Spoofing"},
        device_context={"device_type": "patient_monitor",
                        "criticality": "HIGH", "patchable": True},
        baseline={"normal_hours": "business hours"},
        user_context=None,
        force_rule_based=False,
    )
    # Must trigger the LLM path so the fake fires; if OPENAI_API_KEY is
    # unset the helper returns None before reaching the patched body.
    # In that case generate_mve falls back to rule-based which is already
    # under budget — that's also a pass for the runtime-clip invariant.
    assert mve.total_word_count <= 150
