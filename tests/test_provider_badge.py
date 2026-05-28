"""Path B · commit 4 — _render_provider_badge helper.

The dashboard surfaces an LLM-degradation banner when the MVE attached
to an alert came from the rule-based fallback (Mode B). This test
exercises the helper directly, capturing st.warning / st.caption calls
via monkeypatching.
"""
from __future__ import annotations

import pytest


@pytest.fixture
def capture(monkeypatch):
    """Capture streamlit warning/caption emissions from inside module6_app."""
    captured: dict[str, list[str]] = {"warning": [], "caption": []}
    from module6_evaluation import module6_app as m6

    monkeypatch.setattr(
        m6.st, "warning",
        lambda msg: captured["warning"].append(msg),
    )
    monkeypatch.setattr(
        m6.st, "caption",
        lambda msg: captured["caption"].append(msg),
    )
    return m6, captured


def test_provider_badge_rule_based_warns(capture):
    m6, captured = capture
    m6._render_provider_badge({"mve": {"provider": "rule_based"}})
    assert len(captured["warning"]) == 1
    assert "Rule-based" in captured["warning"][0]
    assert captured["caption"] == []


def test_provider_badge_openai_captions(capture):
    m6, captured = capture
    m6._render_provider_badge({"mve": {"provider": "openai"}})
    assert captured["warning"] == []
    assert len(captured["caption"]) == 1
    assert "openai" in captured["caption"][0]


def test_provider_badge_anthropic_captions(capture):
    m6, captured = capture
    m6._render_provider_badge({"mve": {"provider": "anthropic"}})
    assert captured["warning"] == []
    assert len(captured["caption"]) == 1
    assert "anthropic" in captured["caption"][0]


def test_provider_badge_legacy_mve_provider_key(capture):
    """Legacy alert records store provider at the top level instead of under mve."""
    m6, captured = capture
    m6._render_provider_badge({"mve_provider": "rule_based"})
    assert len(captured["warning"]) == 1


def test_provider_badge_no_provider_silent(capture):
    m6, captured = capture
    m6._render_provider_badge({})
    assert captured["warning"] == []
    assert captured["caption"] == []
