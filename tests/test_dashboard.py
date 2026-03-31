"""Dashboard UI tests — v3 with 6 panels and RBAC."""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).resolve().parents[1] / "dashboard" / "app.py")

ROLES = [
    "IT Security Analyst",
    "Clinical IT Administrator",
    "Attending Physician",
    "Hospital Manager",
    "Regulatory Auditor",
]

ROLE_PAGE_COUNTS = {
    "IT Security Analyst": 6,
    "Clinical IT Administrator": 5,
    "Attending Physician": 4,
    "Hospital Manager": 5,
    "Regulatory Auditor": 4,
}


@pytest.fixture
def app() -> AppTest:
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    return at


class TestAppLaunch:
    def test_runs_without_exception(self, app: AppTest) -> None:
        assert not app.exception, f"App raised: {app.exception}"

    def test_has_sidebar(self, app: AppTest) -> None:
        assert len(app.sidebar.selectbox) > 0

    def test_hipaa_banner(self, app: AppTest) -> None:
        all_md = [m.value for m in app.markdown]
        assert any("research" in str(m).lower() for m in all_md)


class TestRBAC:
    def test_analyst_sees_6_pages(self, app: AppTest) -> None:
        options = app.sidebar.radio[0].options
        assert len(options) == 6

    @pytest.mark.parametrize("role,expected", list(ROLE_PAGE_COUNTS.items()))
    def test_role_page_count(self, app: AppTest, role: str, expected: int) -> None:
        app.sidebar.selectbox[0].set_value(role).run()
        assert not app.exception
        options = app.sidebar.radio[0].options
        assert len(options) == expected, f"{role} should see {expected} pages, got {len(options)}: {options}"

    @pytest.mark.parametrize("role", ROLES)
    def test_each_role_renders(self, app: AppTest, role: str) -> None:
        app.sidebar.selectbox[0].set_value(role).run()
        assert not app.exception

    def test_physician_cannot_see_performance(self, app: AppTest) -> None:
        app.sidebar.selectbox[0].set_value("Attending Physician").run()
        options = app.sidebar.radio[0].options
        assert "Performance" not in options

    def test_auditor_cannot_see_alerts(self, app: AppTest) -> None:
        app.sidebar.selectbox[0].set_value("Regulatory Auditor").run()
        options = app.sidebar.radio[0].options
        assert "Alert Feed" not in options


class TestPageNavigation:
    def test_live_monitor(self, app: AppTest) -> None:
        app.sidebar.radio[0].set_value("Live Monitor").run()
        assert not app.exception

    def test_alert_feed(self, app: AppTest) -> None:
        app.sidebar.radio[0].set_value("Alert Feed").run()
        assert not app.exception

    def test_explanations(self, app: AppTest) -> None:
        app.sidebar.radio[0].set_value("Explanations").run()
        assert not app.exception

    def test_performance(self, app: AppTest) -> None:
        app.sidebar.radio[0].set_value("Performance").run()
        assert not app.exception

    def test_stakeholder(self, app: AppTest) -> None:
        app.sidebar.radio[0].set_value("Stakeholder Intelligence").run()
        assert not app.exception

    def test_system(self, app: AppTest) -> None:
        app.sidebar.radio[0].set_value("System & Compliance").run()
        assert not app.exception


class TestSidebarControls:
    def test_role_selector_has_5_roles(self, app: AppTest) -> None:
        assert len(app.sidebar.selectbox[0].options) == 5

    def test_start_button_exists(self, app: AppTest) -> None:
        labels = [b.label for b in app.sidebar.button]
        assert any("START" in str(l) for l in labels)

    def test_stop_button_exists(self, app: AppTest) -> None:
        labels = [b.label for b in app.sidebar.button]
        assert any("STOP" in str(l) for l in labels)

    def test_reset_button_exists(self, app: AppTest) -> None:
        labels = [b.label for b in app.sidebar.button]
        assert any("RESET" in str(l) for l in labels)
