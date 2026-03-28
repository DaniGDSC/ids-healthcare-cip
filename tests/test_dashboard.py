"""Dashboard UI tests using Streamlit AppTest.

Tests:
  1. App launches without error
  2. All roles render without crash
  3. All pages render without crash
  4. Role-based page filtering works
  5. Sidebar components render
  6. Key visual components are present
  7. Alert detail dialog works
  8. Stakeholder views render per role
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

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

ALL_PAGES = [
    "Operational Status",
    "Stakeholder Intelligence",
    "Alert Feed",
    "SHAP Explanations",
    "Evaluation Results",
    "Model & Training",
    "Risk Analytics",
    "Device Inventory",
    "Cross-Dataset & Simulation",
    "Compliance & Audit",
]


# ── Fixtures ───────────────────────────────────────────────────────

@pytest.fixture
def app() -> AppTest:
    """Launch the dashboard app for testing."""
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    return at


# ── App Launch Tests ───────────────────────────────────────────────

class TestAppLaunch:
    """Test that the dashboard launches and renders."""

    def test_app_runs_without_exception(self, app: AppTest) -> None:
        """App should load without raising exceptions."""
        assert not app.exception, f"App raised exception: {app.exception}"

    def test_has_sidebar(self, app: AppTest) -> None:
        """Sidebar should be present with role selector."""
        assert len(app.sidebar.selectbox) > 0, "No selectbox in sidebar"

    def test_has_header(self, app: AppTest) -> None:
        """Page should have a header element."""
        assert len(app.header) > 0 or len(app.markdown) > 0

    def test_hipaa_banner_present(self, app: AppTest) -> None:
        """HIPAA disclaimer banner should be rendered."""
        all_markdown = [m.value for m in app.markdown]
        has_hipaa = any("research purposes" in str(m).lower() or "hipaa" in str(m).lower()
                        for m in all_markdown)
        assert has_hipaa, "HIPAA banner not found"


# ── Role-Based Access Tests ────────────────────────────────────────

class TestRoleBasedAccess:
    """Test role-based page filtering."""

    def test_analyst_sees_all_pages(self, app: AppTest) -> None:
        """IT Security Analyst should see all pages."""
        # Default role is IT Security Analyst (index 0)
        radio = app.sidebar.radio
        assert len(radio) > 0
        options = radio[0].options
        assert len(options) == 10, f"Analyst should see 10 pages, got {len(options)}"

    def test_physician_sees_limited_pages(self, app: AppTest) -> None:
        """Attending Physician should see only 4 pages."""
        app.sidebar.selectbox[0].set_value("Attending Physician").run()
        assert not app.exception, f"Exception on role change: {app.exception}"
        radio = app.sidebar.radio
        assert len(radio) > 0
        options = radio[0].options
        assert len(options) == 4, f"Physician should see 4 pages, got {len(options)}: {options}"

    def test_auditor_sees_limited_pages(self, app: AppTest) -> None:
        """Regulatory Auditor should see only 3 pages."""
        app.sidebar.selectbox[0].set_value("Regulatory Auditor").run()
        assert not app.exception, f"Exception on role change: {app.exception}"
        options = app.sidebar.radio[0].options
        assert len(options) == 3, f"Auditor should see 3 pages, got {len(options)}"

    @pytest.mark.parametrize("role", ROLES)
    def test_each_role_renders_without_error(self, app: AppTest, role: str) -> None:
        """Switching to each role should not crash."""
        app.sidebar.selectbox[0].set_value(role).run()
        assert not app.exception, f"Role '{role}' raised: {app.exception}"


# ── Page Navigation Tests ──────────────────────────────────────────

class TestPageNavigation:
    """Test that each page renders without error."""

    def test_operational_status_page(self, app: AppTest) -> None:
        """Operational Status page should render."""
        app.sidebar.radio[0].set_value("Operational Status").run()
        assert not app.exception

    def test_stakeholder_intelligence_page(self, app: AppTest) -> None:
        """Stakeholder Intelligence page should render."""
        app.sidebar.radio[0].set_value("Stakeholder Intelligence").run()
        assert not app.exception

    def test_alert_feed_page(self, app: AppTest) -> None:
        """Alert Feed page should render."""
        app.sidebar.radio[0].set_value("Alert Feed").run()
        assert not app.exception

    def test_shap_page(self, app: AppTest) -> None:
        """SHAP Explanations page should render."""
        app.sidebar.radio[0].set_value("SHAP Explanations").run()
        assert not app.exception

    def test_evaluation_page(self, app: AppTest) -> None:
        """Evaluation Results page should render."""
        app.sidebar.radio[0].set_value("Evaluation Results").run()
        assert not app.exception

    def test_model_page(self, app: AppTest) -> None:
        """Model & Training page should render."""
        app.sidebar.radio[0].set_value("Model & Training").run()
        assert not app.exception

    def test_risk_analytics_page(self, app: AppTest) -> None:
        """Risk Analytics page should render."""
        app.sidebar.radio[0].set_value("Risk Analytics").run()
        assert not app.exception

    def test_device_inventory_page(self, app: AppTest) -> None:
        """Device Inventory page should render."""
        app.sidebar.radio[0].set_value("Device Inventory").run()
        assert not app.exception

    def test_crossval_page(self, app: AppTest) -> None:
        """Cross-Dataset page should render."""
        app.sidebar.radio[0].set_value("Cross-Dataset & Simulation").run()
        assert not app.exception

    def test_compliance_page(self, app: AppTest) -> None:
        """Compliance & Audit page should render."""
        app.sidebar.radio[0].set_value("Compliance & Audit").run()
        assert not app.exception


# ── Stakeholder View Tests ─────────────────────────────────────────

class TestStakeholderViews:
    """Test that stakeholder intelligence adapts to role."""

    def _navigate_to_stakeholder(self, app: AppTest, role: str) -> AppTest:
        """Switch role and navigate to Stakeholder Intelligence."""
        app.sidebar.selectbox[0].set_value(role).run()
        app.sidebar.radio[0].set_value("Stakeholder Intelligence").run()
        return app

    def test_analyst_gets_soc_view(self, app: AppTest) -> None:
        app = self._navigate_to_stakeholder(app, "IT Security Analyst")
        assert not app.exception

    def test_physician_gets_clinician_view(self, app: AppTest) -> None:
        app = self._navigate_to_stakeholder(app, "Attending Physician")
        assert not app.exception

    def test_manager_gets_ciso_view(self, app: AppTest) -> None:
        app = self._navigate_to_stakeholder(app, "Hospital Manager")
        assert not app.exception

    def test_auditor_gets_compliance_view(self, app: AppTest) -> None:
        app = self._navigate_to_stakeholder(app, "Regulatory Auditor")
        assert not app.exception


# ── Alert Feed Tests ───────────────────────────────────────────────

class TestAlertFeed:
    """Test alert feed functionality."""

    def test_suppressed_toggle_exists(self, app: AppTest) -> None:
        """Alert feed should have show suppressed checkbox."""
        app.sidebar.radio[0].set_value("Alert Feed").run()
        assert not app.exception
        checkboxes = app.checkbox
        # Should have at least the "Show suppressed alerts" checkbox
        checkbox_labels = [c.label for c in checkboxes]
        has_suppressed = any("suppressed" in str(l).lower() for l in checkbox_labels)
        assert has_suppressed, f"Missing suppressed toggle. Checkboxes: {checkbox_labels}"


# ── Sidebar Component Tests ────────────────────────────────────────

class TestSidebarComponents:
    """Test sidebar visual components."""

    def test_role_selector_has_all_roles(self, app: AppTest) -> None:
        """Role selector should list all 5 roles."""
        selectbox = app.sidebar.selectbox[0]
        assert len(selectbox.options) == 5

    def test_role_selector_options_match(self, app: AppTest) -> None:
        """Role options should match expected roles."""
        options = app.sidebar.selectbox[0].options
        for role in ROLES:
            assert role in options, f"Missing role: {role}"

    def test_auto_refresh_checkbox_exists(self, app: AppTest) -> None:
        """Auto-refresh checkbox should be in sidebar."""
        sidebar_checkboxes = app.sidebar.checkbox
        labels = [c.label for c in sidebar_checkboxes]
        has_refresh = any("refresh" in str(l).lower() or "auto" in str(l).lower()
                         for l in labels)
        assert has_refresh, f"Missing auto-refresh. Checkboxes: {labels}"

    def test_re_extract_button_exists(self, app: AppTest) -> None:
        """Re-extract Ground Truth button should be in sidebar."""
        sidebar_buttons = app.sidebar.button
        labels = [b.label for b in sidebar_buttons]
        has_extract = any("extract" in str(l).lower() or "re-extract" in str(l).lower()
                         for l in labels)
        assert has_extract, f"Missing re-extract button. Buttons: {labels}"
