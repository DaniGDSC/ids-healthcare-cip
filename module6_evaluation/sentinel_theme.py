"""Sentinel design-system injection for the Streamlit dashboard.

One public function — `inject_theme()` — emits the full prototype token block
from `docs/sentinel_dashboard.html` (`:root` at L12-45) as a single `<style>`
plus a `<link>` to Google Fonts. Called once near the top of `main()` before
any other render.

Streamlit cannot natively express the prototype's full token set via
`config.toml`; this module is the inline-injection path (D8) forced by D1=A.
"""
from __future__ import annotations

import streamlit as st


_FONTS_LINK = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=IBM+Plex+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
"""


_TOKENS_CSS = """
:root {
  --bg: #0E0F12;
  --surface-1: #16181D;
  --surface-2: #1D2026;
  --surface-3: #24272F;
  --border-subtle: #1F2229;
  --border: #262A33;
  --border-strong: #353944;

  --text-primary: #E8E9EB;
  --text-secondary: #9CA0AB;
  --text-tertiary: #6A6F7B;
  --text-quaternary: #4A4F5A;

  --tier-low: #5B8FB9;
  --tier-low-bg: rgba(91, 143, 185, 0.08);
  --tier-medium: #D4A445;
  --tier-medium-bg: rgba(212, 164, 69, 0.08);
  --tier-high: #E07A5F;
  --tier-high-bg: rgba(224, 122, 95, 0.10);
  --tier-critical: #C53030;
  --tier-critical-bg: rgba(197, 48, 48, 0.10);

  --accent: #7BA7BC;
  --accent-bg: rgba(123, 167, 188, 0.10);
  --success: #5F9E7B;
  --success-bg: rgba(95, 158, 123, 0.10);
  --warning: #D4A445;
  --neutral: #6A6F7B;
}
"""


_BASE_CSS = """
* { -webkit-font-smoothing: antialiased; -moz-osx-font-smoothing: grayscale; }

html, body, [data-testid="stAppViewContainer"], .stApp {
  background: var(--bg) !important;
  color: var(--text-primary) !important;
  font-family: 'IBM Plex Sans', system-ui, sans-serif !important;
  font-feature-settings: "ss01", "ss02", "cv11";
}

[data-testid="stHeader"] {
  background: var(--surface-1) !important;
  border-bottom: 1px solid var(--border) !important;
}

.block-container {
  padding-top: 1rem !important;
  padding-bottom: 5rem !important;
  max-width: none !important;
}

[data-testid="stSidebar"] {
  background: var(--surface-1) !important;
  border-right: 1px solid var(--border) !important;
}

[data-testid="stSidebar"] * {
  color: var(--text-secondary);
}

.font-display {
  font-family: 'Instrument Serif', Georgia, serif !important;
  font-feature-settings: "liga", "dlig";
  letter-spacing: -0.02em;
}

.font-mono {
  font-family: 'JetBrains Mono', ui-monospace, monospace !important;
  font-feature-settings: "ss01", "zero", "tnum";
}

.text-primary   { color: var(--text-primary); }
.text-secondary { color: var(--text-secondary); }
.text-tertiary  { color: var(--text-tertiary); }
.text-quaternary{ color: var(--text-quaternary); }

::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border-strong); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--text-quaternary); }
"""


_GLYPH_CSS = """
.glyph {
  display: inline-block;
  width: 10px;
  height: 10px;
  flex-shrink: 0;
  vertical-align: middle;
}
.glyph-low {
  background: var(--tier-low);
  border-radius: 50%;
}
.glyph-medium {
  background: var(--tier-medium);
  transform: rotate(45deg);
}
.glyph-high {
  background: var(--tier-high);
  clip-path: polygon(50% 0%, 100% 100%, 0% 100%);
  width: 11px;
  height: 10px;
}
.glyph-critical {
  background: var(--tier-critical);
  clip-path: polygon(25% 0%, 75% 0%, 100% 50%, 75% 100%, 25% 100%, 0% 50%);
  width: 12px;
}
.tier-header {
  font-family: 'IBM Plex Sans', sans-serif;
  font-size: 10px;
  font-weight: 600;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--text-tertiary);
}
"""


_ALERT_ROW_CSS = """
.alert-row {
  transition: background 120ms ease, border-left-color 120ms ease;
  border-left: 2px solid transparent;
  padding: 12px 16px;
  cursor: pointer;
  border-bottom: 1px solid var(--border-subtle);
}
.alert-row:hover {
  background: var(--surface-2);
}
.alert-row.active {
  background: var(--surface-2);
  border-left-color: var(--accent);
}
.alert-row .row-title {
  font-family: 'Instrument Serif', serif;
  font-size: 1rem;
  line-height: 1.1;
  letter-spacing: -0.01em;
  color: var(--text-primary);
}
.alert-row .row-meta {
  color: var(--text-secondary);
  font-size: 0.75rem;
  margin-top: 2px;
}
.alert-row .row-id {
  font-family: 'JetBrains Mono', monospace;
  font-size: 10px;
  color: var(--text-tertiary);
}
"""


_CALIBRATION_CSS = """
.calibration-bar {
  position: relative;
  height: 4px;
  background: var(--surface-3);
  border-radius: 2px;
  overflow: hidden;
  margin-top: 6px;
}
.calibration-tick {
  position: absolute;
  top: 0;
  bottom: 0;
  width: 1px;
  background: var(--border-strong);
  z-index: 2;
}
.calibration-fill {
  height: 100%;
  border-radius: 2px;
  transition: width 300ms ease;
}
"""


_FLOOR_BADGE_CSS = """
.floor-badge {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  padding: 2px 6px;
  background: var(--accent-bg);
  border: 1px solid rgba(123, 167, 188, 0.2);
  border-radius: 3px;
  color: var(--accent);
  font-family: 'JetBrains Mono', monospace;
  font-size: 10px;
  font-weight: 500;
  letter-spacing: 0.04em;
}
"""


_BTN_CSS = """
.sentinel-btn-row { display: flex; flex-direction: column; gap: 8px; }

.sentinel-btn {
  font-family: 'IBM Plex Sans', sans-serif;
  font-size: 12px;
  font-weight: 500;
  letter-spacing: 0.02em;
  padding: 8px 14px;
  border-radius: 4px;
  border: 1px solid var(--border-strong);
  background: var(--surface-2);
  color: var(--text-primary);
  cursor: pointer;
  display: inline-flex;
  align-items: center;
  gap: 8px;
  width: 100%;
  text-align: left;
}
.sentinel-btn-acknowledge {
  background: var(--success-bg);
  border-color: rgba(95, 158, 123, 0.3);
  color: var(--success);
}
.sentinel-btn-escalate {
  background: var(--tier-high-bg);
  border-color: rgba(224, 122, 95, 0.3);
  color: var(--tier-high);
}
.sentinel-btn-dismiss {
  background: transparent;
}

/* Native Streamlit button overrides — used for the three action buttons.
   We map by container key via [data-testid="stBaseButton-secondary"] under a
   keyed wrapper div. */
div[data-sentinel-action="acknowledge"] button {
  background: var(--success-bg) !important;
  border: 1px solid rgba(95, 158, 123, 0.3) !important;
  color: var(--success) !important;
}
div[data-sentinel-action="escalate"] button {
  background: var(--tier-high-bg) !important;
  border: 1px solid rgba(224, 122, 95, 0.3) !important;
  color: var(--tier-high) !important;
}
div[data-sentinel-action="dismiss"] button {
  background: transparent !important;
  border: 1px solid var(--border) !important;
  color: var(--text-secondary) !important;
}
"""


_FACTOR_ROW_CSS = """
.factor-row {
  display: grid;
  grid-template-columns: 1fr 80px 60px;
  gap: 12px;
  align-items: center;
  padding: 8px 0;
  border-bottom: 1px solid var(--border-subtle);
}
.factor-row:last-child { border-bottom: none; }
.factor-row .factor-label {
  font-size: 0.875rem;
  color: var(--text-primary);
}
.factor-row .factor-sublabel {
  font-family: 'JetBrains Mono', monospace;
  font-size: 10px;
  color: var(--text-tertiary);
  margin-top: 2px;
}
.factor-bar {
  height: 3px;
  background: var(--surface-3);
  border-radius: 2px;
  overflow: hidden;
  position: relative;
}
.factor-bar-fill {
  height: 100%;
  background: var(--accent);
  border-radius: 2px;
}
.factor-bar-fill.negative {
  background: var(--tier-high);
}
.factor-row .factor-value {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.75rem;
  text-align: right;
  font-variant-numeric: tabular-nums;
}
"""


_TIMELINE_CSS = """
.timeline-item {
  position: relative;
  padding-left: 24px;
  padding-bottom: 14px;
}
.timeline-item::before {
  content: '';
  position: absolute;
  left: 5px;
  top: 4px;
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: var(--surface-3);
  border: 2px solid var(--border-strong);
  z-index: 1;
}
.timeline-item.system::before {
  background: var(--accent);
  border-color: var(--accent);
}
.timeline-item.human::before {
  background: var(--success);
  border-color: var(--success);
}
.timeline-item:not(:last-child)::after {
  content: '';
  position: absolute;
  left: 8px;
  top: 14px;
  bottom: 0;
  width: 1px;
  background: var(--border);
}
.timeline-item .tl-head {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  margin-bottom: 2px;
}
.timeline-item .tl-label {
  font-size: 0.875rem;
  font-weight: 500;
  color: var(--text-primary);
}
.timeline-item .tl-time {
  font-family: 'JetBrains Mono', monospace;
  font-size: 10px;
  color: var(--text-tertiary);
}
.timeline-item .tl-body {
  font-size: 0.75rem;
  color: var(--text-secondary);
}
"""


_STAT_NUM_CSS = """
.stat-num {
  font-family: 'Instrument Serif', serif;
  font-feature-settings: "tnum";
  letter-spacing: -0.02em;
  line-height: 1;
  font-size: 1.5rem;
}
.stat-num-lg { font-size: 3.75rem; }
.stat-num-sm { font-size: 1.25rem; }
"""


_PULSE_CSS = """
@keyframes sentinel-pulse-soft {
  0%, 100% { opacity: 1; }
  50%      { opacity: 0.4; }
}
.pulse-live {
  animation: sentinel-pulse-soft 2.4s ease-in-out infinite;
  display: inline-block;
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: var(--success);
  vertical-align: middle;
}
"""


_ROLE_TOGGLE_CSS = """
/* Map st.pills to the prototype's role-toggle pill aesthetic. */
[data-sentinel-role-pills] [data-testid="stPills"] [role="radiogroup"] {
  display: inline-flex;
  background: var(--surface-3);
  border-radius: 4px;
  padding: 2px;
  border: 1px solid var(--border);
  gap: 0;
}
[data-sentinel-role-pills] [data-testid="stPills"] label {
  padding: 4px 12px !important;
  font-size: 11px !important;
  font-weight: 500 !important;
  letter-spacing: 0.04em !important;
  text-transform: uppercase !important;
  color: var(--text-tertiary) !important;
  background: transparent !important;
  border: none !important;
  border-radius: 3px !important;
}
[data-sentinel-role-pills] [data-testid="stPills"] label[data-checked="true"] {
  background: var(--surface-1) !important;
  color: var(--text-primary) !important;
  box-shadow: 0 1px 2px rgba(0,0,0,0.3) !important;
}
"""


_STATUS_STRIP_CSS = """
.sentinel-status-strip {
  position: fixed;
  bottom: 0;
  left: 0;
  right: 0;
  height: 36px;
  padding: 0 20px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  background: var(--surface-1);
  border-top: 1px solid var(--border);
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px;
  color: var(--text-tertiary);
  z-index: 999;
}
.sentinel-status-strip .strip-group {
  display: flex;
  align-items: center;
  gap: 14px;
}
.sentinel-status-strip .strip-dot {
  background: var(--text-primary);
}
.sentinel-status-strip span.live {
  color: var(--text-primary);
}
"""


_CARD_CSS = """
.sentinel-card {
  background: var(--surface-1);
  border: 1px solid var(--border-subtle);
  border-radius: 4px;
  padding: 16px;
}
.sentinel-card .card-label {
  font-size: 10px;
  font-weight: 500;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--text-tertiary);
  margin-bottom: 8px;
}
.sentinel-card .card-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 12px;
  margin-top: 12px;
  font-size: 0.75rem;
}
.sentinel-card .card-grid span {
  color: var(--text-tertiary);
}
"""


_COLUMNS_OVERRIDE_CSS = """
/* Tighten the three-column Triage layout. Streamlit's st.columns is
   proportional; we pin approximate target widths via per-column overrides. */
[data-sentinel-triage="root"] [data-testid="stHorizontalBlock"] {
  gap: 0 !important;
}
[data-sentinel-triage="root"] [data-testid="column"] {
  border-right: 1px solid var(--border);
  padding: 0 !important;
}
[data-sentinel-triage="root"] [data-testid="column"]:last-child {
  border-right: none;
  border-left: 1px solid var(--border);
}
"""


def inject_theme() -> None:
    """Emit the Sentinel design tokens + base CSS + component classes.

    Must be called exactly once per page render, before any markup that
    depends on the classes. Subsequent calls are harmless (Streamlit
    deduplicates identical markdown blocks) but wasted.
    """
    blocks = (
        _FONTS_LINK,
        "<style>",
        _TOKENS_CSS,
        _BASE_CSS,
        _GLYPH_CSS,
        _ALERT_ROW_CSS,
        _CALIBRATION_CSS,
        _FLOOR_BADGE_CSS,
        _BTN_CSS,
        _FACTOR_ROW_CSS,
        _TIMELINE_CSS,
        _STAT_NUM_CSS,
        _PULSE_CSS,
        _ROLE_TOGGLE_CSS,
        _STATUS_STRIP_CSS,
        _CARD_CSS,
        _COLUMNS_OVERRIDE_CSS,
        "</style>",
    )
    st.markdown("\n".join(blocks), unsafe_allow_html=True)
