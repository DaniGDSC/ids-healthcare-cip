"""Sentinel component library — pure HTML-string helpers.

Each function returns a string that the caller passes through
`st.markdown(..., unsafe_allow_html=True)`. No business logic; these only
translate alert data into the prototype's markup classes (see
`docs/sentinel_dashboard.html` for the design contract and
`sentinel_theme.py` for the class definitions).
"""
from __future__ import annotations

from html import escape
from typing import Iterable, Mapping


_TIER_NORMALIZE = {
    "CRITICAL": "critical",
    "HIGH": "high",
    "MEDIUM": "medium",
    "LOW": "low",
    "critical": "critical",
    "high": "high",
    "medium": "medium",
    "low": "low",
}


def _tier_key(tier: str | None) -> str:
    if not tier:
        return "low"
    return _TIER_NORMALIZE.get(tier, "low")


def render_tier_glyph(tier: str, size_px: int = 10) -> str:
    """Shape-coded glyph for tier — colorblind-safe per prototype L82-107."""
    cls = _tier_key(tier)
    return (
        f'<span class="glyph glyph-{cls}" '
        f'style="width:{size_px}px;height:{size_px}px;"></span>'
    )


def render_floor_badge(invariant_name: str = "floor-elevated") -> str:
    """Prototype L504-507 / L670-673."""
    safe = escape(invariant_name)
    return (
        '<span class="floor-badge">'
        '<svg width="8" height="8" viewBox="0 0 8 8" fill="none">'
        '<path d="M4 1L7 5H1L4 1Z" fill="currentColor"/></svg>'
        f'{safe}</span>'
    )


def render_alert_row(
    *,
    alert_id: str,
    title: str,
    subtitle: str,
    tier: str,
    age: str,
    owner: str | None = None,
    floor_elevated: bool = False,
    active: bool = False,
) -> str:
    """Single row in the alert queue (prototype L494-540).

    Layout: glyph + (title / subtitle / [floor-badge | age · owner]) + id.
    """
    glyph = render_tier_glyph(tier, size_px=10)
    active_cls = " active" if active else ""
    floor = render_floor_badge() if floor_elevated else ""
    owner_html = f" · {escape(owner)}" if owner else ""
    return (
        f'<div class="alert-row{active_cls}" data-alert-id="{escape(alert_id)}">'
        f'  <div style="display:flex;gap:10px;align-items:flex-start;">'
        f'    <span style="margin-top:6px;">{glyph}</span>'
        f'    <div style="flex:1;min-width:0;">'
        f'      <div style="display:flex;justify-content:space-between;gap:8px;align-items:baseline;">'
        f'        <span class="row-title">{escape(title)}</span>'
        f'        <span class="row-id">{escape(alert_id)}</span>'
        f'      </div>'
        f'      <div class="row-meta">{escape(subtitle)}</div>'
        f'      <div style="display:flex;justify-content:space-between;align-items:center;margin-top:6px;">'
        f'        <span>{floor}</span>'
        f'        <span class="row-id">{escape(age)}{owner_html}</span>'
        f'      </div>'
        f'    </div>'
        f'  </div>'
        f'</div>'
    )


def render_tier_count_tile(tier: str, count: int) -> str:
    """One of four tier-count tiles at the top of the queue (prototype L446-475)."""
    cls = _tier_key(tier)
    label = {"critical": "Crit", "high": "High", "medium": "Med", "low": "Low"}.get(cls, cls)
    return (
        f'<div style="padding:8px;border-radius:4px;text-align:center;background:var(--tier-{cls}-bg);">'
        f'  <div class="stat-num" style="color:var(--tier-{cls});font-size:1.5rem;">{count}</div>'
        f'  <div style="display:flex;justify-content:center;align-items:center;gap:4px;margin-top:2px;">'
        f'    {render_tier_glyph(tier, size_px=8)}'
        f'    <span style="font-size:9px;font-weight:500;letter-spacing:0.08em;text-transform:uppercase;color:var(--tier-{cls});">{label}</span>'
        f'  </div>'
        f'</div>'
    )


def render_calibration_bar(
    value: float,
    color_var: str = "--accent",
    with_ticks: bool = False,
) -> str:
    """Filled-percentage bar with optional quartile ticks (prototype L700-705)."""
    pct = max(0.0, min(1.0, value)) * 100
    ticks = ""
    if with_ticks:
        ticks = "".join(
            f'<div class="calibration-tick" style="left:{p}%;"></div>'
            for p in (25, 50, 75)
        )
    return (
        '<div class="calibration-bar">'
        f'  <div class="calibration-fill" style="width:{pct:.1f}%;background:var({color_var});"></div>'
        f'  {ticks}'
        '</div>'
    )


def render_metric_with_bar(
    label: str,
    value: float | str,
    sublabel: str = "",
    color_var: str = "--accent",
    bar_value: float | None = None,
    with_ticks: bool = False,
) -> str:
    """Composite-risk component breakdown cell (prototype L693-737)."""
    bar = ""
    if bar_value is not None:
        bar = render_calibration_bar(bar_value, color_var=color_var, with_ticks=with_ticks)
    val_html = f"{value:.2f}" if isinstance(value, (int, float)) else escape(str(value))
    sub_html = (
        f'<span class="font-mono" style="font-size:10px;color:var(--text-tertiary);"> {escape(sublabel)}</span>'
        if sublabel else ""
    )
    return (
        '<div>'
        f'  <div style="font-size:10px;font-weight:500;letter-spacing:0.08em;text-transform:uppercase;color:var(--text-tertiary);margin-bottom:6px;">{escape(label)}</div>'
        f'  <div style="display:flex;align-items:baseline;gap:8px;margin-bottom:4px;">'
        f'    <span class="font-mono" style="font-size:1.125rem;font-weight:500;font-variant-numeric:tabular-nums;">{val_html}</span>'
        f'    {sub_html}'
        f'  </div>'
        f'  {bar}'
        '</div>'
    )


def render_factor_row(
    label: str,
    sublabel: str,
    weight_pct: int,
    contribution: float,
    negative: bool = False,
) -> str:
    """SHAP top-N factor row (prototype L751-815)."""
    sign = "+" if contribution >= 0 and not negative else ""
    neg_cls = " negative" if negative else ""
    val_color = "var(--tier-high)" if negative else "var(--text-primary)"
    return (
        '<div class="factor-row">'
        '  <div>'
        f'    <div class="factor-label">{escape(label)}</div>'
        f'    <div class="factor-sublabel">{escape(sublabel)}</div>'
        '  </div>'
        '  <div class="factor-bar">'
        f'    <div class="factor-bar-fill{neg_cls}" style="width:{weight_pct}%;"></div>'
        '  </div>'
        f'  <span class="factor-value" style="color:{val_color};">{sign}{contribution:.2f}</span>'
        '</div>'
    )


def render_timeline_item(
    kind: str,
    label: str,
    timestamp: str,
    body: str,
    is_last: bool = False,
) -> str:
    """Audit-trail timeline item (prototype L961-991). `kind` ∈ {system, human}."""
    safe_kind = "human" if kind == "human" else "system"
    extra_style = " style=\"padding-bottom:0;\"" if is_last else ""
    return (
        f'<div class="timeline-item {safe_kind}"{extra_style}>'
        '  <div class="tl-head">'
        f'    <span class="tl-label">{escape(label)}</span>'
        f'    <span class="tl-time">{escape(timestamp)}</span>'
        '  </div>'
        f'  <div class="tl-body">{escape(body)}</div>'
        '</div>'
    )


def render_stat_num(
    value: str,
    label: str,
    color_var: str = "--text-primary",
    size: str = "lg",
) -> str:
    """Big serif numeric (prototype L448 / L687). `size` ∈ {sm, md, lg}."""
    size_cls = {"sm": "stat-num-sm", "md": "stat-num", "lg": "stat-num-lg"}.get(size, "stat-num")
    label_html = ""
    if label:
        label_html = (
            f'<div style="font-size:10px;font-weight:500;letter-spacing:0.08em;'
            f'text-transform:uppercase;color:var(--text-tertiary);margin-bottom:4px;">{escape(label)}</div>'
        )
    return (
        '<div>'
        f'  {label_html}'
        f'  <div class="stat-num {size_cls}" style="color:var({color_var});">{escape(value)}</div>'
        '</div>'
    )


def render_card(label: str, body_html: str) -> str:
    """Generic surface-1 card with uppercase label header (prototype L823-852)."""
    return (
        '<div class="sentinel-card">'
        f'  <div class="card-label">{escape(label)}</div>'
        f'  {body_html}'
        '</div>'
    )


def render_status_strip(metrics: Mapping[str, str]) -> str:
    """Fixed-position footer (prototype L996-1019).

    Expected keys (any subset; missing ones are skipped):
      `system`, `p95_ms`, `threshold`, `drift`, `last_calibration`,
      `n_val`, `fnr_delta`, `build`
    """
    left_parts: list[str] = []
    if "system" in metrics:
        left_parts.append(
            f'<span><span class="pulse-live"></span> <span>{escape(metrics["system"])}</span></span>'
        )
    if "p95_ms" in metrics:
        left_parts.append(
            f'<span>Module 4 p95 <span class="live">{escape(metrics["p95_ms"])}</span></span>'
        )
    if "threshold" in metrics:
        left_parts.append(
            f'<span>Active threshold a_high = <span class="live">{escape(metrics["threshold"])}</span></span>'
        )
    if "drift" in metrics:
        left_parts.append(
            f'<span>Model drift <span class="live">{escape(metrics["drift"])}</span></span>'
        )
    if "last_calibration" in metrics:
        left_parts.append(
            f'<span>Last calibration <span class="live">{escape(metrics["last_calibration"])}</span></span>'
        )
    right_parts: list[str] = []
    if "n_val" in metrics:
        right_parts.append(f'<span>n=<span class="live">{escape(metrics["n_val"])}</span> val set</span>')
    if "fnr_delta" in metrics:
        right_parts.append(
            f'<span>FNR_crit_Δ_max = <span class="live">{escape(metrics["fnr_delta"])}</span></span>'
        )
    if "build" in metrics:
        right_parts.append(f'<span>Build <span class="live">{escape(metrics["build"])}</span></span>')

    left = '<span>·</span>'.join(left_parts) if left_parts else ""
    right = '<span>·</span>'.join(right_parts) if right_parts else ""
    return (
        '<div class="sentinel-status-strip">'
        f'  <div class="strip-group">{left}</div>'
        f'  <div class="strip-group">{right}</div>'
        '</div>'
    )


def render_tier_header(tier: str, count: int) -> str:
    """Section header above each tier group in the queue (prototype L488-492)."""
    cls = _tier_key(tier)
    label = {"critical": "Critical", "high": "High", "medium": "Medium", "low": "Low"}.get(cls, cls.title())
    return (
        '<div class="tier-header" style="padding:12px 16px 6px;display:flex;align-items:center;gap:8px;">'
        f'  {render_tier_glyph(tier, size_px=8)}'
        f'  <span>{label} · {count}</span>'
        '  <div style="flex:1;height:1px;background:var(--border);"></div>'
        '</div>'
    )


def render_investigation_header(
    alert_id: str,
    tier: str,
    title: str,
    subtitle_html: str,
    composite_risk: float,
    raw_risk: float | None = None,
    floor_delta: float | None = None,
    floor_elevated: bool = False,
    invariant_label: str = "Invariant 2",
) -> str:
    """Top of the investigation column (prototype L662-690)."""
    cls = _tier_key(tier)
    tier_label = {"critical": "Critical", "high": "High", "medium": "Medium", "low": "Low"}.get(cls, cls.title())
    floor_html = ""
    if floor_elevated:
        suffix = f" · floor-elevated" if floor_elevated else ""
        tier_label = tier_label + suffix
        floor_html = (
            f'<span class="floor-badge" style="margin-left:4px;">'
            '<svg width="8" height="8" viewBox="0 0 8 8" fill="none">'
            '<path d="M4 1L7 5H1L4 1Z" fill="currentColor"/></svg>'
            f'{escape(invariant_label)}</span>'
        )
    risk_sub = ""
    if raw_risk is not None and floor_delta is not None:
        sign = "↑" if floor_delta >= 0 else "↓"
        risk_sub = (
            f'<div class="font-mono" style="font-size:11px;color:var(--text-tertiary);">'
            f'raw {raw_risk:.2f} · {sign}{abs(floor_delta):.2f} floor</div>'
        )
    return (
        '<div style="padding:24px 32px 20px;border-bottom:1px solid var(--border);">'
        '  <div style="display:flex;justify-content:space-between;gap:24px;align-items:flex-start;">'
        '    <div style="flex:1;">'
        '      <div style="display:flex;align-items:center;gap:12px;margin-bottom:8px;">'
        f'        {render_tier_glyph(tier, size_px=14)}'
        f'        <span class="font-mono" style="font-size:12px;text-transform:uppercase;letter-spacing:0.08em;color:var(--tier-{cls});">{escape(tier_label)}</span>'
        '        <span class="font-mono" style="font-size:11px;color:var(--text-tertiary);">·</span>'
        f'        <span class="font-mono" style="font-size:12px;color:var(--text-tertiary);">{escape(alert_id)}</span>'
        f'        {floor_html}'
        '      </div>'
        f'      <h1 class="font-display" style="font-size:2.25rem;line-height:1.05;margin:0 0 8px;letter-spacing:-0.025em;color:var(--text-primary);">{escape(title)}</h1>'
        f'      <p style="font-size:0.875rem;max-width:620px;color:var(--text-secondary);margin:0;">{subtitle_html}</p>'
        '    </div>'
        '    <div style="text-align:right;flex-shrink:0;">'
        '      <div style="font-size:10px;font-weight:500;letter-spacing:0.08em;text-transform:uppercase;color:var(--text-tertiary);margin-bottom:4px;">Composite risk</div>'
        f'      <div class="stat-num stat-num-lg" style="color:var(--tier-{cls});margin-bottom:4px;">{composite_risk:.2f}</div>'
        f'      {risk_sub}'
        '    </div>'
        '  </div>'
        '</div>'
    )


def render_actions_disclaimer() -> str:
    """One-liner under the action buttons reminding ops of the no-auto invariant."""
    return (
        '<p class="font-mono" style="font-size:10px;margin-top:12px;line-height:1.5;color:var(--text-quaternary);">'
        'No action is auto-executed. Every decision is logged with operator, timestamp, and rationale.'
        '</p>'
    )
