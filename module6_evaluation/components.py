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
    """Prototype L504-507 / L670-673.

    Tone decision (P2-16): the accent (blue-teal) is intentional and stays.
    Floor elevation is *informational* — "policy bumped this tier because
    of life-critical context" — not alarming. Using the high/critical tier
    colors here would muddle the signal (the tier itself already carries
    that color); the badge complements rather than competes. A `title`
    attribute provides hover context for operators unfamiliar with the
    convention.
    """
    safe = escape(invariant_name)
    tooltip = (
        "Tier was elevated by the Module-5 safety floor "
        "because a life-critical device is involved."
    )
    return (
        f'<span class="floor-badge" title="{escape(tooltip)}">'
        '<svg width="8" height="8" viewBox="0 0 8 8" fill="none" aria-hidden="true">'
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
    time_html = (
        f'<span class="tl-time">{escape(timestamp)}</span>' if timestamp else ""
    )
    body_html = (
        f'<div class="tl-body">{escape(body)}</div>' if body else ""
    )
    return (
        f'<div class="timeline-item {safe_kind}"{extra_style}>'
        '  <div class="tl-head">'
        f'    <span class="tl-label">{escape(label)}</span>'
        f'    {time_html}'
        '  </div>'
        f'  {body_html}'
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


def render_status_strip(
    metrics: Mapping[str, str],
    *,
    is_live: bool = True,
) -> str:
    """Fixed-position footer (prototype L996-1019).

    Expected keys (any subset; missing ones are skipped):
      `system`, `p95_ms`, `threshold`, `drift`, `last_calibration`,
      `n_val`, `fnr_delta`, `build`

    `is_live=False` swaps the pulsing dot for a static one — used on pages
    whose data is a file snapshot (e.g. the Triage Dashboard) so the
    indicator doesn't imply liveness the page doesn't have.
    """
    dot_cls = "pulse-live" if is_live else "pulse-static"
    left_parts: list[str] = []
    if "system" in metrics:
        left_parts.append(
            f'<span><span class="{dot_cls}"></span> <span>{escape(metrics["system"])}</span></span>'
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
        suffix = " · floor-elevated"
        tier_label = tier_label + suffix
        tooltip = (
            "Tier was elevated by the Module-5 safety floor "
            "because a life-critical device is involved."
        )
        floor_html = (
            f'<span class="floor-badge" title="{escape(tooltip)}" '
            f'style="margin-left:4px;">'
            '<svg width="8" height="8" viewBox="0 0 8 8" fill="none" aria-hidden="true">'
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


def render_consensus_badge(
    n_flagged: int,
    total: int,
    *,
    label: str = "Detector consensus",
) -> str:
    """Visual badge for the Module 4 detector-ensemble consensus.

    `n_flagged` is the number of detectors that flagged the sample as
    anomalous; `total` is the ensemble size (4 in the current pipeline).
    Tone is mapped to consensus strength:
      * 4/4 → success (unanimous)
      * 3/4 → accent  (strong)
      * 2/4 → tier-medium (mixed)
      * 1/4 → tier-low (weak)
      * 0/4 → text-tertiary (no flags)

    The badge renders both as N filled / M-N empty dots AND a numeric
    string so colour-blind operators retain the signal. A tooltip
    explains the threshold semantics.
    """
    n = max(0, min(int(n_flagged), int(total)))
    m = max(1, int(total))
    if n == m:
        tone = ("var(--success)", "var(--success-bg)", "rgba(95,158,123,0.3)", "unanimous")
    elif n / m >= 0.75:
        tone = ("var(--accent)", "var(--accent-bg)", "rgba(123,167,188,0.3)", "strong")
    elif n / m >= 0.5:
        tone = ("var(--tier-medium)", "var(--tier-medium-bg)", "rgba(212,164,69,0.3)", "mixed")
    elif n / m > 0:
        tone = ("var(--tier-low)", "var(--tier-low-bg)", "rgba(91,143,185,0.3)", "weak")
    else:
        tone = ("var(--text-tertiary)", "var(--surface-2)", "var(--border)", "no flags")
    fg, bg, border, descriptor = tone
    dots = ""
    for i in range(m):
        filled = i < n
        if filled:
            dots += (
                f'<span style="display:inline-block;width:8px;height:8px;'
                f'border-radius:50%;background:{fg};margin-right:3px;"></span>'
            )
        else:
            dots += (
                f'<span style="display:inline-block;width:8px;height:8px;'
                f'border-radius:50%;border:1px solid {fg};box-sizing:border-box;'
                f'margin-right:3px;opacity:0.4;"></span>'
            )
    tooltip = (
        f"{n} of {m} detectors flagged this sample as anomalous. "
        "4/4 = unanimous; 3/4 = strong; 2/4 = mixed; 1/4 = weak; 0/4 = no flags."
    )
    return (
        f'<div title="{escape(tooltip)}" style="display:inline-flex;'
        f'align-items:center;gap:10px;padding:6px 12px;border-radius:4px;'
        f'background:{bg};border:1px solid {border};color:{fg};">'
        f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:10px;'
        f'font-weight:500;letter-spacing:0.08em;text-transform:uppercase;">'
        f'{escape(label)}</span>'
        f'<span style="display:inline-flex;align-items:center;">{dots}</span>'
        f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:12px;'
        f'font-weight:500;font-variant-numeric:tabular-nums;">'
        f'{n}/{m}<span style="opacity:0.7;font-size:10px;'
        f'margin-left:6px;">{descriptor}</span></span>'
        f'</div>'
    )


def render_model_breakdown(models: Mapping[str, Mapping]) -> str:
    """Per-detector breakdown table — Module 4 `models` dict → 4-row mini-table.

    Expected shape per model:
      {prediction: 0|1, confidence: float, ...}
    The DAE model uses `reconstruction_error` instead of `confidence`; we
    surface that as the magnitude. Missing models render as a muted row
    so the operator can tell which detector didn't produce output.
    """
    if not models:
        return ""
    expected = ("xgboost", "random_forest", "decision_tree", "dae")
    display_names = {
        "xgboost":       "XGBoost",
        "random_forest": "Random Forest",
        "decision_tree": "Decision Tree",
        "dae":           "DAE (autoencoder)",
    }
    rows = ""
    for key in expected:
        info = models.get(key) or {}
        if not info:
            rows += (
                f'<div style="display:grid;grid-template-columns:140px 80px 1fr 60px;'
                f'gap:12px;align-items:center;padding:6px 0;border-bottom:1px solid '
                f'var(--border-subtle);color:var(--text-quaternary);">'
                f'<span style="font-size:0.875rem;">{escape(display_names.get(key, key))}</span>'
                f'<span class="font-mono" style="font-size:11px;">— no output —</span>'
                f'<span></span><span></span>'
                f'</div>'
            )
            continue
        pred = info.get("prediction")
        if pred == 1:
            pred_chip = (
                '<span style="display:inline-block;padding:1px 8px;border-radius:3px;'
                'background:var(--tier-high-bg);color:var(--tier-high);'
                'font-family:\'JetBrains Mono\',monospace;font-size:10px;'
                'font-weight:500;letter-spacing:0.04em;">FLAG</span>'
            )
        elif pred == 0:
            pred_chip = (
                '<span style="display:inline-block;padding:1px 8px;border-radius:3px;'
                'background:var(--success-bg);color:var(--success);'
                'font-family:\'JetBrains Mono\',monospace;font-size:10px;'
                'font-weight:500;letter-spacing:0.04em;">CLEAR</span>'
            )
        else:
            pred_chip = (
                '<span style="font-family:\'JetBrains Mono\',monospace;font-size:10px;'
                'color:var(--text-tertiary);">—</span>'
            )

        # DAE uses reconstruction_error; others use confidence in [0,1].
        if key == "dae" and "reconstruction_error" in info:
            magnitude = float(info.get("reconstruction_error", 0.0))
            # Clamp to a usable visual scale; DAE errors can be tiny.
            mag_pct = min(100, max(0, int(magnitude * 1e6))) if magnitude > 0 else 0
            value_label = f"err {magnitude:.2e}"
        else:
            magnitude = float(info.get("confidence", 0.0))
            mag_pct = int(round(min(1.0, max(0.0, magnitude)) * 100))
            value_label = f"{magnitude:.2f}"

        bar_color = "var(--tier-high)" if pred == 1 else "var(--accent)"
        rows += (
            f'<div style="display:grid;grid-template-columns:140px 80px 1fr 60px;'
            f'gap:12px;align-items:center;padding:6px 0;border-bottom:1px solid '
            f'var(--border-subtle);">'
            f'<span style="font-size:0.875rem;color:var(--text-primary);">'
            f'{escape(display_names.get(key, key))}</span>'
            f'{pred_chip}'
            f'<div style="height:4px;background:var(--surface-3);border-radius:2px;overflow:hidden;">'
            f'<div style="height:100%;width:{mag_pct}%;background:{bar_color};border-radius:2px;"></div>'
            f'</div>'
            f'<span class="font-mono" style="font-size:11px;font-variant-numeric:tabular-nums;'
            f'text-align:right;color:var(--text-secondary);">{escape(value_label)}</span>'
            f'</div>'
        )

    return (
        '<div style="padding:8px 0 4px;">'
        '<div style="font-size:10px;font-weight:500;letter-spacing:0.08em;'
        'text-transform:uppercase;color:var(--text-tertiary);margin-bottom:8px;">'
        'Per-detector breakdown</div>'
        f'{rows}'
        '</div>'
    )


def render_actions_disclaimer() -> str:
    """One-liner under the action buttons reminding ops of the no-auto invariant.

    Color bumped from --text-quaternary (~2.6:1 contrast) to --text-tertiary
    (~4.4:1, meets WCAG AA for normal text). This line is a safety statement
    — it needs to stay readable, not just decorative.
    """
    return (
        '<p class="font-mono" style="font-size:10px;margin-top:12px;line-height:1.5;'
        'font-weight:500;color:var(--text-tertiary);">'
        'No action is auto-executed. Every decision is logged with operator, timestamp, and rationale.'
        '</p>'
    )
