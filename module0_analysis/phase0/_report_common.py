"""Shared rendering helpers for Phase 0 Markdown report modules.

Single Responsibility
---------------------
Hosts presentation primitives reused by every ``*_report.py`` module so each
report file owns only its own section content, not the boilerplate.
"""

from __future__ import annotations

from typing import Callable

# A "writer": ``lines.append`` from inside each report renderer.
Writer = Callable[[str], None]


def render_section_header(w: Writer, title: str, body: str) -> None:
    """Render a top-level section header followed by an introductory paragraph.

    Output shape::

        {title}
        <blank>
        {body}
        <blank>

    Args:
        w: Line writer (typically ``lines.append`` of a list-of-strings).
        title: Markdown heading line, including the leading ``##``.
        body: Single introductory paragraph for the section.
    """
    w(title)
    w("")
    w(body)
    w("")
