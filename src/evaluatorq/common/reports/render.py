"""Shared rendering dispatch for Markdown and HTML reports.

Section builders in ``redteam.reports`` / ``simulation.reports`` produce a
``list[ReportSection]``. Each module registers its own ``kind -> render_fn``
mapping (a ``RendererRegistry``) and calls ``render_markdown`` /
``render_html`` here. Common owns the rendering loop, collapsible-block
wrapping, and document header — modules own the per-section logic.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from datetime import datetime

from loguru import logger

from evaluatorq.common.reports.md_helpers import details_block
from evaluatorq.contracts import ReportSection

RendererRegistry = dict[str, Callable[[ReportSection], str]]
"""Map a ``ReportSection.kind`` to a function rendering it as a string."""


def render_header_md(
    *,
    title: str,
    rows: Iterable[tuple[str, str]],
) -> str:
    """Render a Markdown document header.

    Args:
        title: Top-level ``# Title`` heading.
        rows: Ordered ``(label, value)`` pairs rendered as ``**Label:** value``
            lines (one per line, with a trailing two-space hard break).
    """
    lines = [f"# {title}", ""]
    lines.extend(f"**{label}:** {value}  " for label, value in rows)
    return "\n".join(lines)


def format_date(dt: datetime | None) -> str:
    """Format a datetime as ``'YYYY-MM-DD HH:MM UTC'`` or ``'unknown'``."""
    return dt.strftime("%Y-%m-%d %H:%M UTC") if dt else "unknown"


def render_markdown(
    sections: list[ReportSection],
    *,
    renderers: RendererRegistry,
    collapsed_kinds: set[str] | None = None,
    header: str = "",
    footer: str = "",
) -> str:
    """Render a list of sections as a Markdown document.

    Args:
        sections: Sections to render, in order.
        renderers: ``kind -> render_fn`` lookup. Sections whose kind is not in
            the registry are skipped silently (so callers can mix in unsupported
            sections without breaking the run).
        collapsed_kinds: Kinds wrapped in a ``<details>`` block. The renderer
            strips a leading ``## {section.title}`` heading from the rendered
            output before wrapping, because ``details_block`` provides its own
            ``<summary>`` title.
        header: Optional document header rendered before the sections.
        footer: Optional document footer rendered after the sections.

    Returns:
        The full Markdown document as a single string.
    """
    collapsed_kinds = collapsed_kinds or set()
    parts: list[str] = []
    if header:
        parts.extend((header, ""))

    for section in sections:
        renderer = renderers.get(section.kind)
        if renderer is None:
            logger.warning(
                "No renderer registered for section kind {!r}; the section "
                "will be missing from the markdown report.",
                section.kind,
            )
            continue
        rendered = renderer(section)
        if not rendered:
            continue
        if section.kind in collapsed_kinds:
            heading_prefix = f"## {section.title}\n"
            if rendered.startswith(heading_prefix):
                rendered = rendered[len(heading_prefix):].lstrip("\n")
            rendered = details_block(section.title, rendered)
        parts.extend((rendered, ""))

    if footer:
        parts.append(footer)

    return "\n".join(parts)


def render_html(
    sections: list[ReportSection],
    *,
    renderers: RendererRegistry,
    head: str = "",
    body_header: str = "",
    body_footer: str = "",
) -> str:
    """Render a list of sections as an HTML document.

    Each module supplies the head (CSS link, title) and the per-section
    renderers. Sections whose kind is not registered are skipped silently.

    Args:
        sections: Sections to render, in order.
        renderers: ``kind -> render_fn`` lookup producing HTML fragments.
        head: HTML content for the ``<head>`` (typically a ``<style>`` block
            and ``<title>``).
        body_header: HTML rendered at the top of the ``<body>``.
        body_footer: HTML rendered at the bottom of the ``<body>``.

    Returns:
        A self-contained HTML5 document.
    """
    body_parts: list[str] = []
    if body_header:
        body_parts.append(body_header)
    for section in sections:
        renderer = renderers.get(section.kind)
        if renderer is None:
            logger.warning(
                "No renderer registered for section kind {!r}; the section "
                "will be missing from the HTML report.",
                section.kind,
            )
            continue
        rendered = renderer(section)
        if rendered:
            body_parts.append(rendered)
    if body_footer:
        body_parts.append(body_footer)

    return (
        "<!DOCTYPE html>\n"
        "<html lang=\"en\">\n"
        f"<head>\n{head}\n</head>\n"
        f"<body>\n{chr(10).join(body_parts)}\n</body>\n"
        "</html>\n"
    )
