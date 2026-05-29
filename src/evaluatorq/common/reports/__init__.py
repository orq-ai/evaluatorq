"""Shared reporting infrastructure for evaluatorq submodules.

Section types live in ``evaluatorq.contracts`` (``ReportSection``).
This package provides the renderer-agnostic helpers and dispatch loop;
``redteam.reports`` and ``simulation.reports`` supply per-module section
builders and per-section render functions.
"""

from evaluatorq.common.reports.md_helpers import (
    bar,
    bold_bar,
    center_table,
    details_block,
    md_table,
    pct,
    truncate,
)
from evaluatorq.common.reports.render import (
    RendererRegistry,
    format_date,
    render_header_md,
    render_html,
    render_markdown,
)


__all__ = [
    "RendererRegistry",
    "bar",
    "bold_bar",
    "center_table",
    "details_block",
    "format_date",
    "md_table",
    "pct",
    "render_header_md",
    "render_html",
    "render_markdown",
    "truncate",
]
