"""Shared reporting infrastructure for evaluatorq submodules.

Section types live in ``evaluatorq.contracts`` (``ReportSection``).
This package provides the renderer-agnostic helpers and dispatch loop;
``redteam.reports`` and ``simulation.reports`` supply per-module section
builders and per-section render functions.
"""

from evaluatorq.common.reports.html_helpers import (
    COLORS,
    STATUS_COLORS,
    charts_available,
    esc,
    html_table,
    load_css,
    render_donut_chart,
    render_horizontal_bar_chart,
    try_render_svg,
)
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
    "COLORS",
    "RendererRegistry",
    "STATUS_COLORS",
    "bar",
    "bold_bar",
    "center_table",
    "charts_available",
    "details_block",
    "esc",
    "format_date",
    "html_table",
    "load_css",
    "md_table",
    "pct",
    "render_donut_chart",
    "render_header_md",
    "render_horizontal_bar_chart",
    "render_html",
    "render_markdown",
    "truncate",
]
