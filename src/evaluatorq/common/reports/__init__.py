"""Shared reporting infrastructure for evaluatorq submodules.

Section types live in ``evaluatorq.contracts`` (``ReportSection``).
This package provides the renderer-agnostic helpers and dispatch loop;
``redteam.reports`` and ``simulation.reports`` supply per-module section
builders and per-section render functions.
"""

from evaluatorq.common.reports import palette
from evaluatorq.common.reports.html_helpers import (
    COLORS,
    STATUS_COLORS,
    charts_available,
    esc,
    html_table,
    kpi_cards,
    load_css,
    render_donut_chart,
    render_heatmap,
    render_histogram,
    render_horizontal_bar_chart,
    render_line_chart,
    render_sparkline,
    scale_color,
    status_badge,
    svg_bar,
    svg_donut,
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
from evaluatorq.common.reports.palette import (
    QUALITATIVE,
    SEVERITY_COLORS,
    SEVERITY_ORDER,
    ORQ_SCALE_AGENT,
    ORQ_SCALE_GOOD_BAD,
    ORQ_SCALE_HEAT,
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
    "QUALITATIVE",
    "RendererRegistry",
    "SEVERITY_COLORS",
    "SEVERITY_ORDER",
    "STATUS_COLORS",
    "ORQ_SCALE_AGENT",
    "ORQ_SCALE_GOOD_BAD",
    "ORQ_SCALE_HEAT",
    "bar",
    "bold_bar",
    "center_table",
    "charts_available",
    "details_block",
    "esc",
    "format_date",
    "html_table",
    "kpi_cards",
    "load_css",
    "md_table",
    "palette",
    "pct",
    "render_donut_chart",
    "render_heatmap",
    "render_header_md",
    "render_histogram",
    "render_horizontal_bar_chart",
    "render_html",
    "render_line_chart",
    "render_markdown",
    "render_sparkline",
    "scale_color",
    "status_badge",
    "svg_bar",
    "svg_donut",
    "truncate",
]
