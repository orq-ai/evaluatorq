"""Report generation for agent simulation results.

Mirrors ``redteam.reports``: ``sections.py`` builds renderer-agnostic
``ReportSection`` objects; ``export_md`` / ``export_html`` render them via
the shared dispatch in ``evaluatorq.common.reports``.
"""

from loguru import logger

try:  # noqa: RUF067
    from evaluatorq.simulation.reports.export_html import export_html
except ImportError:
    logger.debug("HTML export unavailable: missing optional dependency (e.g. plotly)")
    export_html = None  # type: ignore[assignment]

try:  # noqa: RUF067
    from evaluatorq.simulation.reports.export_md import export_markdown
except ImportError:
    logger.debug("Markdown export unavailable: missing optional dependency")
    export_markdown = None  # type: ignore[assignment]

try:  # noqa: RUF067
    from evaluatorq.simulation.reports.sections import build_report_sections
except ImportError:
    logger.debug("Report sections unavailable: missing optional dependency")
    build_report_sections = None  # type: ignore[assignment]


__all__ = [
    "build_report_sections",
    "export_html",
    "export_markdown",
]
