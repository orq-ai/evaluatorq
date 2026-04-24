"""Reporting and result conversion utilities for red teaming."""

from loguru import logger

try:  # noqa: RUF067
    from evaluatorq.redteam.reports.export_html import export_html
except ImportError:
    logger.debug("HTML export unavailable: missing optional dependency (e.g. jinja2)")
    export_html = None  # type: ignore[assignment]

try:  # noqa: RUF067
    from evaluatorq.redteam.reports.export_md import export_markdown
except ImportError:
    logger.debug("Markdown export unavailable: missing optional dependency")
    export_markdown = None  # type: ignore[assignment]

try:  # noqa: RUF067
    from evaluatorq.redteam.reports.sections import build_report_sections
except ImportError:
    logger.debug("Report sections unavailable: missing optional dependency")
    build_report_sections = None  # type: ignore[assignment]

__all__ = [
    "build_report_sections",
    "export_html",
    "export_markdown",
]
