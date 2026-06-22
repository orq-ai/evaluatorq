"""Reporting and result conversion utilities for red teaming.

These modules import only first-party code at module load time. Charts
are rendered as static SVGs via Vega-Lite / vl-convert; when
``vl-convert-python`` is absent charts are omitted and the report
degrades gracefully to a tables-only layout.
"""

from evaluatorq.redteam.reports.export_html import export_html
from evaluatorq.redteam.reports.export_md import export_markdown
from evaluatorq.redteam.reports.sections import build_report_sections


__all__ = [
    "build_report_sections",
    "export_html",
    "export_markdown",
]
