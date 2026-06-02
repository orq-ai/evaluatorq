"""Reporting and result conversion utilities for red teaming.

These modules import only first-party code at module load time. Plotly /
kaleido (used by the HTML chart helpers) are imported lazily inside
``evaluatorq.common.reports.html_helpers`` and degrade to an empty
string when absent, so no ``try/except ImportError`` is needed here —
adding one would only swallow genuine packaging errors (typos, circular
imports) and turn them into ``AttributeError`` at the call site with no
traceback.
"""

from evaluatorq.redteam.reports.export_html import export_html
from evaluatorq.redteam.reports.export_md import export_markdown
from evaluatorq.redteam.reports.sections import build_report_sections


__all__ = [
    "build_report_sections",
    "export_html",
    "export_markdown",
]
