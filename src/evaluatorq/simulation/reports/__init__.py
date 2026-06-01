"""Report generation for agent simulation results.

Mirrors ``redteam.reports``: ``sections.py`` builds renderer-agnostic
``ReportSection`` objects; ``export_md`` / ``export_html`` render them via
the shared dispatch in ``evaluatorq.common.reports``.

These modules import only first-party code at module load time. Plotly /
kaleido are imported lazily inside the chart helpers in
``evaluatorq.common.reports.html_helpers`` and degrade to an empty
string when absent, so no ``try/except ImportError`` is needed here —
adding one would only swallow genuine packaging errors (typos, circular
imports) and turn them into ``AttributeError`` at the call site with no
traceback.
"""

from evaluatorq.simulation.reports.export_html import export_html
from evaluatorq.simulation.reports.export_md import export_markdown
from evaluatorq.simulation.reports.sections import build_report_sections

__all__ = [
    "build_report_sections",
    "export_html",
    "export_markdown",
]
