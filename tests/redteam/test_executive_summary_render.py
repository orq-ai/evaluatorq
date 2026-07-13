from __future__ import annotations

from evaluatorq.redteam.reports.export_html import export_html
from evaluatorq.redteam.reports.export_md import export_markdown
from tests.redteam.test_executive_summary_facts import _report

_NARRATIVE = 'Across 3 attacks the agent resisted 67% but exposed 1 critical finding.'


def test_html_includes_narrative_and_pill():
    report = _report()
    report.executive_summary = _NARRATIVE
    html = export_html(report)
    assert _NARRATIVE in html
    # Confidence pill rendered via shared status_badge component.
    assert 'status-badge' in html
    assert 'CONFIDENCE' in html


def test_markdown_includes_narrative():
    report = _report()
    report.executive_summary = _NARRATIVE
    md = export_markdown(report)
    assert _NARRATIVE in md


def test_html_omits_narrative_when_absent():
    report = _report()  # executive_summary is None
    html = export_html(report)
    assert 'exec-summary-narrative' not in html


def test_terminal_summary_includes_narrative():
    import io

    from rich.console import Console

    from evaluatorq.redteam.reports.display import print_report_summary

    report = _report()
    report.executive_summary = _NARRATIVE
    buffer = io.StringIO()
    console = Console(file=buffer, width=120, highlight=False, markup=False, no_color=True)

    print_report_summary(report, console=console)

    assert _NARRATIVE in buffer.getvalue()
