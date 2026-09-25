"""Run-level warnings (e.g. failed strategy generation) reach every report renderer."""

from __future__ import annotations

from tests.redteam.reports.test_rebuild_filtered import _make_report, _make_result

from evaluatorq.redteam.reports.export_html import render_report_body
from evaluatorq.redteam.reports.export_md import export_markdown
from evaluatorq.redteam.reports.sections import build_report_sections

_WARNING = "Category 'ASI01': strategy generation failed (HTTP 429); ran 3 hardcoded strategies only"


def _report(warnings: list[str]):
    report = _make_report([_make_result()], ['agent-a'])
    return report.model_copy(update={'pipeline_warnings': warnings})


def test_section_follows_the_summary_when_warnings_exist() -> None:
    kinds = [s.kind for s in build_report_sections(_report([_WARNING]))]
    assert kinds[:2] == ['summary', 'pipeline_warnings']


def test_no_section_without_warnings() -> None:
    assert 'pipeline_warnings' not in [s.kind for s in build_report_sections(_report([]))]


def test_markdown_and_html_show_the_warning() -> None:
    report = _report([_WARNING])
    assert _WARNING in export_markdown(report)
    # HTML (and the dashboard, which embeds this body) escapes the quotes.
    assert 'strategy generation failed (HTTP 429)' in render_report_body(report)
