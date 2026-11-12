"""Run-level warnings (e.g. failed strategy generation) reach every report renderer."""

from __future__ import annotations

import pytest

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
    # The standalone HTML export renders the same warning section.
    assert 'strategy generation failed (HTTP 429)' in render_report_body(report)


def test_warning_lines_cannot_escape_the_markdown_callout() -> None:
    warning = 'HTTP 429\n# Forged heading\n<script>alert(1)</script>\n> [!NOTE]'
    markdown = export_markdown(_report([warning]))
    assert '> - HTTP 429' in markdown
    assert '>   \\# Forged heading' in markdown
    assert '>   &lt;script&gt;alert(1)&lt;/script&gt;' in markdown
    assert '>   &gt; \\[!NOTE\\]' in markdown
    assert '<script>' not in markdown


def test_dashboard_shows_run_warnings() -> None:
    from evaluatorq.dashboard.report_tabs import redteam_report_tabs

    html = redteam_report_tabs('rid', _report([_WARNING]))
    assert 'id="section-pipeline_warnings"' in html
    assert 'strategy generation failed (HTTP 429)' in html


_LEAKY = 'Failed to generate executive summary (Error code: 401 - Incorrect API key provided: sk-proj-AbCdEf1234567890AbCdEf1234567890).'


def test_a_credential_in_a_warning_is_scrubbed_on_validation_and_on_save() -> None:
    import json

    from evaluatorq.redteam.contracts import RedTeamReport

    built = RedTeamReport.model_validate({**_report([]).model_dump(mode='json'), 'pipeline_warnings': [_LEAKY]})
    assert built.pipeline_warnings == [_LEAKY.replace('sk-proj-AbCdEf1234567890AbCdEf1234567890', '<API_KEY>')]

    # The run appends warnings after the report is built, which skips the validator; the saved JSON is still clean.
    appended = _report([])
    appended.pipeline_warnings.append(_LEAKY)
    saved = json.loads(appended.model_dump_json())
    assert 'sk-proj-' not in json.dumps(saved['pipeline_warnings'])
    assert '<API_KEY>' in saved['pipeline_warnings'][0]


@pytest.mark.asyncio
async def test_a_failed_executive_summary_records_a_scrubbed_warning(monkeypatch) -> None:
    from openai import AsyncOpenAI

    from evaluatorq.redteam import runner
    from evaluatorq.redteam.contracts import LLMConfig

    async def leaky(*_args, **_kwargs):
        raise RuntimeError('Incorrect API key provided: sk-proj-AbCdEf1234567890AbCdEf1234567890')

    monkeypatch.setattr('evaluatorq.common.reports.executive_summary.generate_executive_summary', leaky)
    report = _report([])
    await runner._generate_executive_summary_section(
        report=report,
        llm_client=AsyncOpenAI(api_key='sk-test', max_retries=0),
        config=LLMConfig(),
        evaluator_model='openai/gpt-6-sol',
    )
    assert len(report.pipeline_warnings) == 1
    assert 'sk-proj-' not in report.pipeline_warnings[0]
    assert '<API_KEY>' in report.pipeline_warnings[0]
