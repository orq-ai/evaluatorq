"""A coerced verdict (the judge abstained and also sent a verdict) shows in the report and the dashboard, for a panel and for one judge."""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

import pytest
from rich.console import Console

from evaluatorq.dashboard.redteam_transcripts import render_attack_fragment
from evaluatorq.redteam.contracts import JuryResult, JuryVote, Pipeline, RedTeamReport, RedTeamResult, ReportSummary
from evaluatorq.redteam.reports import export_html, export_markdown
from evaluatorq.redteam.reports.converters import (
    coerced_passes,
    compute_report_summary,
    samples_with_coerced_verdict,
)
from evaluatorq.redteam.reports.display import print_report_summary


def _panel(*coerced: int) -> JuryResult:
    return JuryResult(
        judges_configured=len(coerced),
        judges_succeeded=len(coerced),
        judges_failed=0,
        votes=[
            JuryVote(model=f'judge-{index}', success=True, value=True, repetitions_coerced=count)
            for index, count in enumerate(coerced)
        ],
    )


def _with(result: RedTeamResult, **evaluation: Any) -> RedTeamResult:
    assert result.evaluation is not None
    return result.model_copy(update={'evaluation': result.evaluation.model_copy(update=evaluation)})


def _report(summary: ReportSummary) -> RedTeamReport:
    return RedTeamReport(
        created_at=datetime(2026, 5, 29, 12, 0, tzinfo=timezone.utc),
        description='smoke',
        pipeline=Pipeline.DYNAMIC,
        framework=None,
        categories_tested=['ASI01'],
        tested_agents=['agent:test'],
        total_results=0,
        results=[],
        summary=summary,
    )


@pytest.mark.parametrize(
    ('evaluation', 'passes'),
    [
        (None, 0),
        (SimpleNamespace(jury=None, raw_output=None), 0),
        (SimpleNamespace(jury=None, raw_output={'raw_content': '{}'}), 0),
        (SimpleNamespace(jury=None, raw_output={'verdict_coerced': False}), 0),
        (SimpleNamespace(jury=None, raw_output={'verdict_coerced': 'true'}), 0),
        (SimpleNamespace(jury=None, raw_output={'verdict_coerced': True}), 1),
        (SimpleNamespace(jury=_panel(0, 0), raw_output=None), 0),
        (SimpleNamespace(jury=_panel(2, 0, 1), raw_output=None), 3),
        # A panel is read from its votes: a stale flag beside it is not added on top.
        (SimpleNamespace(jury=_panel(1), raw_output={'verdict_coerced': True}), 1),
    ],
)
def test_coerced_passes_reads_the_panel_and_the_single_judge_path(evaluation: Any, passes: int) -> None:
    assert coerced_passes(evaluation) == passes


def test_the_summary_counts_each_sample_once_on_both_paths(
    rt_result_vuln: RedTeamResult, rt_result_safe: RedTeamResult
) -> None:
    results = [
        _with(rt_result_vuln, jury=_panel(2, 1)),  # panel, three coerced passes: one sample
        _with(rt_result_safe, raw_output={'verdict_coerced': True}),  # single judge: one sample
        _with(rt_result_safe, raw_output={'verdict_coerced': False}),
        _with(rt_result_vuln, jury=_panel(0, 0)),
        rt_result_safe,
    ]

    assert compute_report_summary(results).samples_with_coerced_verdict == 2
    assert compute_report_summary([rt_result_safe]).samples_with_coerced_verdict == 0


def test_the_report_exports_show_the_count_and_stay_silent_at_zero() -> None:
    shown = _report(ReportSummary(samples_with_coerced_verdict=3))
    markdown, html = export_markdown(shown), export_html(shown)

    assert '| Coerced Verdicts | 3 samples (judge abstained and also sent a verdict) |' in markdown
    # The HTML export is the dashboard's tabs, so the count is on its Config tab.
    config = html.split('id="rttab-config"', 1)[1]
    assert 'JUDGE VERDICTS' in config
    assert 'Coerced verdicts' in config
    assert '3 samples (judge abstained and also sent a verdict; not counted under Abstentions)' in config

    quiet = _report(ReportSummary())
    for rendered in (export_markdown(quiet), export_html(quiet)):
        assert 'coerced verdicts' not in rendered.lower()
        assert 'JUDGE VERDICTS' not in rendered


def test_the_console_summary_shows_the_count_and_stays_silent_at_zero() -> None:
    console = Console(record=True, width=200)
    print_report_summary(_report(ReportSummary(samples_with_coerced_verdict=3)), console=console)
    shown = console.export_text()
    assert 'Coerced Verdicts' in shown
    assert '3 samples' in shown

    console = Console(record=True, width=200)
    print_report_summary(_report(ReportSummary()), console=console)
    assert 'Coerced Verdicts' not in console.export_text()


def test_the_attack_panel_tags_each_panel_judge_with_its_coerced_repetitions(rt_result_vuln: RedTeamResult) -> None:
    html = render_attack_fragment(_with(rt_result_vuln, jury=_panel(2, 1, 0)))

    assert 'Jury panel health' in html
    assert '>2 coerced reps<' in html
    assert '>1 coerced rep<' in html
    assert html.count('coerced rep') == 2, 'the judge with none carries no tag'


def test_the_attack_panel_shows_a_single_judge_coerced_verdict(rt_result_vuln: RedTeamResult) -> None:
    html = render_attack_fragment(_with(rt_result_vuln, raw_output={'verdict_coerced': True}))

    assert 'Judge health' in html
    assert 'verdict coerced to abstention' in html
    assert 'Jury panel health' not in html


@pytest.mark.parametrize('raw_output', [None, {'raw_content': '{}'}, {'verdict_coerced': False}])
def test_the_attack_panel_adds_nothing_for_a_single_judge_that_was_not_coerced(
    rt_result_vuln: RedTeamResult, raw_output: dict[str, Any] | None
) -> None:
    html = render_attack_fragment(_with(rt_result_vuln, raw_output=raw_output))

    assert 'Judge health' not in html
    assert 'coerced' not in html


def test_one_coerced_sample_reads_as_one_sample_everywhere() -> None:
    single = _report(ReportSummary(samples_with_coerced_verdict=1))

    assert '| Coerced Verdicts | 1 sample (judge abstained and also sent a verdict) |' in export_markdown(single)
    config = export_html(single).split('id="rttab-config"', 1)[1]
    assert '>1 sample (judge abstained' in config
    console = Console(record=True, width=200)
    print_report_summary(single, console=console)
    shown = console.export_text()
    assert '1 sample' in shown
    assert '1 samples' not in shown


def test_a_report_saved_before_the_count_existed_loads_with_zero() -> None:
    saved = ReportSummary().model_dump(mode='json')
    del saved['samples_with_coerced_verdict']

    loaded = ReportSummary.model_validate(saved)

    assert loaded.samples_with_coerced_verdict == 0
    for rendered in (export_markdown(_report(loaded)), export_html(_report(loaded))):
        assert 'coerced verdicts' not in rendered.lower()


def test_an_older_saved_report_counts_its_coerced_samples_from_the_results(rt_result_vuln: RedTeamResult) -> None:
    """Saved before the summary carried the count: the summary says 0 and the results still record the passes."""
    results = [
        _with(rt_result_vuln, jury=_panel(2, 0)),
        _with(rt_result_vuln, raw_output={'verdict_coerced': True}),
        rt_result_vuln,
    ]
    older = _report(ReportSummary()).model_copy(update={'results': results, 'total_results': len(results)})
    assert older.summary.samples_with_coerced_verdict == 0

    assert samples_with_coerced_verdict(older) == 2
    assert '| Coerced Verdicts | 2 samples (judge abstained and also sent a verdict) |' in export_markdown(older)
    assert '>2 samples (judge abstained' in export_html(older).split('id="rttab-config"', 1)[1]
    console = Console(record=True, width=200)
    print_report_summary(older, console=console)
    assert '2 samples' in console.export_text()


def test_a_report_without_results_keeps_the_count_its_summary_carries() -> None:
    assert samples_with_coerced_verdict(_report(ReportSummary(samples_with_coerced_verdict=4))) == 4
