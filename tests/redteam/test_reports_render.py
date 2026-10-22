"""Smoke tests for redteam.reports rendering (RES-846).

The redteam report renderers were refactored to consume shared helpers
from evaluatorq.common.reports. These smoke tests catch regressions
during that refactor and any future common/reports changes that drop a
helper the redteam renderers depend on.
"""

from __future__ import annotations

from datetime import datetime, timezone

import re

import pytest

from evaluatorq.redteam.contracts import JuryHealth, Pipeline, RedTeamReport, ReportSummary
from evaluatorq.redteam.reports import export_html, export_markdown


def _make_empty_report(summary: ReportSummary | None = None) -> RedTeamReport:
    """Smallest valid RedTeamReport — no attacks run yet."""
    return RedTeamReport(
        created_at=datetime(2026, 5, 29, 12, 0, tzinfo=timezone.utc),
        description="smoke",
        pipeline=Pipeline.DYNAMIC,
        framework=None,
        categories_tested=["ASI01"],
        tested_agents=["agent:test"],
        total_results=0,
        results=[],
        summary=summary or ReportSummary(),
    )


def test_redteam_export_markdown_smoke():
    """Smoke: redteam export_markdown produces a usable Markdown document."""
    md = export_markdown(_make_empty_report())
    assert "# Red Team Security Report" in md
    assert "Executive Summary" in md
    # Must not crash on an empty result set.
    assert "**Target:**" in md


def test_redteam_export_html_smoke():
    """Smoke: redteam export_html produces a self-contained HTML document."""
    html = export_html(_make_empty_report())
    assert html.startswith("<!DOCTYPE html>")
    assert "<title>Red Team Security Report</title>" in html
    # Inlined CSS - check a class that should be defined.
    assert "<style>" in html
    # The %% bug regression: rendered CSS must not contain '%%'.
    assert "%%" not in html


def test_redteam_export_markdown_renders_jury_health():
    """md export surfaces jury_health as a Jury Health row (RES-1649)."""
    report = _make_empty_report(
        ReportSummary(jury_health=JuryHealth(samples=5, samples_with_judge_failure=2, ties=1, replacements_used=3))
    )
    md = export_markdown(report)
    assert "Jury Health" in md
    assert "2 judge-fail" in md
    assert "3 replacement judge(s)" in md
    assert "5 multi-judge samples" in md


@pytest.mark.parametrize(('health', 'headline', 'label'), [
    (JuryHealth(samples=4), 'clean', 'Jury Health'),
    (JuryHealth(samples=4, inconclusive=1), '1', 'Jury Health (issue types)'),
    (JuryHealth(samples=4, inconclusive=1, ties=1), '2', 'Jury Health (issue types)'),
])
def test_redteam_export_html_renders_jury_health(health, headline, label):
    report = _make_empty_report(ReportSummary(jury_health=health))
    html = export_html(report)
    card = re.search(r'<div class="kpi-card"><div class="kpi-value">([^<]*)</div>'
                     r'<div class="kpi-label">(Jury Health(?: \(issue types\))?)</div>'
                     r'<div class="kpi-subtitle">([^<]*)</div>', html)
    assert card is not None
    assert card.group(1) == headline
    assert card.group(2) == label
    assert '4 multi-judge samples' in card.group(3)



def test_jury_health_loaded_reasons_are_scrubbed_before_deduplication_and_display():
    key = 'sk-' + 'SyntheticCredential1234' * 2
    reason = 'Provider rejected credential: ' + key + ' after request ' + 'x' * 100
    other = reason.replace(key, 'sk-' + 'DifferentCredential5678' * 2)
    health = JuryHealth.model_validate({'samples': 2, 'samples_with_judge_failure': 2,
                                      'judge_failure_reasons': [' ', reason, other]})
    expected = reason.replace(key, '<API_KEY>')
    assert health.judge_failure_reasons == [expected]
    assert JuryHealth.model_validate_json(health.model_dump_json()) == health
    assert key not in health.model_dump_json()
    assert expected[:79] + '…' in health.issue_summary()
    report = _make_empty_report(ReportSummary(jury_health=health))
    for rendered in (export_html(report), export_markdown(report)):
        assert key not in rendered
        assert 'SyntheticCredential' not in rendered
    assert '&lt;API_KEY&gt;' in export_html(report)


@pytest.mark.parametrize('field', ['samples_with_judge_failure', 'ties', 'inconclusive',
                                   'samples_with_repetition_failure', 'samples_with_abstention'])
def test_jury_health_rejects_impossible_sample_counts(field):
    with pytest.raises(ValueError, match='cannot exceed samples'):
        JuryHealth.model_validate({'samples': 1, field: 2})


@pytest.mark.parametrize('extra', [{'replacements_used': 1}, {'judge_failure_reasons': ['timeout']}])
def test_jury_health_requires_samples_for_judges_and_reasons(extra):
    with pytest.raises(ValueError, match='require multi-judge samples'):
        JuryHealth.model_validate({'samples': 0, **extra})


def test_jury_health_failure_reasons_never_render_as_clean():
    health = JuryHealth(samples=1, judge_failure_reasons=['timeout'])
    assert health.issue_count() == 1
    assert health.issue_summary() == 'judge failures: timeout'


def test_write_report_helpers_exit_on_oserror(tmp_path, monkeypatch):
    """The CLI write_*_report helpers must convert OSError into a friendly
    typer.Exit instead of leaking a raw traceback.
    """
    import typer

    from evaluatorq.redteam.cli import write_html_report, write_markdown_report

    report = _make_empty_report()
    out_dir = tmp_path / "reports"

    def fail_write(self, *args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("pathlib.Path.write_text", fail_write)

    import pytest

    with pytest.raises(typer.Exit) as exc:
        write_markdown_report(report, out_dir, target="t")
    assert exc.value.exit_code == 1

    with pytest.raises(typer.Exit) as exc:
        write_html_report(report, out_dir, target="t")
    assert exc.value.exit_code == 1
