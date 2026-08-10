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


def test_unevaluated_vulnerability_does_not_render_as_passed():
    """A vulnerability with no verdicts must not print "3 tested, 3 passed" in green.

    ``passed`` was derived as ``total_attacks - vulnerabilities_found``, so a slice
    where every judge call failed reported the target as having resisted every
    attack — next to an honest ``ASR n/a``, and styled green because the count
    equalled the total. This is the report-level face of the bug the run-level
    coverage gate catches.
    """
    import io

    from rich.console import Console

    from evaluatorq.redteam.contracts import ReportSummary, VulnerabilitySummary
    from evaluatorq.redteam.reports.display import print_report_summary

    report = _report()
    report.summary = ReportSummary(
        total_attacks=3,
        evaluated_attacks=0,
        unevaluated_attacks=3,
        by_vulnerability={
            'agent_goal_hijacking': VulnerabilitySummary(
                vulnerability='agent_goal_hijacking',
                vulnerability_name='Agent Goal Hijacking',
                domain='agent',
                total_attacks=3,
                evaluated_attacks=0,
                vulnerabilities_found=0,
                resistance_rate=None,
            )
        },
    )
    buffer = io.StringIO()
    print_report_summary(report, console=Console(file=buffer, width=120, highlight=False, markup=False, no_color=True))
    out = buffer.getvalue()

    row = next(line for line in out.splitlines() if 'agent_goal_hijacking' in line)
    cells = [c.strip() for c in row.strip('│ ').split('│')]
    assert cells[2] == '0 of 3', row  # Tested names the denominator that was actually scored
    assert cells[3] == '0', row  # Passed, not 3
    assert cells[4] == 'n/a', row
