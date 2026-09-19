"""Characterization tests for the Rich red-team report summary display."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from rich.console import Console

from evaluatorq.redteam.contracts import (
    AttackInfo,
    AttackTechnique,
    AgentInfo,
    CategorySummary,
    DeliveryMethod,
    Framework,
    Pipeline,
    RedTeamReport,
    RedTeamResult,
    ReportSummary,
    Severity,
    TechniqueSummary,
    TurnType,
    UnifiedEvaluationResult,
    VulnerabilitySummary,
)
from evaluatorq.redteam.reports import display
from evaluatorq.redteam.reports.display import print_report_summary


def _make_report(
    summary: ReportSummary,
    results: list[RedTeamResult] | None = None,
    *,
    executive_summary: str | None = None,
) -> RedTeamReport:
    return RedTeamReport(
        created_at=datetime(2026, 5, 29, 12, 0, tzinfo=timezone.utc),
        description='display test',
        pipeline=Pipeline.DYNAMIC,
        framework=Framework.OWASP_ASI,
        categories_tested=['ASI01'],
        tested_agents=['agent:test'],
        total_results=len(results or []),
        results=results or [],
        summary=summary,
        executive_summary=executive_summary,
    )


def _make_error_result() -> RedTeamResult:
    return RedTeamResult(
        attack=AttackInfo(
            id='ASI01-display-001',
            category='ASI01',
            framework=Framework.OWASP_ASI,
            attack_technique=AttackTechnique.DIRECT_INJECTION,
            delivery_methods=[DeliveryMethod.DIRECT_REQUEST],
            turn_type=TurnType.SINGLE,
            severity=Severity.MEDIUM,
            source='test',
        ),
        agent=AgentInfo(key='agent:test'),
        messages=[],
        evaluation=UnifiedEvaluationResult(passed=None, explanation='failed'),
        error='request timed out',
        error_type='timeout',
        error_code='timeout',
        error_stage='target',
    )


def test_print_report_summary_renders_zero_results_state() -> None:
    console = Console(record=True, width=120)

    print_report_summary(_make_report(ReportSummary()), console=console)

    output = console.export_text()
    assert 'RED TEAM REPORT SUMMARY' in output
    assert 'Total Attacks' in output
    assert '│ Total Attacks          │ 0               │' in output
    assert '│ Evaluated              │ 0               │' in output
    assert '│ Vulnerabilities        │ 0               │' in output
    assert '│ ASR                    │ n/a             │' in output
    assert '│ Eval Coverage          │ 0%              │' in output
    assert 'Per-Vulnerability Breakdown:' not in output
    assert 'Per-Category Breakdown:' not in output
    assert 'Top Vulnerable Techniques:' not in output
    assert 'Top Error Causes:' not in output


def test_print_report_summary_renders_each_populated_section() -> None:
    result = _make_error_result()
    summary = ReportSummary(
        total_attacks=1,
        evaluated_attacks=1,
        evaluation_coverage=1.0,
        vulnerabilities_found=1,
        vulnerability_rate=1.0,
        resistance_rate=0.0,
        total_errors=1,
        errors_by_type={'timeout': 1},
        by_vulnerability={
            'goal_hijacking': VulnerabilitySummary(
                vulnerability='goal_hijacking',
                vulnerability_name='Goal Hijacking',
                domain='agentic_security',
                total_attacks=1,
                evaluated_attacks=1,
                vulnerabilities_found=1,
                resistance_rate=0.0,
            )
        },
        by_category={
            'ASI01': CategorySummary(
                category='ASI01',
                category_name='Goal Hijacking',
                total_attacks=1,
                evaluated_attacks=1,
                vulnerabilities_found=1,
                vulnerability_rate=1.0,
                resistance_rate=0.0,
            )
        },
        by_technique={'direct_injection': TechniqueSummary(total_attacks=1, vulnerabilities_found=1)},
    )
    console = Console(record=True, width=120)

    print_report_summary(
        _make_report(summary, [result], executive_summary='Goal hijacking is the highest observed risk.'),
        console=console,
    )

    output = console.export_text()
    assert 'Executive Summary' in output
    assert 'Goal hijacking is the highest observed risk.' in output
    assert 'Per-Vulnerability Breakdown:' in output
    assert 'Per-Category Breakdown:' in output
    assert 'Top Vulnerable Techniques:' in output
    assert 'Top Error Causes:' in output


def test_print_report_summary_prints_earlier_section_before_later_builder_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summary = ReportSummary(
        by_vulnerability={
            'goal_hijacking': VulnerabilitySummary(
                vulnerability='goal_hijacking',
                vulnerability_name='Goal Hijacking',
                domain='agentic_security',
                total_attacks=1,
                evaluated_attacks=1,
                vulnerabilities_found=1,
                resistance_rate=0.0,
            )
        }
    )
    console = Console(record=True, width=120)

    def fail_to_build_category_table(summary: ReportSummary) -> None:
        raise RuntimeError('category table failed')

    monkeypatch.setattr(display, '_build_category_table', fail_to_build_category_table)

    with pytest.raises(RuntimeError, match='category table failed'):
        print_report_summary(_make_report(summary), console=console)

    output = console.export_text()
    assert 'Per-Vulnerability Breakdown:' in output
    assert 'Goal Hijacking' in output
