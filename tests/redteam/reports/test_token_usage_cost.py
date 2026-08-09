"""Cost-breakdown coverage for redteam token-usage rendering (SDD task 4).

Verifies both the `None`-vs-`0.0` cost distinction and the plumbing that
carries cost from `RedTeamResult.execution.token_usage` through
`_build_token_usage_section` into the HTML and Markdown renderers.

- Cost known -> cost rows/cards appear with correct values.
- Cost unknown (`None`, matching old saved reports) -> cost rows/cards are
  absent entirely; never rendered as `$0.00`.
"""

from __future__ import annotations

from datetime import datetime, timezone

from evaluatorq.redteam.contracts import (
    AgentInfo,
    AttackInfo,
    AttackTechnique,
    DeliveryMethod,
    ExecutionDetails,
    Framework,
    Pipeline,
    RedTeamReport,
    RedTeamResult,
    Severity,
    TokenUsage,
    TurnType,
    UnifiedEvaluationResult,
)
from evaluatorq.redteam.reports.converters import compute_report_summary
from evaluatorq.redteam.reports.export_html import export_html
from evaluatorq.redteam.reports.export_md import export_markdown
from evaluatorq.redteam.reports.sections import _build_token_usage_section


def _make_result(*, agent_key: str, token_usage: TokenUsage | None) -> RedTeamResult:
    return RedTeamResult(
        attack=AttackInfo(
            id='ASI01-test-001',
            category='ASI01',
            framework=Framework.OWASP_ASI,
            attack_technique=AttackTechnique.INDIRECT_INJECTION,
            delivery_methods=[DeliveryMethod.DIRECT_REQUEST],
            turn_type=TurnType.SINGLE,
            severity=Severity.MEDIUM,
            source='test',
        ),
        agent=AgentInfo(key=agent_key),
        messages=[],
        vulnerable=False,
        evaluation=UnifiedEvaluationResult(passed=True, explanation='test'),
        execution=ExecutionDetails(token_usage=token_usage),
    )


def _make_report(results: list[RedTeamResult]) -> RedTeamReport:
    return RedTeamReport(
        created_at=datetime(2026, 5, 29, 12, 0, tzinfo=timezone.utc),
        description='test',
        pipeline=Pipeline.DYNAMIC,
        framework=Framework.OWASP_ASI,
        categories_tested=['ASI01'],
        tested_agents=['agent-a'],
        total_results=len(results),
        results=results,
        summary=compute_report_summary(results),
    )


# ---------------------------------------------------------------------------
# _build_token_usage_section
# ---------------------------------------------------------------------------


def test_build_token_usage_section_carries_known_cost():
    results = [
        _make_result(
            agent_key='agent-a',
            token_usage=TokenUsage(
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15,
                cache_creation_tokens=2,
                input_cost=0.001,
                output_cost=0.002,
                total_cost=0.003,
            ),
        ),
        _make_result(
            agent_key='agent-a',
            token_usage=TokenUsage(
                prompt_tokens=20,
                completion_tokens=10,
                total_tokens=30,
                cache_creation_tokens=1,
                input_cost=0.004,
                output_cost=0.006,
                total_cost=0.01,
            ),
        ),
    ]
    section = _build_token_usage_section(_make_report(results))
    assert section is not None
    overall = section.data['overall']
    assert overall['input_cost'] == 0.005
    assert overall['output_cost'] == 0.008
    assert round(overall['total_cost'], 6) == 0.013
    assert overall['cache_creation_tokens'] == 3

    per_agent = section.data['per_agent']
    assert len(per_agent) == 1
    assert round(per_agent[0]['total_cost'], 6) == 0.013


def test_build_token_usage_section_cost_none_when_unreported():
    results = [
        _make_result(
            agent_key='agent-a',
            token_usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        ),
        _make_result(
            agent_key='agent-a',
            token_usage=TokenUsage(prompt_tokens=20, completion_tokens=10, total_tokens=30),
        ),
    ]
    section = _build_token_usage_section(_make_report(results))
    assert section is not None
    overall = section.data['overall']
    assert overall['input_cost'] is None
    assert overall['output_cost'] is None
    assert overall['total_cost'] is None
    assert section.data['per_agent'][0]['total_cost'] is None


# ---------------------------------------------------------------------------
# HTML export
# ---------------------------------------------------------------------------


def test_export_html_shows_cost_cards_when_known():
    results = [
        _make_result(
            agent_key='agent-a',
            token_usage=TokenUsage(
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15,
                input_cost=0.0012,
                output_cost=0.0034,
                total_cost=0.0046,
            ),
        )
    ]
    html = export_html(_make_report(results))
    assert 'Input Cost' in html
    assert 'Output Cost' in html
    assert 'Total Cost' in html
    assert '$0.0046' in html


def test_export_html_omits_cost_cards_when_unknown():
    results = [
        _make_result(
            agent_key='agent-a',
            token_usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )
    ]
    html = export_html(_make_report(results))
    assert 'Input Cost' not in html
    assert 'Output Cost' not in html
    assert 'Total Cost' not in html
    assert '$0.00' not in html


# ---------------------------------------------------------------------------
# Markdown export
# ---------------------------------------------------------------------------


def test_export_markdown_shows_cost_rows_when_known():
    results = [
        _make_result(
            agent_key='agent-a',
            token_usage=TokenUsage(
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15,
                input_cost=0.0012,
                output_cost=0.0034,
                total_cost=0.0046,
            ),
        )
    ]
    md = export_markdown(_make_report(results))
    assert 'Input Cost' in md
    assert 'Output Cost' in md
    assert 'Total Cost' in md
    assert '$0.0046' in md


def test_export_markdown_omits_cost_rows_when_unknown():
    results = [
        _make_result(
            agent_key='agent-a',
            token_usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )
    ]
    md = export_markdown(_make_report(results))
    assert 'Input Cost' not in md
    assert 'Output Cost' not in md
    assert 'Total Cost' not in md
    assert '$0.00' not in md


def test_export_marks_partial_coverage_on_per_agent_costs():
    """Per-agent cost columns are lower bounds when only some calls were priced.

    The overall Total Cost row already carried the "(N of M calls)" qualifier;
    the per-agent breakdown rendered a bare figure from the same partial data.
    agent-b is fully priced so the overall label reads "2 of 3" — "1 of 2" can
    only come from agent-a's row.
    """
    results = [
        _make_result(
            agent_key='agent-a',
            token_usage=TokenUsage(
                prompt_tokens=10, completion_tokens=5, total_tokens=15, total_cost=0.5, calls=1, priced_calls=1
            ),
        ),
        _make_result(
            agent_key='agent-a',
            token_usage=TokenUsage(prompt_tokens=20, completion_tokens=10, total_tokens=30, calls=1),
        ),
        _make_result(
            agent_key='agent-b',
            token_usage=TokenUsage(
                prompt_tokens=4, completion_tokens=2, total_tokens=6, total_cost=0.25, calls=1, priced_calls=1
            ),
        ),
    ]
    report = _make_report(results)
    md, html = export_markdown(report), export_html(report)
    for rendered in (md, html):
        assert '$0.5000 (1 of 2 calls)' in rendered  # agent-a row, partial
        assert '$0.2500 (' not in rendered  # agent-b fully priced: no qualifier
    # Overall total: markdown puts the qualifier on the value, HTML on the label.
    assert '$0.7500 (2 of 3 calls)' in md
    assert 'Total Cost (2 of 3 calls)' in html
