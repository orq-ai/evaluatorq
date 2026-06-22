"""Filter-rebuild core: rebuild_filtered_report reproduces aggregates and
preserves multi-target structure. Guards Plan C filter path."""

from __future__ import annotations

from datetime import datetime, timezone

from evaluatorq.redteam.contracts import (
    AgentInfo,
    AttackInfo,
    AttackTechnique,
    DeliveryMethod,
    Framework,
    Pipeline,
    RedTeamReport,
    RedTeamResult,
    Severity,
    TurnType,
    UnifiedEvaluationResult,
)
from evaluatorq.redteam.reports.converters import compute_report_summary, rebuild_filtered_report
from evaluatorq.redteam.reports.sections import build_report_sections


def _make_result(
    category: str = 'ASI01',
    passed: bool | None = True,
    agent_key: str = 'agent-a',
    severity: Severity = Severity.MEDIUM,
) -> RedTeamResult:
    return RedTeamResult(
        attack=AttackInfo(
            id=f'{category}-{agent_key}-{passed}',
            category=category,
            framework=Framework.OWASP_ASI,
            attack_technique=AttackTechnique.INDIRECT_INJECTION,
            delivery_methods=[DeliveryMethod.DIRECT_REQUEST],
            turn_type=TurnType.SINGLE,
            severity=severity,
            source='test',
        ),
        agent=AgentInfo(key=agent_key),
        messages=[],
        vulnerable=passed is False,
        evaluation=UnifiedEvaluationResult(passed=passed, explanation='test') if passed is not None else None,
    )


def _make_report(results: list[RedTeamResult], tested_agents: list[str]) -> RedTeamReport:
    return RedTeamReport(
        created_at=datetime.now(tz=timezone.utc),
        description='Test report',
        pipeline=Pipeline.STATIC,
        framework=Framework.OWASP_ASI,
        categories_tested=sorted({r.attack.category for r in results}),
        tested_agents=tested_agents,
        total_results=len(results),
        results=results,
        summary=compute_report_summary(results),
    )


def test_noop_filter_reproduces_summary_and_sections():
    """Rebuilding with ALL results yields the same summary + section kinds."""
    results = [
        _make_result(category='ASI01', passed=True),
        _make_result(category='ASI01', passed=False),
        _make_result(category='LLM01', passed=None),
    ]
    report = _make_report(results, tested_agents=['agent-a'])

    rebuilt = rebuild_filtered_report(report, report.results)

    assert rebuilt.summary.model_dump() == report.summary.model_dump()
    assert rebuilt.total_results == report.total_results
    assert rebuilt.categories_tested == report.categories_tested
    assert [s.kind for s in build_report_sections(rebuilt)] == [s.kind for s in build_report_sections(report)]


def test_empty_filter_returns_empty_report() -> None:
    """Filtering to zero results yields an empty but valid report."""
    results = [
        _make_result(category='ASI01', passed=True),
        _make_result(category='ASI01', passed=False),
    ]
    report = _make_report(results, tested_agents=['agent-a'])

    rebuilt = rebuild_filtered_report(report, [])

    assert rebuilt.total_results == 0
    assert rebuilt.categories_tested == []
    assert rebuilt.summary.total_attacks == 0
    # Identity fields preserved.
    assert rebuilt.created_at == report.created_at
    assert rebuilt.pipeline == report.pipeline


def test_category_filter_shrinks_aggregates() -> None:
    """Filtering to a subset (drop LLM01) shrinks total_results, categories, and summary."""
    results = [
        _make_result(category='ASI01', passed=True),
        _make_result(category='ASI01', passed=False),
        _make_result(category='LLM01', passed=False),
    ]
    report = _make_report(results, tested_agents=['agent-a'])

    asi_only = [r for r in results if r.attack.category == 'ASI01']
    rebuilt = rebuild_filtered_report(report, asi_only)

    assert rebuilt.total_results == 2
    assert rebuilt.categories_tested == ['ASI01']
    assert rebuilt.summary.total_attacks == 2
    # Original metadata preserved.
    assert rebuilt.created_at == report.created_at
    assert rebuilt.pipeline == report.pipeline


def test_multi_target_structure_preserved_after_rebuild() -> None:
    """tested_agents/agent_contexts survive rebuild so agent sections still fire."""
    results = [
        _make_result(category='ASI01', passed=False, agent_key='agent-a'),
        _make_result(category='ASI01', passed=True, agent_key='agent-b'),
        _make_result(category='LLM01', passed=False, agent_key='agent-a'),
        _make_result(category='LLM01', passed=True, agent_key='agent-b'),
    ]
    report = _make_report(results, tested_agents=['agent-a', 'agent-b'])

    # Narrow to vulnerable-only — both agents still represented.
    vulnerable = [r for r in results if r.vulnerable]
    rebuilt = rebuild_filtered_report(report, vulnerable)

    assert rebuilt.tested_agents == ['agent-a', 'agent-b']
    assert rebuilt.agent_contexts == report.agent_contexts
    kinds = {s.kind for s in build_report_sections(rebuilt)}
    assert 'agent_comparison' in kinds
    assert 'agent_disagreements' in kinds
