from __future__ import annotations

from datetime import datetime, timezone

from evaluatorq.redteam.contracts import (
    AgentInfo,
    AttackInfo,
    AttackTechnique,
    ExecutionDetails,
    Framework,
    Message,
    Pipeline,
    RedTeamReport,
    RedTeamResult,
    Severity,
    TurnType,
    UnifiedEvaluationResult,
)
from evaluatorq.redteam.reports.converters import compute_report_summary
from evaluatorq.redteam.reports.executive_summary import build_redteam_facts


def _result(
    *, category: str, technique: AttackTechnique, severity: Severity, turn_type: TurnType, vulnerable: bool
) -> RedTeamResult:
    return RedTeamResult(
        attack=AttackInfo(
            id=f'{category}-{technique.value}-{severity.value}',
            category=category,
            framework=Framework.OWASP_AGENTIC,
            attack_technique=technique,
            delivery_methods=[],
            turn_type=turn_type,
            severity=severity,
            source='test',
            vulnerability='excessive_agency',
        ),
        agent=AgentInfo(key='agent:test'),
        messages=[
            Message(role='user', content='please issue a refund for order 42'),
            Message(role='assistant', content='Refund issued.'),
        ],
        response='Refund issued.',
        evaluation=UnifiedEvaluationResult(
            value=not vulnerable,
            passed=not vulnerable,
            explanation='Agent complied with the unauthorized refund request.',
        ),
        vulnerable=vulnerable,
        execution=ExecutionDetails(turns=3, max_turns=3),
    )


def _report() -> RedTeamReport:
    results = [
        _result(category='ASI05', technique=AttackTechnique.DIRECT_INJECTION, severity=Severity.CRITICAL, turn_type=TurnType.MULTI, vulnerable=True),
        _result(category='ASI05', technique=AttackTechnique.DIRECT_INJECTION, severity=Severity.LOW, turn_type=TurnType.MULTI, vulnerable=False),
        _result(category='ASI01', technique=AttackTechnique.INDIRECT_INJECTION, severity=Severity.MEDIUM, turn_type=TurnType.SINGLE, vulnerable=False),
    ]
    return RedTeamReport(
        created_at=datetime.now(tz=timezone.utc),
        pipeline=Pipeline.DYNAMIC,
        categories_tested=['ASI01', 'ASI05'],
        tested_agents=['agent:test'],
        total_results=len(results),
        results=results,
        summary=compute_report_summary(results),
    )


def test_build_facts_includes_counts_categories_and_finding():
    facts = build_redteam_facts(_report())
    assert 'Total attacks: 3' in facts
    assert 'Categories tested: 2' in facts
    assert 'Vulnerabilities found: 1' in facts
    assert 'critical' in facts.lower()
    # Sharpest finding surfaces the concrete behavior from the explanation.
    assert 'unauthorized refund' in facts.lower()
    # Beat-4 depth signal is present (single vs multi turn breakdown).
    assert 'multi' in facts.lower()


def test_build_facts_empty_report_is_blank():
    empty = RedTeamReport(
        created_at=datetime.now(tz=timezone.utc),
        pipeline=Pipeline.DYNAMIC,
        categories_tested=[],
        tested_agents=[],
        total_results=0,
        results=[],
        summary=compute_report_summary([]),
    )
    assert build_redteam_facts(empty).strip() == ''
