from __future__ import annotations

from evaluatorq.simulation.reports.executive_summary import build_sim_facts
from evaluatorq.simulation.types import SimulationResult, TerminatedBy, TokenUsage


def _result(*, goal: bool, rules: list[str]) -> SimulationResult:
    return SimulationResult(
        messages=[],
        terminated_by=TerminatedBy.judge if goal else TerminatedBy.max_turns,
        reason='PII was disclosed to the simulated user.',
        goal_achieved=goal,
        goal_completion_score=1.0 if goal else 0.0,
        rules_broken=rules,
        turn_count=3,
        token_usage=TokenUsage(),
        turn_metrics=[],
    )


def test_build_sim_facts_counts_and_dominant_failure():
    results = [
        _result(goal=True, rules=[]),
        _result(goal=False, rules=['must_not_reveal_pii']),
        _result(goal=False, rules=['must_not_reveal_pii']),
    ]
    facts = build_sim_facts(results)
    assert 'Total simulations: 3' in facts
    assert 'Goals achieved: 1' in facts
    assert 'must_not_reveal_pii' in facts


def test_build_sim_facts_empty_is_blank():
    assert build_sim_facts([]).strip() == ''
