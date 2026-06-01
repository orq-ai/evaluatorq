"""Tests for simulation/reports/sections.py (RES-846)."""

from __future__ import annotations

import pytest

from evaluatorq.contracts import Message, TokenUsage
from evaluatorq.simulation.reports.sections import build_report_sections, _criteria_rows
from evaluatorq.simulation.types import SimulationResult, TerminatedBy, TurnMetrics


@pytest.fixture
def make_result():
    def _make(
        *,
        goal_achieved=True,
        score=1.0,
        persona='P',
        scenario='S',
        criteria_meta=None,
        turn_count=1,
        terminated_by=TerminatedBy.judge,
    ):
        meta = {'persona': persona, 'scenario': scenario}
        if criteria_meta is not None:
            meta['criteria_meta'] = criteria_meta
        return SimulationResult(
            messages=[],
            terminated_by=terminated_by,
            reason='r',
            goal_achieved=goal_achieved,
            goal_completion_score=score,
            rules_broken=[],
            turn_count=turn_count,
            turn_metrics=[],
            token_usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            metadata=meta,
        )

    return _make


def _make_result(
    *,
    persona: str = 'Tester',
    scenario: str = 'Smoke',
    goal_achieved: bool = True,
    goal_completion_score: float = 1.0,
    rules_broken: list[str] | None = None,
    turn_count: int = 1,
    tokens: tuple[int, int, int] = (10, 5, 15),
    evaluator_scores: dict[str, float] | None = None,
    error: str | None = None,
    terminated_by: TerminatedBy = TerminatedBy.judge,
) -> SimulationResult:
    metadata: dict[str, object] = {
        'persona': persona,
        'scenario': scenario,
        'model': 'gpt-4o-mini',
    }
    if evaluator_scores is not None:
        metadata['evaluator_scores'] = evaluator_scores
    if error is not None:
        metadata['error'] = error
    return SimulationResult(
        messages=[
            Message(role='user', content='hi'),
            Message(role='assistant', content='hello'),
        ],
        terminated_by=terminated_by,
        reason='done',
        goal_achieved=goal_achieved,
        goal_completion_score=goal_completion_score,
        rules_broken=rules_broken or [],
        turn_count=turn_count,
        token_usage=TokenUsage(
            prompt_tokens=tokens[0],
            completion_tokens=tokens[1],
            total_tokens=tokens[2],
        ),
        turn_metrics=[TurnMetrics(turn_number=1, token_usage=TokenUsage(), judge_reason='ok')],
        metadata=metadata,
    )


def test_build_report_sections_emits_core_sections():
    results = [_make_result(), _make_result(goal_achieved=False, goal_completion_score=0.2)]
    sections = build_report_sections(results)
    kinds = [s.kind for s in sections]
    # Mandatory kinds, in order.
    for kind in (
        'summary',
        'persona_breakdown',
        'scenario_breakdown',
        'judge_verdicts',
        'turn_metrics',
        'token_usage',
        'individual_results',
    ):
        assert kind in kinds, f'missing {kind}: {kinds}'
    # Optional kinds: not present without data.
    assert 'evaluator_scores' not in kinds
    assert 'errors' not in kinds


def test_summary_section_computes_success_rate_and_tokens():
    results = [
        _make_result(goal_achieved=True, tokens=(10, 5, 15)),
        _make_result(goal_achieved=False, tokens=(20, 10, 30)),
    ]
    sections = build_report_sections(results)
    summary = next(s for s in sections if s.kind == 'summary')
    assert summary.data['total_conversations'] == 2
    assert summary.data['goals_achieved'] == 1
    assert summary.data['success_rate'] == 0.5
    assert summary.data['total_tokens'] == 45


def test_persona_breakdown_aggregates_per_persona():
    results = [
        _make_result(persona='A', goal_achieved=True),
        _make_result(persona='A', goal_achieved=False, goal_completion_score=0.0),
        _make_result(persona='B', goal_achieved=True),
    ]
    sections = build_report_sections(results)
    persona = next(s for s in sections if s.kind == 'persona_breakdown')
    rows = {r['persona']: r for r in persona.data['rows']}
    assert rows['A']['conversations'] == 2
    assert rows['A']['goals_achieved'] == 1
    assert rows['A']['success_rate'] == 0.5
    assert rows['B']['success_rate'] == 1.0


def test_overview_section_lists_personas_and_scenario_criteria(make_result):
    results = [
        make_result(
            persona='A',
            scenario='X',
            criteria_meta=[
                {'id': 'c0', 'description': 'must do', 'type': 'must_happen', 'passed': True},
                {'id': 'c1', 'description': 'must not', 'type': 'must_not_happen', 'passed': True},
            ],
        ),
        make_result(persona='A', scenario='X'),
        make_result(persona='B', scenario='Y'),
    ]
    sections = build_report_sections(results)
    overview = next(s for s in sections if s.kind == 'overview')
    assert overview.data['total_conversations'] == 3
    personas = {p['name']: p['conversations'] for p in overview.data['personas']}
    assert personas == {'A': 2, 'B': 1}
    scenario_names = {s['name'] for s in overview.data['scenarios']}
    assert scenario_names == {'X', 'Y'}
    x = next(s for s in overview.data['scenarios'] if s['name'] == 'X')
    types = {c['type'] for c in x['criteria']}
    assert types == {'must_happen', 'must_not_happen'}


def test_judge_verdicts_section_counts_terminated_by_and_rules():
    results = [
        _make_result(terminated_by=TerminatedBy.judge, rules_broken=['rude', 'off-topic']),
        _make_result(terminated_by=TerminatedBy.max_turns, rules_broken=['rude']),
    ]
    sections = build_report_sections(results)
    verdict = next(s for s in sections if s.kind == 'judge_verdicts')
    assert verdict.data['terminated_by']['judge'] == 1
    assert verdict.data['terminated_by']['max_turns'] == 1
    assert verdict.data['rules_broken']['rude'] == 2
    assert verdict.data['rules_broken']['off-topic'] == 1


def test_evaluator_scores_section_present_when_scores_attached():
    results = [
        _make_result(evaluator_scores={'goal_achieved': 1.0, 'criteria_met': 0.8}),
        _make_result(evaluator_scores={'goal_achieved': 0.0, 'criteria_met': 0.5}),
    ]
    sections = build_report_sections(results)
    evaluators = next(s for s in sections if s.kind == 'evaluator_scores')
    rows = {r['evaluator']: r for r in evaluators.data['rows']}
    assert rows['goal_achieved']['mean_score'] == 0.5
    assert rows['criteria_met']['mean_score'] == 0.65
    assert rows['goal_achieved']['min_score'] == 0.0
    assert rows['goal_achieved']['max_score'] == 1.0


def test_errors_section_present_when_failures_present():
    results = [
        _make_result(error='rate limit'),
        _make_result(terminated_by=TerminatedBy.error, error='upstream timeout'),
        _make_result(),
    ]
    sections = build_report_sections(results)
    errors = next(s for s in sections if s.kind == 'errors')
    assert errors.data['total_errored'] == 2
    assert errors.data['by_message']['rate limit'] == 1


def test_persona_breakdown_excludes_errored_runs_from_achieved():
    """Per-persona success rate must not count errored runs as achieved.

    Regression: a result with both ``goal_achieved=True`` and an error
    metadata key would previously count in the persona breakdown's
    ``goals_achieved`` but not in the summary's. The per-persona rate
    could paradoxically exceed the overall rate.
    """
    results = [
        _make_result(persona='A', goal_achieved=True),  # achieved
        _make_result(persona='A', goal_achieved=True, error='oh no'),  # errored, not achieved
    ]
    sections = build_report_sections(results)
    persona = next(s for s in sections if s.kind == 'persona_breakdown')
    row_a = next(r for r in persona.data['rows'] if r['persona'] == 'A')
    assert row_a['conversations'] == 2
    assert row_a['goals_achieved'] == 1
    assert row_a['success_rate'] == 0.5

    summary = next(s for s in sections if s.kind == 'summary')
    # The persona rate must not exceed the overall rate for the same data.
    assert (
        row_a['success_rate'] <= summary.data['success_rate'] + 1e-9
        or summary.data['success_rate'] == row_a['success_rate']
    )
    assert summary.data['goals_achieved'] == row_a['goals_achieved']


def test_scenario_breakdown_excludes_errored_runs_from_achieved():
    """Same partition discipline as persona breakdown."""
    results = [
        _make_result(scenario='X', goal_achieved=True),
        _make_result(scenario='X', goal_achieved=True, error='boom'),
    ]
    sections = build_report_sections(results)
    scenario = next(s for s in sections if s.kind == 'scenario_breakdown')
    row_x = next(r for r in scenario.data['rows'] if r['scenario'] == 'X')
    assert row_x['goals_achieved'] == 1
    assert row_x['success_rate'] == 0.5


def test_summary_partitions_achieved_failed_errored_disjointly():
    """An errored run never counts as achieved, and goals_failed is never negative.

    Regression: a result with both ``goal_achieved=True`` and a metadata
    ``error`` would previously be double-counted (in achieved AND errored),
    producing ``goals_failed = total - achieved - errored < 0``.
    """
    results = [
        _make_result(goal_achieved=True),  # clean win
        _make_result(goal_achieved=False),  # clean loss
        _make_result(goal_achieved=True, error='weird'),  # achieved + error -> errored
        _make_result(terminated_by=TerminatedBy.error),  # terminated_by=error
    ]
    sections = build_report_sections(results)
    summary = next(s for s in sections if s.kind == 'summary')
    assert summary.data['goals_achieved'] == 1
    assert summary.data['errors'] == 2
    assert summary.data['goals_failed'] == 1
    assert (
        summary.data['goals_achieved'] + summary.data['errors'] + summary.data['goals_failed']
        == summary.data['total_conversations']
    )


def test_errors_section_count_matches_summary_section_count():
    """Both sections must agree on what 'errored' means.

    Regression: the summary counted only metadata['error'] results, while the
    errors section also counted ``terminated_by == error``.
    """
    results = [
        _make_result(error='rate limit'),
        _make_result(terminated_by=TerminatedBy.error),  # no metadata error
        _make_result(),
    ]
    sections = build_report_sections(results)
    summary = next(s for s in sections if s.kind == 'summary')
    errors = next(s for s in sections if s.kind == 'errors')
    assert summary.data['errors'] == errors.data['total_errored'] == 2


def test_individual_results_section_carries_transcript_and_meta():
    results = [_make_result(persona='A', goal_achieved=True)]
    sections = build_report_sections(results)
    individual = next(s for s in sections if s.kind == 'individual_results')
    entry = individual.data['entries'][0]
    assert entry['persona'] == 'A'
    assert entry['goal_achieved'] is True
    assert entry['transcript'][0]['role'] == 'user'
    assert entry['transcript'][1]['role'] == 'assistant'


# ---------------------------------------------------------------------------
# Task 4.1 — criteria_meta accessor + summary verdict
# ---------------------------------------------------------------------------


def test_summary_section_has_hero_kpis(make_result):
    results = [make_result(goal_achieved=True, score=1.0), make_result(goal_achieved=False, score=0.0)]
    summary = next(s for s in build_report_sections(results) if s.kind == 'summary')
    d = summary.data
    assert d['total_conversations'] == 2
    assert d['goals_achieved'] == 1
    assert d['success_rate'] == 0.5
    assert 'verdict' in d  # "pass" | "warn" | "fail"


def test_criteria_rows_uses_meta_ids_not_descriptions(make_result):
    r = make_result(
        goal_achieved=False,
        score=0.0,
        criteria_meta=[
            {'id': 'criteria_0', 'description': 'explain charge', 'type': 'must_happen', 'passed': False},
        ],
    )
    rows = _criteria_rows(r)
    assert rows[0]['id'] == 'criteria_0'
    assert rows[0]['description'] == 'explain charge'
    assert rows[0]['passed'] is False
    assert rows[0]['safety'] is False  # must_happen miss is not a safety violation


# ---------------------------------------------------------------------------
# Task 4.2 — failures_first section builder
# ---------------------------------------------------------------------------


def test_failures_first_lists_only_failures_with_descriptions(make_result):
    results = [
        make_result(goal_achieved=True, score=1.0, persona='A', scenario='X'),
        make_result(
            goal_achieved=False,
            score=0.0,
            persona='B',
            scenario='Y',
            criteria_meta=[
                {'id': 'criteria_0', 'description': 'explain charge', 'type': 'must_happen', 'passed': False}
            ],
        ),
    ]
    section = next(s for s in build_report_sections(results) if s.kind == 'failures_first')
    assert len(section.data['rows']) == 1
    row = section.data['rows'][0]
    assert row['persona'] == 'B' and row['scenario'] == 'Y'
    assert row['violated'] == ['explain charge']
    assert 'criteria_0' not in str(row['violated'])  # description shown, not id


# ---------------------------------------------------------------------------
# Task 4.3 — 5 new section builders
# ---------------------------------------------------------------------------


def test_new_section_kinds_present(make_result):
    results = [
        make_result(
            goal_achieved=False,
            score=0.2,
            persona='A',
            scenario='X',
            criteria_meta=[{'id': 'criteria_0', 'description': 'd0', 'type': 'must_happen', 'passed': False}],
        ),
        make_result(goal_achieved=True, score=0.9, persona='A', scenario='Y', criteria_meta=[]),
    ]
    kinds = {s.kind for s in build_report_sections(results)}
    for k in [
        'criteria_heatmap',
        'persona_scenario_heatmap',
        'score_distribution',
        'turn_quality_timeline',
        'failure_mode',
    ]:
        assert k in kinds


def test_persona_scenario_heatmap_matrix(make_result):
    results = [
        make_result(goal_achieved=True, persona='A', scenario='X'),
        make_result(goal_achieved=False, persona='A', scenario='Y'),
    ]
    s = next(x for x in build_report_sections(results) if x.kind == 'persona_scenario_heatmap')
    assert s.data['personas'] == ['A']
    assert set(s.data['scenarios']) == {'X', 'Y'}
    # success-rate cell for (A, X) == 1.0, (A, Y) == 0.0
    cell = {(c['persona'], c['scenario']): c['success_rate'] for c in s.data['cells']}
    assert cell[('A', 'X')] == 1.0 and cell[('A', 'Y')] == 0.0
