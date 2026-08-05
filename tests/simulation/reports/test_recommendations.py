"""Tests for simulation/reports/recommendations.py.

Covers the selective trigger filter (benign failures produce no LLM calls),
the generation path with a mocked LLM client, and rendering of the
recommendations section in the Markdown / HTML exports.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from evaluatorq.contracts import Message, TokenUsage
from evaluatorq.simulation.reports import export_html, export_markdown
from evaluatorq.simulation.reports.recommendations import (
    find_triggers,
    generate_recommendations,
)
from evaluatorq.simulation.reports.sections import build_report_sections
from evaluatorq.simulation.types import (
    SimulationRecommendation,
    SimulationResult,
    TerminatedBy,
    TurnMetrics,
)


def _make_result(
    *,
    goal_achieved: bool = True,
    rules_broken: list[str] | None = None,
    terminated_by: TerminatedBy = TerminatedBy.judge,
    metadata: dict[str, Any] | None = None,
    factual_accuracy: float | None = None,
) -> SimulationResult:
    meta = {'persona': 'Alice', 'scenario': 'Billing'}
    meta.update(metadata or {})
    return SimulationResult(
        messages=[
            Message(role='user', content='hi'),
            Message(role='assistant', content='response'),
        ],
        terminated_by=terminated_by,
        reason='judge decided',
        goal_achieved=goal_achieved,
        goal_completion_score=1.0 if goal_achieved else 0.2,
        rules_broken=rules_broken or [],
        turn_count=2,
        token_usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        turn_metrics=[
            TurnMetrics(
                turn_number=1,
                token_usage=TokenUsage(),
                judge_reason='ok',
                factual_accuracy=factual_accuracy,
            )
        ],
        metadata=meta,
    )


def _mock_client(payload: dict[str, Any] | Exception) -> MagicMock:
    client = MagicMock()
    if isinstance(payload, Exception):
        client.chat.completions.create = AsyncMock(side_effect=payload)
    else:
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = json.dumps(payload)
        client.chat.completions.create = AsyncMock(return_value=response)
    return client


# ---------------------------------------------------------------------------
# Trigger filter
# ---------------------------------------------------------------------------


def test_clean_run_has_no_triggers():
    assert find_triggers(_make_result(goal_achieved=True)) == []


def test_benign_max_turns_has_no_triggers():
    result = _make_result(goal_achieved=False, terminated_by=TerminatedBy.max_turns)
    assert find_triggers(result) == []


def test_errored_run_has_no_triggers():
    result = _make_result(
        goal_achieved=False,
        rules_broken=['leaked internal data'],
        metadata={'error': 'boom'},
    )
    assert find_triggers(result) == []


def test_rules_broken_triggers():
    result = _make_result(goal_achieved=False, rules_broken=['leaked internal data'])
    assert ('rule_broken', 'leaked internal data') in find_triggers(result)


def test_failed_criterion_triggers():
    result = _make_result(
        goal_achieved=False,
        metadata={
            'criteria_meta': [
                {'id': 'c0', 'description': 'explain the charge', 'type': 'must_happen', 'passed': False},
            ]
        },
    )
    assert ('criterion_failed', 'explain the charge') in find_triggers(result)


def test_low_factual_accuracy_triggers():
    result = _make_result(goal_achieved=True, factual_accuracy=0.2)
    triggers = find_triggers(result)
    assert any(kind == 'low_factual_accuracy' for kind, _ in triggers)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_triggered_results_makes_no_llm_calls():
    client = _mock_client({'suggestions': ['unused']})
    recs = await generate_recommendations([_make_result()], client, 'test-model')
    assert recs == []
    client.chat.completions.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_broken_rule_produces_targeted_recommendation():
    client = _mock_client({'suggestions': ['Add a system prompt rule forbidding internal data']})
    result = _make_result(
        goal_achieved=False,
        rules_broken=['leaked internal data'],
        metadata={'datapoint_id': 'dp-7'},
    )
    recs = await generate_recommendations([_make_result(), result], client, 'test-model')

    assert len(recs) == 1
    rec = recs[0]
    assert rec.result_index == 1
    assert rec.datapoint_id == 'dp-7'
    assert rec.persona == 'Alice'
    assert rec.scenario == 'Billing'
    assert rec.triggers == ['rule_broken: leaked internal data']
    assert rec.suggestions == ['Add a system prompt rule forbidding internal data']

    call_kwargs = client.chat.completions.create.await_args.kwargs
    user_prompt = call_kwargs['messages'][1]['content']
    assert 'leaked internal data' in user_prompt
    # Reasoning models reject non-default temperatures; omit unless asked for.
    assert 'temperature' not in call_kwargs


@pytest.mark.asyncio
async def test_llm_failure_skips_result_without_raising():
    client = _mock_client(RuntimeError('boom'))
    result = _make_result(goal_achieved=False, rules_broken=['rude tone'])
    recs = await generate_recommendations([result], client, 'test-model')
    assert recs == []


@pytest.mark.asyncio
async def test_max_results_caps_llm_calls():
    client = _mock_client({'suggestions': ['fix it']})
    results = [_make_result(goal_achieved=False, rules_broken=['r']) for _ in range(5)]
    recs = await generate_recommendations(results, client, 'test-model', max_results=2)
    assert len(recs) == 2
    assert client.chat.completions.create.await_count == 2


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _sample_recommendation() -> SimulationRecommendation:
    return SimulationRecommendation(
        result_index=0,
        datapoint_id='dp-1',
        persona='Alice',
        scenario='Billing',
        triggers=['rule_broken: leaked internal data'],
        suggestions=['Add a guardrail blocking internal identifiers'],
    )


def test_sections_include_recommendations_only_when_provided():
    results = [_make_result()]
    kinds = [s.kind for s in build_report_sections(results)]
    assert 'recommendations' not in kinds

    kinds = [s.kind for s in build_report_sections(results, recommendations=[_sample_recommendation()])]
    assert 'recommendations' in kinds


def test_markdown_renders_recommendations():
    md = export_markdown([_make_result()], recommendations=[_sample_recommendation()])
    assert 'Remediation Suggestions' in md
    assert 'Add a guardrail blocking internal identifiers' in md
    assert 'Rule broken: leaked internal data' in md


def test_html_renders_recommendations():
    html = export_html([_make_result()], recommendations=[_sample_recommendation()])
    assert 'Remediation Suggestions' in html
    assert 'Add a guardrail blocking internal identifiers' in html


def test_html_omits_recommendations_section_when_absent():
    html = export_html([_make_result()])
    assert 'Remediation Suggestions' not in html


def test_find_triggers_resolves_rule_ids_and_dedupes():
    """Broken-rule ids resolve to the criterion description, and a rule that
    is also a failed criterion is reported once, not twice."""
    from evaluatorq.simulation.reports.recommendations import find_triggers

    r = _make_result()
    r.rules_broken = ['criteria_0']
    r.metadata['criteria_meta'] = [
        {'id': 'criteria_0', 'description': 'No internal identifiers leak.', 'type': 'must_not_happen', 'passed': False},
    ]
    triggers = find_triggers(r)
    descs = [evidence for _, evidence in triggers]
    assert descs.count('No internal identifiers leak.') == 1
    assert 'criteria_0' not in descs
