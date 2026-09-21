"""Tests for trace-backed red-team seed datapoints and selection."""

from __future__ import annotations

from typing import Any

import pytest

from evaluatorq import DataPoint
from evaluatorq.contracts import AgentResponse, AgentTarget, Message
from evaluatorq.redteam.contracts import (
    AgentContext,
    AttackStrategy,
    AttackTechnique,
    DeliveryMethod,
    Severity,
    SaveMode,
    Pipeline,
    TurnType,
    Vulnerability,
)
from evaluatorq.types import Trace


FIRST_USER: dict[str, Any] = {'role': 'user', 'content': 'Please summarize my order.'}


def _trace() -> Trace:
    return Trace(
        trace_id='trace-1',
        input_messages=[Message(**FIRST_USER)],
        output_messages=[Message(role='assistant', content='Your order is on the way.')],
    )


@pytest.mark.asyncio
async def test_trace_seed_defaults_to_first_user() -> None:
    from evaluatorq.redteam.traces import datapoints_from_traces

    rows = await datapoints_from_traces([_trace()])

    assert rows[0].inputs['trace_start_from'] == 'first_user'
    assert rows[0].inputs['trace_seed_messages'] == [FIRST_USER]
    assert rows[0].inputs['source_trace_id'] == 'trace-1'


@pytest.mark.asyncio
async def test_trace_seed_can_continue_after_last_assistant() -> None:
    from evaluatorq.redteam.traces import datapoints_from_traces

    rows = await datapoints_from_traces([_trace()], start_from='last_assistant')

    assert rows[0].inputs['trace_start_from'] == 'last_assistant'
    assert rows[0].inputs['trace_seed_messages'][-1]['role'] == 'assistant'


@pytest.mark.asyncio
async def test_trace_seed_rejects_import_error() -> None:
    from evaluatorq.redteam.traces import datapoints_from_traces

    trace = Trace(trace_id='broken', import_error='span payload was malformed')

    with pytest.raises(ValueError, match='broken.*malformed'):
        await datapoints_from_traces([trace])


@pytest.mark.asyncio
async def test_trace_seed_requires_a_user_turn() -> None:
    from evaluatorq.redteam.traces import datapoints_from_traces

    trace = Trace(
        trace_id='no-user',
        input_messages=[Message(role='system', content='system')],
        output_messages=[Message(role='assistant', content='answer')],
    )

    with pytest.raises(ValueError, match='no-user.*user'):
        await datapoints_from_traces([trace])


@pytest.mark.asyncio
async def test_last_assistant_seed_requires_an_assistant_turn() -> None:
    from evaluatorq.redteam.traces import datapoints_from_traces

    trace = Trace(
        trace_id='no-assistant',
        input_messages=[Message(role='user', content='question')],
        output_messages=[Message(role='tool', content='tool result')],
    )

    with pytest.raises(ValueError, match='no-assistant.*assistant'):
        await datapoints_from_traces([trace], start_from='last_assistant')


class _Target(AgentTarget):
    async def respond(self, messages: list[Message]) -> AgentResponse:
        return AgentResponse(text='ok')

    def new(self) -> '_Target':
        return _Target()

    async def get_agent_context(self) -> AgentContext:
        return AgentContext(key='target')


def _seed() -> DataPoint:
    return DataPoint(
        inputs={
            'trace_seed_messages': [FIRST_USER],
            'trace_start_from': 'first_user',
            'source_trace_id': 'trace-1',
        }
    )


def _seed_without_identity() -> DataPoint:
    return DataPoint(
        inputs={
            'trace_seed_messages': [FIRST_USER],
            'trace_start_from': 'first_user',
        }
    )


@pytest.mark.asyncio
async def test_trace_seed_validation_rejects_non_trace_datapoints() -> None:
    from evaluatorq.redteam.runner import red_team

    with pytest.raises(ValueError, match='trace seed'):
        await red_team(_Target(), datapoints=[DataPoint(inputs={'query': 'not a trace'})], save=SaveMode.NONE)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'messages',
    [None, [], [{'role': 'not-a-message-role', 'content': 'bad'}], ['not a mapping']],
)
async def test_trace_seed_validation_rejects_invalid_messages(messages: Any) -> None:
    from evaluatorq.redteam.runner import red_team

    seed = _seed()
    seed.inputs['trace_seed_messages'] = messages

    with pytest.raises((TypeError, ValueError), match=r'datapoints\[0\].*trace_seed_messages'):
        await red_team(_Target(), datapoints=[seed], save=SaveMode.NONE)


@pytest.mark.asyncio
async def test_red_team_rejects_unknown_attack_technique_at_boundary() -> None:
    from evaluatorq.redteam.runner import red_team

    with pytest.raises(ValueError, match='attack technique'):
        await red_team(_Target(), attack_techniques=['not-a-technique'], save=SaveMode.NONE)


@pytest.mark.asyncio
async def test_attack_technique_filter_is_rejected_for_static_only_runs() -> None:
    from evaluatorq.redteam.runner import red_team

    with pytest.raises(ValueError, match='static'):
        await red_team(
            _Target(),
            mode=Pipeline.STATIC,
            attack_techniques=[AttackTechnique.DIRECT_INJECTION],
            save=SaveMode.NONE,
        )


@pytest.mark.asyncio
async def test_trace_seed_rejects_dataset_and_previous_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    from evaluatorq.redteam.runner import red_team

    with pytest.raises(ValueError, match='datapoints.*dataset'):
        await red_team(_Target(), datapoints=[_seed()], dataset=tmp_path / 'data.json', save=SaveMode.NONE)

    with pytest.raises(ValueError, match='datapoints.*previous_run'):
        await red_team(_Target(), datapoints=[_seed()], previous_run='missing', save=SaveMode.NONE)


def test_trace_seed_filter_types_are_public() -> None:
    from evaluatorq.redteam.traces import TraceStart

    assert TraceStart.FIRST_USER.value == 'first_user'
    assert AttackTechnique.DIRECT_INJECTION.value == 'direct-injection'
    assert DeliveryMethod.ROLE_PLAY.value == 'role-play'


def test_trace_seeds_cross_product_with_selected_attack_strategies() -> None:
    from evaluatorq.redteam.adaptive.pipeline import expand_trace_seed_datapoints

    strategy = AttackStrategy(
        category='ASI01',
        name='direct',
        description='direct attack',
        attack_technique=AttackTechnique.DIRECT_INJECTION,
        delivery_methods=[DeliveryMethod.ROLE_PLAY],
        turn_type=TurnType.SINGLE,
        severity=Severity.MEDIUM,
        objective_template='test objective',
    )
    attacks = [DataPoint(inputs={'id': 'attack-1', 'category': 'ASI01', 'strategy': strategy.model_dump(mode='json')})]

    rows = expand_trace_seed_datapoints([_seed()], attacks)

    assert len(rows) == 1
    assert rows[0].inputs['trace_seed_messages'] == [FIRST_USER]
    assert rows[0].inputs['trace_start_from'] == 'first_user'
    assert rows[0].inputs['source_trace_id'] == 'trace-1'
    assert rows[0].inputs['strategy']['attack_technique'] == 'direct-injection'


def test_trace_seed_ids_are_unique_for_duplicate_and_missing_source_identity() -> None:
    from evaluatorq.redteam.adaptive.pipeline import expand_trace_seed_datapoints

    strategy = DataPoint(inputs={'id': 'attack-1', 'category': 'ASI01', 'strategy': {}})
    seeds = [_seed(), _seed(), _seed_without_identity()]

    rows = expand_trace_seed_datapoints(seeds, [strategy])

    ids = [row.inputs['id'] for row in rows]
    assert len(set(ids)) == 3
    assert ids[0] == 'trace_trace-1_0_attack-1'
    assert ids[1] == 'trace_trace-1_1_attack-1'
    assert ids[2] == 'trace_seed_2_attack-1'


@pytest.mark.asyncio
async def test_attack_technique_filter_is_applied_before_trace_cross_product() -> None:
    from evaluatorq.redteam.adaptive.strategy_planner import plan_strategies_for_vulnerabilities

    strategies, _, _ = await plan_strategies_for_vulnerabilities(
        agent_context=AgentContext(key='target'),
        vulnerabilities=[Vulnerability.PROMPT_INJECTION],
        llm_client=None,
        attack_model='test-model',
        max_turns=5,
        max_per_category=None,
        generate_additional_strategies=False,
        generated_strategy_count=0,
        attack_techniques={AttackTechnique.DIRECT_INJECTION},
    )

    assert all(strategy.attack_technique is AttackTechnique.DIRECT_INJECTION for strategy in strategies[next(iter(strategies))])
