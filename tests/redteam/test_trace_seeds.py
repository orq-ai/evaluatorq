"""Tests for trace-backed red-team seed datapoints and selection."""

from __future__ import annotations

from typing import Any

import pytest

from evaluatorq import DataPoint
from evaluatorq.contracts import AgentResponse, AgentTarget, ConversationHistoryMode, Message
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
async def test_trace_seed_skips_import_error_but_keeps_the_rest_of_the_batch() -> None:
    """One unreadable trace out of many is a skip, not a batch-killing raise (finding 3)."""
    from evaluatorq.redteam.traces import datapoints_from_traces

    broken = Trace(trace_id='broken', import_error='span payload was malformed')
    good = _trace()

    rows = await datapoints_from_traces([broken, good])

    assert len(rows) == 1
    assert rows[0].inputs['source_trace_id'] == 'trace-1'


@pytest.mark.asyncio
async def test_trace_seed_rejects_batch_when_every_trace_failed_to_import() -> None:
    from evaluatorq.redteam.traces import datapoints_from_traces

    trace = Trace(trace_id='broken', import_error='span payload was malformed')

    with pytest.raises(ValueError, match='failed to import'):
        await datapoints_from_traces([trace])


@pytest.mark.asyncio
async def test_trace_seed_skips_a_trace_with_no_user_turn() -> None:
    """A trace with no seedable turn is a per-trace skip, not a batch kill (finding 3)."""
    from evaluatorq.redteam.traces import datapoints_from_traces

    no_user = Trace(
        trace_id='no-user',
        input_messages=[Message(role='system', content='system')],
        output_messages=[Message(role='assistant', content='answer')],
    )
    good = _trace()

    rows = await datapoints_from_traces([no_user, good])

    assert len(rows) == 1
    assert rows[0].inputs['source_trace_id'] == 'trace-1'


@pytest.mark.asyncio
async def test_trace_seed_rejects_batch_when_no_trace_has_a_user_turn() -> None:
    from evaluatorq.redteam.traces import datapoints_from_traces

    trace = Trace(
        trace_id='no-user',
        input_messages=[Message(role='system', content='system')],
        output_messages=[Message(role='assistant', content='answer')],
    )

    with pytest.raises(ValueError, match='seedable turn'):
        await datapoints_from_traces([trace])


@pytest.mark.asyncio
async def test_last_assistant_seed_skips_a_trace_with_no_assistant_turn() -> None:
    from evaluatorq.redteam.traces import datapoints_from_traces

    no_assistant = Trace(
        trace_id='no-assistant',
        input_messages=[Message(role='user', content='question')],
        output_messages=[Message(role='tool', content='tool result')],
    )
    good = _trace()

    rows = await datapoints_from_traces([no_assistant, good], start_from='last_assistant')

    assert len(rows) == 1
    assert rows[0].inputs['source_trace_id'] == 'trace-1'


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


def _last_assistant_seed() -> DataPoint:
    return DataPoint(
        inputs={
            'trace_seed_messages': [FIRST_USER, {'role': 'assistant', 'content': 'Your order is on the way.'}],
            'trace_start_from': 'last_assistant',
            'source_trace_id': 'trace-1',
        }
    )


class _TargetOwnedHistory(_Target):
    history_mode = ConversationHistoryMode.TARGET

    async def get_agent_context(self) -> AgentContext:
        raise AssertionError('target context must not be fetched for a rejected trace seed')


@pytest.mark.asyncio
async def test_trace_seed_validation_rejects_non_trace_datapoints() -> None:
    from evaluatorq.redteam.runner import red_team

    with pytest.raises(ValueError, match='trace seed'):
        await red_team(_Target(), datapoints=[DataPoint(inputs={'query': 'not a trace'})], save=SaveMode.NONE)


@pytest.mark.asyncio
async def test_last_assistant_rejects_target_owned_history_before_pipeline() -> None:
    from unittest.mock import AsyncMock, patch

    from evaluatorq.redteam.exceptions import RedTeamError
    from evaluatorq.redteam.runner import red_team

    with (
        patch('evaluatorq.redteam.runner._run_dynamic_or_hybrid', new_callable=AsyncMock) as pipeline,
        patch('evaluatorq.redteam.runner.resolve_backend', side_effect=AssertionError('backend must not resolve')),
    ):
        with pytest.raises(RedTeamError, match='caller-owned history'):
            await red_team(
                _TargetOwnedHistory(),
                datapoints=[_last_assistant_seed()],
                recommendations=False,
                generate_executive_summary=False,
                save=SaveMode.NONE,
            )

    pipeline.assert_not_awaited()


@pytest.mark.asyncio
async def test_last_assistant_accepts_caller_owned_direct_target() -> None:
    from unittest.mock import AsyncMock, patch

    from evaluatorq.redteam.runner import red_team
    from tests.redteam.test_runner import _make_report, _run_result

    report = _make_report(target='_Target')
    with patch(
        'evaluatorq.redteam.runner._run_dynamic_or_hybrid',
        new_callable=AsyncMock,
        return_value=_run_result(report),
    ) as pipeline:
        result = await red_team(
            _Target(),
            datapoints=[_last_assistant_seed()],
            recommendations=False,
            generate_executive_summary=False,
            save=SaveMode.NONE,
        )

    assert result is report
    pipeline.assert_awaited_once()


@pytest.mark.asyncio
async def test_last_assistant_rejects_hosted_agent_string_target() -> None:
    """A hosted ``agent:`` target keeps history server-side; no string target can replay last_assistant.

    ``ORQAgentTarget.history_mode`` is ``TARGET`` and its ``respond()`` only ever
    sends the latest message, so this used to raise deep inside the pipeline
    after credentials, context discovery, strategy planning and target
    construction had all already run and billed. The rejection must happen
    up front instead, before the pipeline is ever awaited.
    """
    from unittest.mock import AsyncMock, patch

    from evaluatorq.redteam.exceptions import RedTeamError
    from evaluatorq.redteam.runner import red_team

    with (
        patch('evaluatorq.redteam.runner._run_dynamic_or_hybrid', new_callable=AsyncMock) as pipeline,
        patch('evaluatorq.redteam.runner.resolve_backend', side_effect=AssertionError('backend must not resolve')),
    ):
        with pytest.raises(RedTeamError, match='caller-owned history'):
            await red_team(
                'agent:test',
                datapoints=[_last_assistant_seed()],
                recommendations=False,
                generate_executive_summary=False,
                save=SaveMode.NONE,
            )

    pipeline.assert_not_awaited()


@pytest.mark.asyncio
async def test_last_assistant_rejects_target_owned_string_before_backend() -> None:
    from unittest.mock import AsyncMock, patch

    from evaluatorq.redteam.exceptions import RedTeamError
    from evaluatorq.redteam.runner import red_team

    with (
        patch('evaluatorq.redteam.runner._run_dynamic_or_hybrid', new_callable=AsyncMock) as pipeline,
        patch('evaluatorq.redteam.runner.resolve_backend', side_effect=AssertionError('backend must not resolve')),
    ):
        with pytest.raises(RedTeamError, match='caller-owned history'):
            await red_team(
                'deployment:test',
                datapoints=[_last_assistant_seed()],
                recommendations=False,
                generate_executive_summary=False,
                save=SaveMode.NONE,
            )

    pipeline.assert_not_awaited()


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


@pytest.mark.asyncio
async def test_red_team_rejects_explicitly_empty_datapoints() -> None:
    """``datapoints=[]`` passes ``is not None`` but selects nothing; reject it at the boundary (finding 2)."""
    from evaluatorq.redteam.runner import red_team

    with pytest.raises(ValueError, match='datapoints=\\[\\]'):
        await red_team(_Target(), datapoints=[], save=SaveMode.NONE)


def test_trace_seed_expansion_drops_full_trace_payload() -> None:
    """Only the three keys anything reads back are carried; the rest of the imported

    transcript (full messages, recorded_output, retrievals, trace_metadata) must not
    be copied into every seed x strategy row (finding 4).
    """
    from evaluatorq.redteam.adaptive.pipeline import expand_trace_seed_datapoints

    seed = DataPoint(
        inputs={
            'trace_seed_messages': [FIRST_USER],
            'trace_start_from': 'first_user',
            'source_trace_id': 'trace-1',
            'messages': [FIRST_USER, {'role': 'assistant', 'content': 'full production transcript'}],
            'recorded_output': [{'role': 'assistant', 'content': 'full production transcript'}],
            'retrievals': ['doc-1', 'doc-2'],
            'trace_metadata': {'session': 'abc'},
        }
    )
    strategy = DataPoint(inputs={'id': 'attack-1', 'category': 'ASI01', 'strategy': {}})

    rows = expand_trace_seed_datapoints([seed], [strategy])

    assert len(rows) == 1
    row_inputs = rows[0].inputs
    assert row_inputs['trace_seed_messages'] == [FIRST_USER]
    assert row_inputs['trace_start_from'] == 'first_user'
    assert row_inputs['source_trace_id'] == 'trace-1'
    for dropped_key in ('messages', 'recorded_output', 'retrievals', 'trace_metadata'):
        assert dropped_key not in row_inputs


def test_trace_seed_expansion_seed_slice_wins_over_attack_inputs() -> None:
    """The seed identifies which trace this row replays; the seed's slice wins the merge."""
    from evaluatorq.redteam.adaptive.pipeline import expand_trace_seed_datapoints

    seed = DataPoint(
        inputs={
            'trace_seed_messages': [FIRST_USER],
            'trace_start_from': 'first_user',
            'source_trace_id': 'trace-1',
        }
    )
    # An attack row that happens to carry a stray key of the same name must not win.
    strategy = DataPoint(inputs={'id': 'attack-1', 'category': 'ASI01', 'strategy': {}, 'source_trace_id': 'wrong'})

    rows = expand_trace_seed_datapoints([seed], [strategy])

    assert rows[0].inputs['source_trace_id'] == 'trace-1'


def test_pipeline_trace_seed_read_raises_on_missing_start_from() -> None:
    """`trace_seed_messages` without `trace_start_from` must raise, never default to first_user (finding 5)."""
    from evaluatorq.redteam.adaptive.pipeline import _trace_seed_from_inputs

    with pytest.raises(ValueError, match='trace_start_from'):
        _trace_seed_from_inputs({'trace_seed_messages': [FIRST_USER]})


def test_pipeline_trace_seed_read_returns_none_for_a_non_trace_row() -> None:
    """A dynamic row that never was a trace seed must not raise (pipeline's read is optional)."""
    from evaluatorq.redteam.adaptive.pipeline import _trace_seed_from_inputs

    messages, start_from = _trace_seed_from_inputs({'category': 'ASI01'})

    assert messages is None
    assert start_from is None


@pytest.mark.asyncio
async def test_scorer_prepends_seed_context_for_judge() -> None:
    """The judge must see the imported transcript a last_assistant attack continues (finding 6)."""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, patch

    from evaluatorq.contracts import AgentResponse, TextOutputItem
    from evaluatorq.redteam.adaptive.pipeline import create_dynamic_evaluator
    from evaluatorq.redteam.contracts import AttackEvaluationResult, AttackOutput, Turn

    output = AttackOutput(
        vulnerability='goal_hijacking',
        seed_context=[Message(role='assistant', content='Your order is on the way.')],
        turns=[
            Turn(
                attacker=AgentResponse(output=[TextOutputItem(text='continue the order', annotations=[])]),
                target=AgentResponse(output=[TextOutputItem(text='sure, here is the refund', annotations=[])]),
            )
        ],
    )
    judge = AsyncMock(return_value=AttackEvaluationResult(passed=True, explanation='ok', evaluator_id='x'))

    with patch('evaluatorq.redteam.adaptive.pipeline.OWASPEvaluator') as cls:
        cls.return_value.evaluate_vulnerability = judge
        scorer = create_dynamic_evaluator(llm_client=AsyncMock())['scorer']
        await scorer({'data': SimpleNamespace(inputs={}), 'output': output})

    judge.assert_awaited_once()
    assert judge.await_args is not None
    passed_messages = judge.await_args.kwargs['messages']
    assert passed_messages[0]['role'] == 'assistant'
    assert 'Your order is on the way.' in passed_messages[0]['content']
    assert passed_messages[-1]['content'] == 'continue the order'


def test_last_assistant_seed_must_end_with_an_assistant_turn() -> None:
    """The attack is appended as a user turn, so a user-ending prefix doubles up."""
    from evaluatorq.redteam.traces import TRACE_SEED_MESSAGES_KEY, TRACE_START_FROM_KEY, parse_trace_seed

    inputs = {
        TRACE_SEED_MESSAGES_KEY: [
            {'role': 'assistant', 'content': 'Your order is on the way.'},
            {'role': 'user', 'content': 'thanks'},
        ],
        TRACE_START_FROM_KEY: 'last_assistant',
    }
    with pytest.raises(ValueError, match='must end with an assistant message'):
        parse_trace_seed(inputs, label='datapoint[0]')


def test_last_assistant_seed_accepts_an_assistant_ending_prefix() -> None:
    from evaluatorq.redteam.traces import TRACE_SEED_MESSAGES_KEY, TRACE_START_FROM_KEY, parse_trace_seed

    inputs = {
        TRACE_SEED_MESSAGES_KEY: [
            {'role': 'user', 'content': 'where is my order?'},
            {'role': 'assistant', 'content': 'Your order is on the way.'},
        ],
        TRACE_START_FROM_KEY: 'last_assistant',
    }
    messages, start_from = parse_trace_seed(inputs, label='datapoint[0]')
    assert messages is not None and messages[-1].role == 'assistant'
    assert start_from is not None and start_from.value == 'last_assistant'
