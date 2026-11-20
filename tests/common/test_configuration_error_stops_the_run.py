"""A configuration error ends the run where it is raised; it is not recorded on every item.

`LLMConfigurationError` means no call can succeed with this client, model or setting. The handlers that isolate one
failing item (a row, an attack, a trace, a cluster, a judge) let it through, and still isolate every other error.
`test_a_handler_lets_a_configuration_error_through_and_isolates_any_other` checks both directions for each handler;
the tests after it cover what a whole run does once the error is through.
"""

from __future__ import annotations

import asyncio
import importlib
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from evaluatorq.common.llm_client import LLMConfigurationError, ModelClientMismatchError
from evaluatorq.types import DataPoint

WRONG_MODEL = ModelClientMismatchError("An Anthropic client needs a Claude model id, got 'openai/gpt-6-sol'.")
ROW = DataPoint(inputs={'text': 'hi'})


def _raising(error: BaseException) -> Any:
    async def call(*_args: Any, **_kwargs: Any) -> Any:
        raise error

    return call


# ---------------------------------------------------------------------------
# Simulation fakes
# ---------------------------------------------------------------------------


def _persona() -> Any:
    from evaluatorq.simulation.contracts import CommunicationStyle, Persona

    return Persona(
        name='Test User',
        patience=0.5,
        assertiveness=0.5,
        politeness=0.5,
        technical_level=0.5,
        communication_style=CommunicationStyle.casual,
        background='bg',
    )


def _datapoints(n: int) -> list[Any]:
    from evaluatorq.simulation.contracts import Scenario, SimulationDatapoint

    return [
        SimulationDatapoint(
            id=f'dp{i}', persona=_persona(), scenario=Scenario(name='S', goal='g'), user_system_prompt='', first_message='hi'
        )
        for i in range(n)
    ]


class _NoUsage:
    def reset_usage(self) -> None:
        return None

    def get_usage(self) -> Any:
        from evaluatorq.contracts import TokenUsage

        return TokenUsage()


class _User(_NoUsage):
    def update_context(self, **_kwargs: Any) -> None:
        return None

    async def generate_first_message(self) -> str:
        return 'hello'

    async def respond_async(self, _messages: Any, **_kwargs: Any) -> str:
        return 'more'


class _Judge(_NoUsage):
    def __init__(self, error: BaseException) -> None:
        self.error = error

    async def evaluate(self, _messages: Any) -> Any:
        raise self.error


def _runner(error: BaseException, target: Any = None) -> Any:
    from evaluatorq.simulation.runner.simulation import SimulationRunner

    async def fine(_messages: Any) -> str:
        return 'fine'

    return SimulationRunner(
        target=target or fine, model='test-model', user_simulator=cast('Any', _User()), judge=cast('Any', _Judge(error))
    )


# ---------------------------------------------------------------------------
# One case per handler: run it with an error, and check what an ordinary error leaves behind
# ---------------------------------------------------------------------------

Run = Callable[[BaseException, pytest.MonkeyPatch, Path], Awaitable[Any]]


@dataclass(frozen=True)
class Handler:
    run: Run
    isolated: Callable[[Any], bool]
    ordinary: BaseException = RuntimeError('boom')


async def _job(error: BaseException, *_: Any) -> Any:
    from evaluatorq.processings import process_job

    return await process_job(_raising(error), ROW, row_index=0)


async def _decorated_job(error: BaseException, *_: Any) -> Any:
    from evaluatorq import job
    from evaluatorq.processings import process_job

    return await process_job(job('decorated')(_raising(error)), ROW, row_index=0)


async def _evaluator(error: BaseException, *_: Any) -> Any:
    from evaluatorq.processings import process_job

    async def fine(_data: DataPoint, _row: int) -> dict[str, Any]:
        return {'name': 'fine', 'output': 'answer'}

    return await process_job(fine, ROW, row_index=0, evaluators=[{'name': 'judge', 'scorer': _raising(error)}])


async def _datapoint(error: BaseException, *_: Any) -> Any:
    from evaluatorq import evaluatorq

    async def job_fn(_data: DataPoint, _row: int) -> dict[str, Any]:
        return {'name': 'job', 'output': 'ok'}

    return await evaluatorq('unresolvable-row', data=[_raising(error)()], jobs=[job_fn], print_results=False)


async def _jury_judge(error: BaseException, *_: Any) -> Any:
    from evaluatorq.common.jury import _call_prediction

    return await _call_prediction(_raising(error), 'openai/gpt-6-sol')


async def _conversation(error: BaseException, *_: Any) -> Any:
    from evaluatorq.simulation.contracts import Scenario

    return await _runner(error).run(persona=_persona(), scenario=Scenario(name='S', goal='fix my bug'), first_message='Hello')


async def _conversation_batch(error: BaseException, *_: Any) -> Any:
    return await _runner(error).run_batch(_datapoints(3), max_turns=1)


async def _summary(error: BaseException, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Any:
    from evaluatorq.insights import summarize as summarize_module
    from evaluatorq.insights.cache import InsightsCache
    from tests.insights.test_summarize import fake_client, make_trace

    monkeypatch.setattr(summarize_module, 'generate_structured', _raising(error))
    cache = InsightsCache(tmp_path / f'{type(error).__name__}.sqlite')
    try:
        return await summarize_module.summarize_traces([make_trace('t')], client=fake_client(), model='m', cache=cache)
    finally:
        cache.close()


async def _cluster_description(error: BaseException, monkeypatch: pytest.MonkeyPatch, *_: Any) -> Any:
    from evaluatorq.insights import describe as describe_module
    from tests.insights.test_describe import fake_client

    monkeypatch.setattr(describe_module, 'generate_structured', _raising(error))
    return await describe_module.describe_clusters(
        {0: ['hello']}, neighbours={0: []}, dimension='intent', client=fake_client(), model='m'
    )


async def _top_level_description(error: BaseException, monkeypatch: pytest.MonkeyPatch, *_: Any) -> Any:
    from evaluatorq.insights import describe as describe_module
    from evaluatorq.insights.describe import ClusterName
    from tests.insights.test_describe import fake_client

    monkeypatch.setattr(describe_module, 'generate_structured', _raising(error))
    return await describe_module.describe_top_level(
        {0: [ClusterName(name='a', description='b')]}, client=fake_client(), model='m'
    )


async def _merge_pair(error: BaseException, monkeypatch: pytest.MonkeyPatch, *_: Any) -> Any:
    from evaluatorq.insights import merge as merge_module
    from tests.insights.test_merge import _names, fake_client

    monkeypatch.setattr(merge_module, 'run_classify', _raising(error))
    return await merge_module.merge_similar(
        _names(0, 1), examples={}, neighbours={0: [1]}, client=fake_client(), model='m', threshold=0.5
    )


async def _label_request(error: BaseException, monkeypatch: pytest.MonkeyPatch, *_: Any) -> Any:
    from evaluatorq.insights import labeling as labeling_module
    from tests.insights.test_labeling import SENTIMENT, _client, make_trace

    monkeypatch.setattr(labeling_module, 'run_classify', _raising(error))
    return await labeling_module.label_traces(
        [make_trace('a'), make_trace('b')], labels=[SENTIMENT], compiled=None, client=_client(), model='typesafe/jev-latest'
    )


async def _attacker_turn(error: BaseException, *_: Any) -> Any:
    from evaluatorq.redteam.adaptive.orchestrator import MultiTurnOrchestrator
    from evaluatorq.redteam.contracts import AgentContext
    from tests.redteam.test_orchestrator import _make_strategy

    llm = AsyncMock()
    llm.chat.completions.create = AsyncMock(side_effect=error)
    return await MultiTurnOrchestrator(llm_client=llm, model='azure/gpt-5-mini').run_attack(
        target=AsyncMock(), strategy=_make_strategy(), objective='Test', agent_context=AgentContext(key='test_agent'), max_turns=3
    )


async def _tool_chain_plan(error: BaseException, *_: Any) -> Any:
    from tests.redteam.test_tool_chaining import _ctx_with_tools, _make_orchestrator, _tc_strategy

    plan, _ = await _make_orchestrator(parse=AsyncMock(side_effect=error))._plan_tool_chain(
        _tc_strategy(), 'x', _ctx_with_tools(), 5
    )
    return plan


async def _attack_job(error: BaseException, *_: Any) -> Any:
    from evaluatorq.redteam.adaptive.pipeline import create_dynamic_redteam_job
    from tests.redteam import test_dynamic_job as helpers

    with (
        patch(helpers._PATCH_REDTEAM_SPAN, side_effect=helpers._noop_span_ctx),
        patch(helpers._PATCH_SET_SPAN_ATTRS),
        patch(helpers._PATCH_ATTACK_SPAN_ATTRS),
        patch(f'{helpers._PIPELINE_MOD}.MultiTurnOrchestrator') as orchestrator_class,
        patch('evaluatorq.redteam.adaptive.orchestrator._get_active_progress', return_value=None),
    ):
        orchestrator_class.return_value = MagicMock(run_attack=AsyncMock(side_effect=error))
        job_fn = create_dynamic_redteam_job(
            agent_key='test-agent',
            agent_context=helpers._make_agent_context(),
            backend=helpers._make_target_factory(helpers._make_target()),
            attack_llm_client=MagicMock(),
        )
        return await helpers._call_dynamic_job(job_fn, helpers._make_datapoint(strategy=helpers._make_strategy()))


def _terminated_by_error(results: Any) -> bool:
    from evaluatorq.simulation.contracts import TerminatedBy

    return all(result.terminated_by == TerminatedBy.error for result in (results if isinstance(results, list) else [results]))


HANDLERS = {
    'job': Handler(_job, lambda r: r.error == 'boom'),
    'decorated job': Handler(_decorated_job, lambda r: r.error == 'boom' and r.job_name == 'decorated'),
    'evaluator': Handler(_evaluator, lambda r: r.error is None and [s.error for s in r.evaluator_scores] == ['boom']),
    'datapoint': Handler(_datapoint, lambda rs: [str(r.error) for r in rs] == ['boom']),
    'jury judge': Handler(_jury_judge, lambda p: p.error == 'boom'),
    'conversation': Handler(_conversation, _terminated_by_error),
    'conversation batch': Handler(_conversation_batch, lambda rs: len(rs) == 3 and _terminated_by_error(rs)),
    'insights summary': Handler(_summary, lambda r: r['t'] == 'summary: boom'),
    'insights cluster description': Handler(_cluster_description, lambda r: 'boom' in r[0]),
    'insights top-level description': Handler(_top_level_description, lambda r: 'boom' in r[0]),
    'insights merge pair': Handler(_merge_pair, lambda r: r.n_failed == 1),
    'insights label request': Handler(_label_request, lambda r: [o.error for o in r] == ['boom'] * 2, ValueError('boom')),
    'red-team attacker turn': Handler(_attacker_turn, lambda r: r.error_type == 'llm_error'),
    # Any other planner failure still falls back to a plan-free attack.
    'red-team tool-chain plan': Handler(_tool_chain_plan, lambda plan: plan is None),
    'red-team attack job': Handler(_attack_job, lambda output: bool(output.get('error'))),
}


@pytest.mark.asyncio
@pytest.mark.parametrize('handler', HANDLERS.values(), ids=HANDLERS.keys())
async def test_a_handler_lets_a_configuration_error_through_and_isolates_any_other(
    handler: Handler, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from evaluatorq.job_helper import JobError

    with pytest.raises(Exception) as info:  # noqa: PT011 - identity is asserted below
        await handler.run(WRONG_MODEL, monkeypatch, tmp_path)
    # The error itself. The `@job` decorator wraps what a job raises; the runner unwraps it (the 'decorated job' case).
    raised = info.value.original_error if isinstance(info.value, JobError) else info.value
    assert raised is WRONG_MODEL

    assert handler.isolated(await handler.run(handler.ordinary, monkeypatch, tmp_path))


def test_the_error_the_handlers_let_through_is_the_configuration_error_family() -> None:
    assert issubclass(ModelClientMismatchError, LLMConfigurationError)
    assert issubclass(LLMConfigurationError, ValueError), 'callers that catch ValueError still catch it'


# ---------------------------------------------------------------------------
# What a whole run does once the error is through
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_whole_evaluation_stops_on_a_configuration_error_and_cancels_the_other_rows() -> None:
    from evaluatorq import evaluatorq

    started: list[int] = []
    finished: list[int] = []

    async def job_fn(_data: DataPoint, row: int) -> dict[str, Any]:
        started.append(row)
        if row == 0:
            raise WRONG_MODEL
        await asyncio.sleep(0.2)
        finished.append(row)
        return {'name': 'job', 'output': 'ok'}

    with pytest.raises(ModelClientMismatchError):
        await evaluatorq(
            'stops-on-configuration-error',
            data=[DataPoint(inputs={'i': i}) for i in range(6)],
            jobs=[job_fn],
            datapoint_parallelism=3,
            print_results=False,
        )

    assert finished == [], 'no other row ran to the end: the ones in flight were cancelled'
    assert len(started) < 6, 'the rows still waiting never started'


async def _run_streamed(monkeypatch: pytest.MonkeyPatch, batches: Any, processing: Any) -> None:
    from evaluatorq.types import DatasetIdInput

    module = importlib.import_module('evaluatorq.evaluatorq')
    monkeypatch.setattr(module, 'setup_orq_client', lambda _api_key: object())
    monkeypatch.setattr(module, 'fetch_dataset_batches', batches)
    monkeypatch.setattr(module, 'process_data_point', processing)
    monkeypatch.setenv('ORQ_API_KEY', 'test-key')

    async def job(_data: DataPoint, _row: int) -> dict[str, Any]:
        return {'name': 'job', 'output': 'output'}

    await module.evaluatorq(
        'streamed', data=DatasetIdInput(dataset_id='dataset'), jobs=[job], print_results=False, _send_results=False
    )


@pytest.mark.asyncio
@pytest.mark.parametrize('failing_rows', [{0}, {0, 1, 2}])
async def test_a_streamed_dataset_stops_on_the_first_configuration_error_and_raises_it_unwrapped(
    monkeypatch: pytest.MonkeyPatch, failing_rows: set[int]
) -> None:
    from evaluatorq.fetch_data import DataPointBatch

    started: list[int] = []
    finished: list[int] = []

    async def two_batches(*_args: object, **_kwargs: object) -> Any:
        for batch in range(2):
            yield DataPointBatch(
                datapoints=[DataPoint(inputs={'row': batch * 4 + i}) for i in range(4)], has_more=batch == 0, batch_number=batch + 1
            )
            await asyncio.sleep(0.05)

    async def processing(_data: object, row: int, *_args: object, **_kwargs: object) -> list[object]:
        started.append(row)
        if row in failing_rows:
            raise WRONG_MODEL
        await asyncio.sleep(0.3)
        finished.append(row)
        return []

    with pytest.raises(LLMConfigurationError) as error:
        await _run_streamed(monkeypatch, two_batches, processing)

    # The error itself, even when several rows raised it: a caller catching LLMConfigurationError sees it.
    assert error.value is WRONG_MODEL
    assert finished == [], 'rows in flight were cancelled'
    assert max(started) < 4, 'the second batch never started'


@pytest.mark.asyncio
async def test_a_streamed_run_logs_the_other_errors_it_does_not_raise(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from evaluatorq.fetch_data import DataPointBatch

    async def one_batch_then_a_fetch_failure(*_args: object, **_kwargs: object) -> Any:
        yield DataPointBatch(datapoints=[DataPoint(inputs={'row': i}) for i in range(3)], has_more=True, batch_number=1)
        await asyncio.sleep(0.05)
        raise ConnectionError('dataset page 2 unavailable')

    async def processing(_data: object, row: int, *_args: object, **_kwargs: object) -> list[object]:
        if row == 0:
            raise RuntimeError('row 0 broke for another reason')
        if row == 1:
            await asyncio.sleep(0.01)
            raise WRONG_MODEL
        await asyncio.sleep(0.3)
        return []

    with caplog.at_level('WARNING'), pytest.raises(LLMConfigurationError) as error:
        await _run_streamed(monkeypatch, one_batch_then_a_fetch_failure, processing)

    assert error.value is WRONG_MODEL
    discarded = [r.getMessage() for r in caplog.records if 'Discarding error while a configuration error' in r.getMessage()]
    # The other row's error and the fetch failure are each logged once; the cancelled row is ours and is not.
    assert sorted(discarded) == [
        'Discarding error while a configuration error stops the run: ConnectionError: dataset page 2 unavailable',
        'Discarding error while a configuration error stops the run: RuntimeError: row 0 broke for another reason',
    ]


@pytest.mark.asyncio
async def test_a_simulation_batch_stops_at_the_first_configuration_error_and_starts_no_further_conversation() -> None:
    started: list[str] = []

    async def target(_messages: Any) -> str:
        started.append('conversation')
        return 'fine'

    with pytest.raises(ModelClientMismatchError):
        await _runner(WRONG_MODEL, target).run_batch(_datapoints(10), max_turns=1, max_concurrency=2)

    # Two rows hold the two slots when the first judge call fails; the other eight never get one.
    assert 1 <= len(started) <= 2


@pytest.mark.asyncio
async def test_an_insights_dimension_that_hits_a_configuration_error_ends_the_run_with_one_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from types import SimpleNamespace

    from openai import AsyncOpenAI

    from evaluatorq.insights import pipeline
    from tests.insights import test_pipeline as helpers

    async def run(error: BaseException) -> tuple[Any, list[str]]:
        built: list[str] = []

        async def build(dimension: str, *_args: Any, **_kwargs: Any) -> Any:
            built.append(dimension)
            raise error

        traces = [helpers._trace(i) for i in range(6)]
        monkeypatch.setattr(pipeline, '_build_dimension', build)
        monkeypatch.setattr(pipeline, 'resolve_orq_client', lambda: object())
        monkeypatch.setattr(pipeline, 'resolve_population', helpers._resolve(traces))
        monkeypatch.setattr(pipeline, 'label_traces', helpers._label(traces))
        monkeypatch.setattr(pipeline, 'summarize_traces', helpers._summarize(traces))
        finished = await pipeline.insights(
            helpers._population(),
            dimensions=('intent', 'failure'),
            labels=(),
            runs_dir=tmp_path / f'runs-{type(error).__name__}',
            llm_client=AsyncOpenAI(api_key='sk-test', max_retries=0),
            embedding_client=cast('Any', SimpleNamespace()),
        )
        return finished, built

    stopped, built = await run(WRONG_MODEL)
    assert built == ['intent'], 'the second dimension was not attempted'
    assert stopped.status == 'error'
    assert len(stopped.stage_failures) == 1

    isolated, built = await run(RuntimeError('provider down'))
    assert built == ['intent', 'failure'], 'an ordinary failure still leaves the other dimension its turn'
    assert [failure.dimension for failure in isolated.stage_failures] == ['intent', 'failure']
