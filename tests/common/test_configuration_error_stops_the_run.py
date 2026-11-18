"""A configuration error ends the run where it is raised; it is not recorded on every item.

`LLMConfigurationError` means no call can succeed with this client, model or setting. The handlers that isolate one
failing item (a row, an attack, a trace, a cluster, a judge) let it through, and still isolate every other error. Each
test checks both directions.
"""

from __future__ import annotations

from typing import Any, cast

import pytest

from evaluatorq.common.llm_client import LLMConfigurationError, ModelClientMismatchError
from evaluatorq.types import DataPoint

WRONG_MODEL = ModelClientMismatchError("An Anthropic client needs a Claude model id, got 'openai/gpt-6-sol'.")


# ---------------------------------------------------------------------------
# The core runner: a job, an evaluator
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_job_that_hits_a_configuration_error_ends_the_run_and_any_other_error_stays_on_its_row() -> None:
    from evaluatorq.processings import process_job

    async def misconfigured(_data: DataPoint, _row: int) -> dict[str, Any]:
        raise WRONG_MODEL

    async def broken(_data: DataPoint, _row: int) -> dict[str, Any]:
        raise RuntimeError('provider down')

    with pytest.raises(ModelClientMismatchError):
        await process_job(misconfigured, DataPoint(inputs={'text': 'hi'}), row_index=0)

    result = await process_job(broken, DataPoint(inputs={'text': 'hi'}), row_index=0)
    assert result.error == 'provider down'


@pytest.mark.asyncio
async def test_a_decorated_job_that_hits_a_configuration_error_ends_the_run_with_the_original_error() -> None:
    from evaluatorq import job
    from evaluatorq.processings import process_job

    @job('decorated')
    async def misconfigured(_data: DataPoint, _row: int) -> str:
        raise WRONG_MODEL

    @job('decorated')
    async def broken(_data: DataPoint, _row: int) -> str:
        raise RuntimeError('provider down')

    with pytest.raises(ModelClientMismatchError) as info:
        await process_job(misconfigured, DataPoint(inputs={'text': 'hi'}), row_index=0)
    assert info.value is WRONG_MODEL, 'the caller gets the error itself, not the job wrapper'

    result = await process_job(broken, DataPoint(inputs={'text': 'hi'}), row_index=0)
    assert result.error == 'provider down'
    assert result.job_name == 'decorated'


@pytest.mark.asyncio
async def test_an_evaluator_that_hits_a_configuration_error_ends_the_run() -> None:
    from evaluatorq.processings import process_job

    async def fine(_data: DataPoint, _row: int) -> dict[str, Any]:
        return {'name': 'fine', 'output': 'answer'}

    async def misconfigured_scorer(_params: Any) -> Any:
        raise WRONG_MODEL

    async def broken_scorer(_params: Any) -> Any:
        raise RuntimeError('scorer down')

    with pytest.raises(ModelClientMismatchError):
        await process_job(
            fine, DataPoint(inputs={'text': 'hi'}), row_index=0, evaluators=[{'name': 'judge', 'scorer': misconfigured_scorer}]
        )

    result = await process_job(
        fine, DataPoint(inputs={'text': 'hi'}), row_index=0, evaluators=[{'name': 'judge', 'scorer': broken_scorer}]
    )
    assert result.error is None
    assert [score.error for score in result.evaluator_scores or []] == ['scorer down']


@pytest.mark.asyncio
async def test_a_whole_evaluation_stops_on_a_configuration_error_and_cancels_the_other_rows() -> None:
    import asyncio

    from evaluatorq import evaluatorq

    started: list[int] = []
    finished: list[int] = []

    async def job_fn(data: DataPoint, row: int) -> dict[str, Any]:
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


# ---------------------------------------------------------------------------
# Jury
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_judge_that_hits_a_configuration_error_is_not_turned_into_a_prediction() -> None:
    from evaluatorq.common.jury import _call_prediction

    async def misconfigured(_model: str) -> Any:
        raise WRONG_MODEL

    async def broken(_model: str) -> Any:
        raise RuntimeError('judge down')

    with pytest.raises(ModelClientMismatchError):
        await _call_prediction(misconfigured, 'openai/gpt-6-sol')

    prediction = await _call_prediction(broken, 'openai/gpt-6-sol')
    assert prediction.error == 'judge down'


# ---------------------------------------------------------------------------
# Simulation
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


class _User:
    def update_context(self, **_kwargs: Any) -> None:
        return None

    async def generate_first_message(self) -> str:
        return 'hello'

    async def respond_async(self, _messages: Any, **_kwargs: Any) -> str:
        return 'more'

    def reset_usage(self) -> None:
        return None

    def get_usage(self) -> Any:
        from evaluatorq.contracts import TokenUsage

        return TokenUsage()


class _Judge:
    def __init__(self, error: BaseException) -> None:
        self.error = error

    async def evaluate(self, _messages: Any) -> Any:
        raise self.error

    def reset_usage(self) -> None:
        return None

    def get_usage(self) -> Any:
        from evaluatorq.contracts import TokenUsage

        return TokenUsage()


def _runner(error: BaseException) -> Any:
    from evaluatorq.simulation.runner.simulation import SimulationRunner

    async def target(_messages: Any) -> str:
        return 'fine'

    return SimulationRunner(
        target=target, model='test-model', user_simulator=cast('Any', _User()), judge=cast('Any', _Judge(error))
    )


@pytest.mark.asyncio
async def test_a_conversation_that_hits_a_configuration_error_raises_and_any_other_error_is_an_error_result() -> None:
    from evaluatorq.simulation.contracts import Scenario, TerminatedBy

    scenario = Scenario(name='S', goal='fix my bug')

    with pytest.raises(ModelClientMismatchError):
        await _runner(WRONG_MODEL).run(persona=_persona(), scenario=scenario, first_message='Hello')

    result = await _runner(RuntimeError('judge down')).run(persona=_persona(), scenario=scenario, first_message='Hello')
    assert result.terminated_by == TerminatedBy.error


@pytest.mark.asyncio
async def test_a_batch_that_hits_a_configuration_error_raises_and_any_other_error_stays_on_its_row() -> None:
    from evaluatorq.simulation.contracts import Scenario, SimulationDatapoint, TerminatedBy

    datapoints = [
        SimulationDatapoint(
            id=f'dp{i}', persona=_persona(), scenario=Scenario(name='S', goal='g'), user_system_prompt='', first_message='hi'
        )
        for i in range(3)
    ]

    with pytest.raises(ModelClientMismatchError):
        await _runner(WRONG_MODEL).run_batch(datapoints, max_turns=1)

    results = await _runner(RuntimeError('judge down')).run_batch(datapoints, max_turns=1)
    assert [r.terminated_by for r in results] == [TerminatedBy.error] * 3


# ---------------------------------------------------------------------------
# Insights: one handler per stage
# ---------------------------------------------------------------------------


def _raising(error: BaseException) -> Any:
    async def call(*_args: Any, **_kwargs: Any) -> Any:
        raise error

    return call


@pytest.mark.asyncio
async def test_a_summary_that_hits_a_configuration_error_ends_the_stage(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    from evaluatorq.insights import summarize as summarize_module
    from evaluatorq.insights.cache import InsightsCache
    from tests.insights.test_summarize import fake_client, make_trace

    cache = InsightsCache(tmp_path / 'cache.sqlite')
    try:
        monkeypatch.setattr(summarize_module, 'generate_structured', _raising(WRONG_MODEL))
        with pytest.raises(ModelClientMismatchError):
            await summarize_module.summarize_traces([make_trace('t')], client=fake_client(), model='m', cache=cache)

        monkeypatch.setattr(summarize_module, 'generate_structured', _raising(RuntimeError('boom')))
        result = await summarize_module.summarize_traces([make_trace('t')], client=fake_client(), model='m', cache=cache)
        assert result['t'] == 'summary: boom'
    finally:
        cache.close()


@pytest.mark.asyncio
async def test_a_cluster_description_that_hits_a_configuration_error_ends_the_stage(monkeypatch: pytest.MonkeyPatch) -> None:
    from evaluatorq.insights import describe as describe_module
    from evaluatorq.insights.describe import ClusterName
    from tests.insights.test_describe import fake_client

    children = {0: [ClusterName(name='a', description='b')]}

    monkeypatch.setattr(describe_module, 'generate_structured', _raising(WRONG_MODEL))
    with pytest.raises(ModelClientMismatchError):
        await describe_module.describe_clusters({0: ['hello']}, neighbours={0: []}, dimension='intent', client=fake_client(), model='m')
    with pytest.raises(ModelClientMismatchError):
        await describe_module.describe_top_level(children, client=fake_client(), model='m')

    monkeypatch.setattr(describe_module, 'generate_structured', _raising(RuntimeError('boom')))
    named = await describe_module.describe_clusters({0: ['hello']}, neighbours={0: []}, dimension='intent', client=fake_client(), model='m')
    top = await describe_module.describe_top_level(children, client=fake_client(), model='m')
    assert 'boom' in cast('str', named[0])
    assert 'boom' in cast('str', top[0])


@pytest.mark.asyncio
async def test_a_merge_pair_that_hits_a_configuration_error_ends_the_stage(monkeypatch: pytest.MonkeyPatch) -> None:
    from evaluatorq.insights import merge as merge_module
    from tests.insights.test_merge import _names, fake_client

    async def run(error: BaseException) -> Any:
        monkeypatch.setattr(merge_module, 'run_classify', _raising(error))
        return await merge_module.merge_similar(
            _names(0, 1), examples={}, neighbours={0: [1]}, client=fake_client(), model='m', threshold=0.5
        )

    with pytest.raises(ModelClientMismatchError):
        await run(WRONG_MODEL)

    result = await run(RuntimeError('unexpected router failure'))
    assert result.n_failed == 1


@pytest.mark.asyncio
async def test_a_label_request_that_hits_a_configuration_error_ends_the_stage(monkeypatch: pytest.MonkeyPatch) -> None:
    from evaluatorq.insights import labeling as labeling_module
    from tests.insights.test_labeling import SENTIMENT, _client, make_trace

    async def run(error: BaseException) -> Any:
        monkeypatch.setattr(labeling_module, 'run_classify', _raising(error))
        return await labeling_module.label_traces(
            [make_trace('a'), make_trace('b')], labels=[SENTIMENT], compiled=None, client=_client(), model='typesafe/jev-latest'
        )

    with pytest.raises(ModelClientMismatchError):
        await run(WRONG_MODEL)

    outcomes = await run(ValueError('malformed reply'))
    assert [outcome.error for outcome in outcomes] == ['malformed reply', 'malformed reply']


def test_the_error_the_handlers_let_through_is_the_configuration_error_family() -> None:
    assert issubclass(ModelClientMismatchError, LLMConfigurationError)
    assert issubclass(LLMConfigurationError, ValueError), 'callers that catch ValueError still catch it'


# ---------------------------------------------------------------------------
# Red team: the attacker turn, the tool-chain planner, the attack job
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_an_attacker_call_that_hits_a_configuration_error_ends_the_attack_by_raising() -> None:
    from unittest.mock import AsyncMock

    from evaluatorq.redteam.adaptive.orchestrator import MultiTurnOrchestrator
    from evaluatorq.redteam.contracts import AgentContext
    from tests.redteam.test_orchestrator import _make_strategy

    async def attack(error: BaseException) -> Any:
        llm = AsyncMock()
        llm.chat.completions.create = AsyncMock(side_effect=error)
        orchestrator = MultiTurnOrchestrator(llm_client=llm, model='azure/gpt-5-mini')
        return await orchestrator.run_attack(
            target=AsyncMock(),
            strategy=_make_strategy(),
            objective='Test',
            agent_context=AgentContext(key='test_agent'),
            max_turns=3,
        )

    with pytest.raises(ModelClientMismatchError):
        await attack(WRONG_MODEL)

    result = await attack(RuntimeError('provider down'))
    assert result.error_type == 'llm_error'


@pytest.mark.asyncio
async def test_a_tool_chain_plan_that_hits_a_configuration_error_does_not_fall_back() -> None:
    from unittest.mock import AsyncMock

    from tests.redteam.test_tool_chaining import _ctx_with_tools, _make_orchestrator, _tc_strategy

    misconfigured = _make_orchestrator(parse=AsyncMock(side_effect=WRONG_MODEL))
    with pytest.raises(ModelClientMismatchError):
        await misconfigured._plan_tool_chain(_tc_strategy(), 'x', _ctx_with_tools(), 5)

    broken = _make_orchestrator(parse=AsyncMock(side_effect=RuntimeError('LLM down')))
    plan, _ = await broken._plan_tool_chain(_tc_strategy(), 'x', _ctx_with_tools(), 5)
    assert plan is None, 'any other planner failure still falls back to a plan-free attack'


@pytest.mark.asyncio
async def test_an_attack_job_that_hits_a_configuration_error_raises_and_any_other_error_is_reported_on_the_row() -> None:
    from unittest.mock import AsyncMock, MagicMock, patch

    from evaluatorq.redteam.adaptive.pipeline import create_dynamic_redteam_job
    from tests.redteam import test_dynamic_job as helpers

    async def run(error: BaseException) -> Any:
        with (
            patch(helpers._PATCH_REDTEAM_SPAN, side_effect=helpers._noop_span_ctx),
            patch(helpers._PATCH_SET_SPAN_ATTRS),
            patch(helpers._PATCH_ATTACK_SPAN_ATTRS),
            patch(f'{helpers._PIPELINE_MOD}.MultiTurnOrchestrator') as orchestrator_class,
            patch('evaluatorq.redteam.adaptive.orchestrator._get_active_progress', return_value=None),
        ):
            orchestrator = MagicMock()
            orchestrator.run_attack = AsyncMock(side_effect=error)
            orchestrator_class.return_value = orchestrator
            job_fn = create_dynamic_redteam_job(
                agent_key='test-agent',
                agent_context=helpers._make_agent_context(),
                backend=helpers._make_target_factory(helpers._make_target()),
                attack_llm_client=MagicMock(),
            )
            return await helpers._call_dynamic_job(job_fn, helpers._make_datapoint(strategy=helpers._make_strategy()))

    from evaluatorq.job_helper import JobError

    # The job decorator wraps what the job raises; the runner unwraps a configuration error (see the decorated-job test).
    with pytest.raises(JobError) as info:
        await run(WRONG_MODEL)
    assert info.value.original_error is WRONG_MODEL

    output = await run(RuntimeError('provider down'))
    assert output.get('error'), 'an ordinary failure is reported on the row, and the run goes on'


@pytest.mark.asyncio
async def test_a_dimension_that_hits_a_configuration_error_ends_the_run_with_one_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
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
