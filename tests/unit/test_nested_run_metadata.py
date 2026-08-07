"""Run-id correlation must survive into a nested ``evaluatorq()`` call.

Both the red-team and simulation surfaces bind ``evaluatorq_pipeline`` +
``evaluatorq_run_id`` at their entrypoint and then route every datapoint
through the shared ``evaluatorq()`` execution engine. Because the binding is
carried on a ``ContextVar`` (see ``evaluatorq.common.thread_context``) and not
threaded as an explicit parameter, nothing currently proves the ContextVar
actually survives the ``asyncio.create_task`` boundary that ``evaluatorq()``
uses to schedule each datapoint's job. If that propagation silently breaks,
every existing test still passes (none of them run inference from inside a
nested ``evaluatorq()`` call) while the feature goes hollow for the vast
majority of LLM calls in a real run.
"""

from __future__ import annotations

# ruff: noqa: S101
from typing import Any, cast

import pytest

from evaluatorq import evaluatorq
from evaluatorq.common.reports.executive_summary import AsyncChatCompletionsClient, generate_executive_summary
from evaluatorq.common.thread_context import (
    evaluatorq_pipeline,
    evaluatorq_run_id,
    pipeline_metadata,
)
from evaluatorq.types import DataPoint


class _Message:
    def __init__(self) -> None:
        self.content = 'summary text'


class _Choice:
    def __init__(self) -> None:
        self.message = _Message()


class _Response:
    def __init__(self) -> None:
        self.choices = [_Choice()]


class _Completions:
    def __init__(self, requests: list[dict[str, Any]]) -> None:
        self._requests = requests

    async def create(self, **kwargs: Any) -> _Response:
        self._requests.append(kwargs)
        return _Response()


class _Client:
    def __init__(self, requests: list[dict[str, Any]]) -> None:
        self.chat = type('_Chat', (), {'completions': _Completions(requests)})()


@pytest.mark.asyncio
async def test_llm_call_inside_nested_evaluatorq_carries_outer_run_id() -> None:
    """An LLM call issued from a job running inside a nested ``evaluatorq()``
    call must carry the run id bound by the OUTER (red-team/sim) scope, purely
    via ContextVar propagation across the ``asyncio.create_task`` boundary.
    """
    requests: list[dict[str, Any]] = []
    client = cast(AsyncChatCompletionsClient, cast(object, _Client(requests)))

    async def job(_data: DataPoint, _row: int):
        await generate_executive_summary('facts', llm_client=client, model='gpt-4o')
        return {'name': 'noop', 'output': 'ok'}

    with evaluatorq_pipeline('red_teaming'), evaluatorq_run_id('outer-run'):
        await evaluatorq(
            'nested-run-metadata',
            data=[DataPoint(inputs={'text': 'x'})],
            jobs=[job],
            evaluators=[],
            print_results=False,
            _send_results=False,
            _exit_on_failure=False,
        )

    assert len(requests) == 1
    assert requests[0]['metadata'] == {'evaluatorq_pipeline': 'red_teaming', 'evaluatorq_run_id': 'outer-run'}


@pytest.mark.asyncio
async def test_nested_evaluatorq_run_id_survives_parallel_datapoints() -> None:
    """Sanity check with multiple concurrent datapoints (parallelism > 1):
    every job's task must independently see the outer-bound run id.
    """
    requests: list[dict[str, Any]] = []
    client = cast(AsyncChatCompletionsClient, cast(object, _Client(requests)))

    async def job(_data: DataPoint, _row: int):
        await generate_executive_summary('facts', llm_client=client, model='gpt-4o')
        return {'name': 'noop', 'output': 'ok'}

    with evaluatorq_pipeline('agent_simulation'), evaluatorq_run_id('outer-run-2'):
        await evaluatorq(
            'nested-run-metadata-parallel',
            data=[DataPoint(inputs={'text': f'x{i}'}) for i in range(5)],
            jobs=[job],
            evaluators=[],
            parallelism=5,
            print_results=False,
            _send_results=False,
            _exit_on_failure=False,
        )

    assert len(requests) == 5
    for request in requests:
        assert request['metadata'] == {'evaluatorq_pipeline': 'agent_simulation', 'evaluatorq_run_id': 'outer-run-2'}


def test_nesting_semantics_inner_binding_shadows_and_restores_outer() -> None:
    """Nested ``evaluatorq_run_id`` scopes shadow the outer value while active
    and restore it exactly on exit — this is the primitive the nested-call
    propagation above relies on.
    """
    with evaluatorq_pipeline('red_teaming'), evaluatorq_run_id('outer'):
        assert pipeline_metadata()['evaluatorq_run_id'] == 'outer'
        with evaluatorq_run_id('inner'):
            assert pipeline_metadata()['evaluatorq_run_id'] == 'inner'
        assert pipeline_metadata()['evaluatorq_run_id'] == 'outer'
