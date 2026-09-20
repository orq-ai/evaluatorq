"""Completion callbacks receive every terminal datapoint result exactly once."""
# ruff: noqa: S101

from __future__ import annotations

import asyncio
import importlib
from typing import TYPE_CHECKING

import pytest

from evaluatorq import DataPoint, DataPointResult, DatasetIdInput, EvaluationResult, evaluatorq
from evaluatorq.fetch_data import DataPointBatch

if TYPE_CHECKING:
    from evaluatorq.types import ScorerParameter


def recorded(identifier: str) -> DataPoint:
    return DataPoint(
        inputs={
            'id': identifier,
            'messages': [{'role': 'assistant', 'content': f'recorded response for {identifier}'}],
        }
    )


async def always_pass(_params: ScorerParameter) -> EvaluationResult:
    await asyncio.sleep(0)
    return EvaluationResult.model_validate({'value': 1, 'pass': True})


@pytest.mark.asyncio
async def test_calls_sync_callback_once_per_terminal_result() -> None:
    seen: list[str] = []

    def complete(result: DataPointResult) -> None:
        seen.append(str(result.data_point.inputs['id']))

    results = await evaluatorq(
        'callback-test',
        data=[recorded('a'), recorded('b')],
        evaluators=[{'name': 'always-pass', 'scorer': always_pass}],
        inference=False,
        print_results=False,
        on_datapoint_complete=complete,
        _send_results=False,
    )

    assert len(results) == 2
    assert sorted(seen) == ['a', 'b']


@pytest.mark.asyncio
async def test_calls_async_callback_once_per_terminal_result() -> None:
    seen: list[str] = []

    async def complete(result: DataPointResult) -> None:
        await asyncio.sleep(0)
        seen.append(str(result.data_point.inputs['id']))

    results = await evaluatorq(
        'callback-test',
        data=[recorded('a'), recorded('b')],
        evaluators=[{'name': 'always-pass', 'scorer': always_pass}],
        inference=False,
        print_results=False,
        on_datapoint_complete=complete,
        _send_results=False,
    )

    assert len(results) == 2
    assert sorted(seen) == ['a', 'b']


@pytest.mark.asyncio
async def test_callback_failure_aborts_the_run() -> None:
    async def fail(_result: DataPointResult) -> None:
        await asyncio.sleep(0)
        raise RuntimeError('ui state unavailable')

    with pytest.raises(RuntimeError, match='ui state unavailable'):
        await evaluatorq(
            'callback-test',
            data=[recorded('a')],
            evaluators=[{'name': 'always-pass', 'scorer': always_pass}],
            inference=False,
            print_results=False,
            on_datapoint_complete=fail,
            _send_results=False,
        )


@pytest.mark.asyncio
async def test_callback_failure_cancels_and_drains_sibling_before_raising() -> None:
    sibling_started = asyncio.Event()
    sibling_release = asyncio.Event()
    sibling_cancelled = asyncio.Event()
    sibling_drained = asyncio.Event()
    cleanup_ready = asyncio.Event()
    sibling_tasks: list[asyncio.Task[object]] = []
    seen: list[str] = []
    failure = RuntimeError('ui state unavailable')

    async def score(params: ScorerParameter) -> EvaluationResult:
        if params['data'].inputs['id'] == 'b':
            task = asyncio.current_task()
            assert task is not None
            sibling_tasks.append(task)
            sibling_started.set()
            try:
                await sibling_release.wait()
            except asyncio.CancelledError:
                sibling_cancelled.set()
                asyncio.get_running_loop().call_soon(cleanup_ready.set)
                await cleanup_ready.wait()
                sibling_drained.set()
                raise
        return await always_pass(params)

    async def complete(result: DataPointResult) -> None:
        identifier = str(result.data_point.inputs['id'])
        seen.append(identifier)
        if identifier == 'a':
            await sibling_started.wait()
            raise failure

    try:
        with pytest.raises(RuntimeError) as caught:
            await evaluatorq(
                'callback-test',
                data=[recorded('a'), recorded('b')],
                evaluators=[{'name': 'blocked-sibling', 'scorer': score}],
                inference=False,
                datapoint_parallelism=2,
                print_results=False,
                on_datapoint_complete=complete,
                _send_results=False,
            )

        assert caught.value is failure
        assert sibling_cancelled.is_set()
        assert sibling_drained.is_set()
        assert all(task.done() for task in sibling_tasks)
        assert seen == ['a']
    finally:
        sibling_release.set()
        await asyncio.gather(*sibling_tasks, return_exceptions=True)


@pytest.mark.asyncio
async def test_streaming_calls_callback_once_per_terminal_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluatorq_module = importlib.import_module('evaluatorq.evaluatorq')
    seen: list[str] = []

    async def fetch_rows(*_args: object, **_kwargs: object):
        yield DataPointBatch(
            datapoints=[recorded('a'), recorded('b')],
            has_more=False,
            batch_number=1,
        )
        await asyncio.sleep(0)

    async def complete(result: DataPointResult) -> None:
        await asyncio.sleep(0)
        seen.append(str(result.data_point.inputs['id']))

    monkeypatch.setattr(evaluatorq_module, 'setup_orq_client', lambda _api_key: object())
    monkeypatch.setattr(evaluatorq_module, 'fetch_dataset_batches', fetch_rows)
    monkeypatch.setenv('ORQ_API_KEY', 'test-key')

    results = await evaluatorq_module.evaluatorq(
        'callback-test',
        data=DatasetIdInput(dataset_id='dataset'),
        evaluators=[{'name': 'always-pass', 'scorer': always_pass}],
        inference=False,
        print_results=False,
        on_datapoint_complete=complete,
        _send_results=False,
    )

    assert len(results) == 2
    assert sorted(seen) == ['a', 'b']
