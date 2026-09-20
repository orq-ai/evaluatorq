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
    return EvaluationResult(value=1, pass_=True)


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
