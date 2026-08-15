"""Test that evaluatorq runs correctly with various scenarios."""

import asyncio
import importlib
import inspect
import random

import pytest

from evaluatorq import evaluatorq
from evaluatorq.evaluatorq import check_pass_failures
from evaluatorq.fetch_data import DataPointBatch
from evaluatorq.types import (
    DataPoint,
    DatasetIdInput,
    DataPointResult,
    EvaluationResult,
    EvaluatorParams,
    EvaluatorScore,
    JobResult,
    ScorerParameter,
)

# Sample text data
SAMPLE_TEXTS = [
    "The quick brown fox jumps over the lazy dog",
    "Python is a powerful programming language",
    "Machine learning models require large datasets",
]


async def text_analyzer(data: DataPoint, _row: int):
    """Simple text analysis job."""
    text = str(data.inputs["text"])
    await asyncio.sleep(0.001)

    words = text.split()
    return {
        "name": "text-analyzer",
        "output": {
            "length": len(text),
            "word_count": len(words),
        },
    }


async def length_check_scorer(params: ScorerParameter) -> EvaluationResult:
    """Evaluate if output length is sufficient."""
    output = params["output"]

    if not isinstance(output, dict) or "length" not in output:
        return EvaluationResult(value="N/A", explanation="Not applicable")

    passes = bool(output["length"] > 20)
    return EvaluationResult(
        value=1 if passes else 0,
        explanation="Text length is sufficient" if passes else "Text too short",
    )


def generate_test_data(count: int):
    """Generate test data points."""
    return [
        DataPoint(inputs={"text": random.choice(SAMPLE_TEXTS)}) for _ in range(count)
    ]


@pytest.mark.asyncio
async def test_evaluatorq_basic():
    """Test that evaluatorq runs correctly with basic setup."""
    data_points = generate_test_data(10)

    results = await evaluatorq(
        "test-basic",
        data=data_points,
        jobs=[text_analyzer],
        evaluators=[
            {
                "name": "length-check",
                "scorer": length_check_scorer,
            },
        ],
        parallelism=5,
        print_results=False,
    )

    # Verify evaluatorq returns results
    assert results is not None
    assert len(results) == 10

    # Verify each result has expected structure (DataPointResult objects)
    for result in results:
        assert hasattr(result, "data_point")
        assert hasattr(result, "job_results")
        assert result.job_results is not None
        assert len(result.job_results) > 0


@pytest.mark.asyncio
async def test_evaluatorq_with_parallelism():
    """Test that evaluatorq handles parallelism correctly."""
    data_points = generate_test_data(100)

    results = await evaluatorq(
        "test-parallelism",
        data=data_points,
        jobs=[text_analyzer],
        evaluators=[
            {
                "name": "length-check",
                "scorer": length_check_scorer,
            },
        ],
        parallelism=10,
        print_results=False,
    )

    assert results is not None
    assert len(results) == 100


@pytest.mark.asyncio
async def test_evaluatorq_parallelism_limit():
    """Test that concurrent data points never exceed the parallelism value."""
    parallelism = 3
    data_points = generate_test_data(20)

    concurrent_count = 0
    max_concurrent = 0
    lock = asyncio.Lock()

    async def tracking_job(data: DataPoint, _row: int):
        nonlocal concurrent_count, max_concurrent
        async with lock:
            concurrent_count += 1
            if concurrent_count > max_concurrent:
                max_concurrent = concurrent_count

        await asyncio.sleep(0.05)

        async with lock:
            concurrent_count -= 1

        text = str(data.inputs["text"])
        return {"name": "tracking-job", "output": {"length": len(text)}}

    results = await evaluatorq(
        "test-concurrency-limit",
        data=data_points,
        jobs=[tracking_job],
        evaluators=[],
        parallelism=parallelism,
        print_results=False,
    )

    assert results is not None
    assert len(results) == 20
    assert max_concurrent <= parallelism, (
        f"Max concurrent data points ({max_concurrent}) exceeded parallelism ({parallelism})"
    )


@pytest.mark.asyncio
async def test_evaluatorq_defaults_to_concurrent_execution():
    """Omitting `parallelism` must run datapoints concurrently, not serially.

    Every other parallelism test passes the value explicitly, so reverting the
    default from 10 back to 1 used to leave the whole suite green while silently
    serializing every caller who takes the default — which is most of them.
    Asserted behaviourally rather than against the literal: a datapoint that
    starts before an earlier one finishes is the property that matters, and it
    holds for any default above 1.
    """
    data_points = generate_test_data(10)

    concurrent_count = 0
    max_concurrent = 0
    lock = asyncio.Lock()

    async def tracking_job(data: DataPoint, _row: int):
        nonlocal concurrent_count, max_concurrent
        async with lock:
            concurrent_count += 1
            max_concurrent = max(max_concurrent, concurrent_count)
        await asyncio.sleep(0.05)
        async with lock:
            concurrent_count -= 1
        return {"name": "tracking-job", "output": {"ok": True}}

    await evaluatorq(
        "test-default-parallelism",
        data=data_points,
        jobs=[tracking_job],
        evaluators=[],
        print_results=False,
    )

    assert max_concurrent > 1, (
        f"bare evaluatorq() ran datapoints serially (max concurrent {max_concurrent})"
    )
    # The two surfaces must agree — `evaluatorq()`'s signature default and the
    # `EvaluatorParams` field are separate declarations of the same contract.
    assert EvaluatorParams.model_fields["parallelism"].default == 10
    assert inspect.signature(evaluatorq).parameters["parallelism"].default == 10


@pytest.mark.asyncio
async def test_evaluatorq_stress():
    """Stress test with larger dataset."""
    data_points = generate_test_data(300)

    results = await evaluatorq(
        "test-stress",
        data=data_points,
        jobs=[text_analyzer],
        evaluators=[
            {
                "name": "length-check",
                "scorer": length_check_scorer,
            },
        ],
        parallelism=10,
        print_results=False,
    )

    assert results is not None
    assert len(results) == 300


@pytest.mark.asyncio
async def test_streaming_fetch_failure_cancels_in_flight_datapoints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fetch failure must not leave processing tasks running or unobserved."""
    evaluatorq_module = importlib.import_module('evaluatorq.evaluatorq')

    created_tasks: list[asyncio.Task[object]] = []
    unhandled: list[dict[str, object]] = []
    loop = asyncio.get_running_loop()
    previous_exception_handler = loop.get_exception_handler()

    def capture_unhandled(_loop: asyncio.AbstractEventLoop, context: dict[str, object]) -> None:
        unhandled.append(context)

    async def failing_fetch(*_args: object, **_kwargs: object):
        yield DataPointBatch(
            datapoints=[DataPoint(inputs={'row': index}) for index in range(3)],
            has_more=True,
            batch_number=1,
        )
        await asyncio.sleep(0)
        raise RuntimeError('fetch failed after first batch')

    async def slow_job(_data: DataPoint, _row: int):
        await asyncio.Event().wait()
        return {'name': 'slow', 'output': 'unreachable'}

    async def track_processing(*_args: object, **_kwargs: object) -> list[object]:
        task = asyncio.current_task()
        assert task is not None
        created_tasks.append(task)
        await asyncio.Event().wait()
        return []

    monkeypatch.setattr(evaluatorq_module, 'setup_orq_client', lambda _api_key: object())
    monkeypatch.setattr(evaluatorq_module, 'fetch_dataset_batches', failing_fetch)
    monkeypatch.setattr(evaluatorq_module, 'process_data_point', track_processing)
    monkeypatch.setenv('ORQ_API_KEY', 'test-key')
    loop.set_exception_handler(capture_unhandled)

    try:
        with pytest.raises(RuntimeError, match='fetch failed after first batch'):
            await evaluatorq_module.evaluatorq(
                'streaming-fetch-failure',
                data=DatasetIdInput(dataset_id='dataset'),
                jobs=[slow_job],
                print_results=False,
                _send_results=False,
            )

        assert len(created_tasks) == 3
        assert all(task.done() and task.cancelled() for task in created_tasks)
        assert unhandled == []
    finally:
        loop.set_exception_handler(previous_exception_handler)
        for task in created_tasks:
            task.cancel()
        await asyncio.gather(*created_tasks, return_exceptions=True)


@pytest.mark.asyncio
async def test_streaming_polling_and_processing_failures_are_both_surfaced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A polling failure must not hide a processing failure from the caller."""
    evaluatorq_module = importlib.import_module('evaluatorq.evaluatorq')
    progress_updates = 0

    class PollingFailureProgress:
        async def update_progress(self, **_kwargs: object) -> None:
            nonlocal progress_updates
            progress_updates += 1
            if progress_updates == 2:
                raise RuntimeError('polling failed')

    async def fetch_one_batch(*_args: object, **_kwargs: object):
        yield DataPointBatch(
            datapoints=[DataPoint(inputs={'row': 0})],
            has_more=False,
            batch_number=1,
        )
        await asyncio.sleep(0)

    async def failing_processing(*_args: object, **_kwargs: object) -> list[object]:
        raise RuntimeError('processing failed')

    async def job(_data: DataPoint, _row: int):
        return {'name': 'job', 'output': 'output'}

    monkeypatch.setattr(evaluatorq_module, 'ProgressService', PollingFailureProgress)
    monkeypatch.setattr(evaluatorq_module, 'setup_orq_client', lambda _api_key: object())
    monkeypatch.setattr(evaluatorq_module, 'fetch_dataset_batches', fetch_one_batch)
    monkeypatch.setattr(evaluatorq_module, 'process_data_point', failing_processing)
    monkeypatch.setenv('ORQ_API_KEY', 'test-key')

    with pytest.raises(RuntimeError) as exc_info:
        await evaluatorq_module.evaluatorq(
            'streaming-multiple-failures',
            data=DatasetIdInput(dataset_id='dataset'),
            jobs=[job],
            print_results=False,
            _send_results=False,
        )

    assert 'polling failed' in str(exc_info.value)
    assert 'processing failed' in str(exc_info.value)


@pytest.mark.asyncio
async def test_evaluatorq_returns_failed_results_to_library_callers() -> None:
    """Library callers receive failed results instead of a process exit."""

    async def job(_data: DataPoint, _row: int):
        return {'name': 'job', 'output': 'output'}

    async def failing_scorer(_params: ScorerParameter) -> EvaluationResult:
        return EvaluationResult.model_validate({'value': 0, 'pass': False})

    results = await evaluatorq(
        'library-failure',
        data=[DataPoint(inputs={'text': 'value'})],
        jobs=[job],
        evaluators=[{'name': 'failing', 'scorer': failing_scorer}],
        print_results=False,
        _send_results=False,
    )

    assert len(results) == 1
    job_results = results[0].job_results
    assert job_results is not None
    evaluator_scores = job_results[0].evaluator_scores
    assert evaluator_scores is not None
    assert evaluator_scores[0].score.pass_ is False


@pytest.mark.asyncio
async def test_evaluator_parallelism_is_bounded_per_job() -> None:
    """Evaluator fan-out must respect the per-datapoint parallelism cap."""
    concurrent_count = 0
    max_concurrent = 0
    invocations = 0
    lock = asyncio.Lock()

    async def evaluator_scorer(_params: ScorerParameter) -> EvaluationResult:
        nonlocal concurrent_count, invocations, max_concurrent
        async with lock:
            invocations += 1
            concurrent_count += 1
            max_concurrent = max(max_concurrent, concurrent_count)
        await asyncio.sleep(0.02)
        async with lock:
            concurrent_count -= 1
        return EvaluationResult.model_validate({'value': 1, 'pass': True})

    async def job(_data: DataPoint, _row: int):
        return {'name': 'job', 'output': 'output'}

    results = await evaluatorq(
        'evaluator-parallelism',
        data=[DataPoint(inputs={'text': 'value'})],
        jobs=[job],
        evaluators=[{'name': f'evaluator-{index}', 'scorer': evaluator_scorer} for index in range(8)],
        parallelism=2,
        print_results=False,
        _send_results=False,
    )

    assert len(results) == 1
    assert invocations == 8
    assert max_concurrent <= 2


def test_check_pass_failures_counts_evaluator_errors():
    """An evaluator whose judge call raised leaves ``pass_`` unset — the run must not exit 0."""
    results = [
        DataPointResult(
            data_point=DataPoint(inputs={"text": "x"}),
            job_results=[
                JobResult(
                    job_name="job1",
                    output="output",
                    evaluator_scores=[
                        EvaluatorScore(
                            evaluator_name="judge",
                            score=EvaluationResult(value=""),
                            error="judge call failed",
                        ),
                    ],
                ),
            ],
        ),
    ]

    assert check_pass_failures(results) is False
    assert check_pass_failures(results, treat_errors_as_failure=True) is True
