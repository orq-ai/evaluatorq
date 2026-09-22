"""RES-993: no-inference mode replays pre-recorded responses to the evaluators."""

import importlib
from unittest.mock import AsyncMock

import pytest

from evaluatorq import Trace, TraceInput, evaluatorq
from evaluatorq.contracts import Message
from evaluatorq.evaluatorq import check_pass_failures, extract_recorded_response
from evaluatorq.types import DataPoint, EvaluationResult, ScorerParameter


def _row(response: str | None, *, with_assistant: bool = True) -> DataPoint:
    messages: list[dict[str, object]] = [{"role": "user", "content": "hi"}]
    if with_assistant:
        messages.append({"role": "assistant", "content": response})
    return DataPoint(inputs={"messages": messages})


async def _capturing_scorer(seen: list[object]):
    async def scorer(params: ScorerParameter) -> EvaluationResult:
        seen.append(params["output"])
        return EvaluationResult(value=1)

    return scorer


# --- extract_recorded_response -------------------------------------------------


def test_extract_returns_last_assistant_message():
    messages = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "first"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "second"},
    ]
    assert extract_recorded_response(messages) == "second"


def test_extract_flattens_multipart_content():
    messages = [
        {"role": "assistant", "content": [{"type": "text", "text": "hello there"}]},
    ]
    assert extract_recorded_response(messages) == "hello there"


def test_extract_raises_on_empty_messages():
    with pytest.raises(ValueError, match="no messages"):
        extract_recorded_response([])


def test_extract_raises_when_no_assistant_message():
    with pytest.raises(ValueError, match="no assistant message"):
        extract_recorded_response([{"role": "user", "content": "only user"}])


def test_extract_raises_when_assistant_content_blank():
    with pytest.raises(ValueError, match="no assistant message"):
        extract_recorded_response([{"role": "assistant", "content": "   "}])


# --- evaluatorq(inference=False) ----------------------------------------------


@pytest.mark.asyncio
async def test_no_inference_feeds_recorded_response_to_evaluators():
    seen: list[object] = []
    results = await evaluatorq(
        "test-no-inference",
        data=[_row("the recorded answer")],
        evaluators=[{"name": "capture", "scorer": await _capturing_scorer(seen)}],
        inference=False,
        print_results=False,
        _send_results=False,
    )

    assert seen == ["the recorded answer"]
    assert len(results) == 1
    assert results[0].job_results is not None
    assert results[0].job_results[0].output == "the recorded answer"
    assert results[0].job_results[0].error is None


@pytest.mark.asyncio
async def test_no_inference_does_not_run_provided_jobs():
    ran = False

    async def should_not_run(_data: DataPoint, _row: int):
        nonlocal ran
        ran = True
        return {"name": "nope", "output": "generated"}

    results = await evaluatorq(
        "test-jobs-ignored",
        data=[_row("recorded")],
        jobs=[should_not_run],
        inference=False,
        print_results=False,
        _send_results=False,
    )

    assert ran is False
    assert results[0].job_results is not None
    assert results[0].job_results[0].output == "recorded"


@pytest.mark.asyncio
async def test_no_inference_missing_response_errors_clearly():
    results = await evaluatorq(
        "test-missing-response",
        data=[_row(None, with_assistant=False)],
        evaluators=[],
        inference=False,
        print_results=False,
        _send_results=False,
    )

    assert len(results) == 1
    assert results[0].job_results is not None
    job_result = results[0].job_results[0]
    assert job_result.output is None
    assert job_result.error is not None
    assert "messages" in job_result.error

    # Fail-loud at the run level: a missing recorded response leaves no evaluator
    # score, so only treat_errors_as_failure surfaces it (the run must not exit 0).
    assert check_pass_failures(results) is False
    assert check_pass_failures(results, treat_errors_as_failure=True) is True


@pytest.mark.asyncio
async def test_inference_true_without_jobs_raises():
    with pytest.raises(ValueError, match="jobs"):
        await evaluatorq(
            "test-needs-jobs",
            data=[DataPoint(inputs={"text": "x"})],
            evaluators=[],
            print_results=False,
            _send_results=False,
        )


TRACE = Trace(
    trace_id='trace-1',
    input_messages=[Message(role='user', content='question')],
    output_messages=[Message(role='assistant', content='recorded answer')],
)

INPUT_ONLY_TRACE = Trace(
    trace_id='trace-input-only',
    input_messages=[Message(role='user', content='question')],
)

TRACE_WITH_EMPTY_SELECTED_OUTPUT = Trace(
    trace_id='trace-empty-selected-output',
    input_messages=[
        Message(role='user', content='earlier question'),
        Message(role='assistant', content='earlier answer'),
        Message(role='user', content='current question'),
    ],
)


@pytest.mark.asyncio
async def test_trace_input_fetches_once_and_scores_recorded_output(monkeypatch: pytest.MonkeyPatch) -> None:
    evaluatorq_module = importlib.import_module('evaluatorq.evaluatorq')
    fetch = AsyncMock(return_value=[TRACE])
    captured: list[object] = []

    async def capture(params: ScorerParameter) -> EvaluationResult:
        captured.append(params['output'])
        return EvaluationResult(value=1)

    monkeypatch.setattr(evaluatorq_module, 'fetch_traces', fetch)

    results = await evaluatorq(
        'trace-eval',
        data=TraceInput(trace_id='trace-1'),
        inference=False,
        evaluators=[{'name': 'capture', 'scorer': capture}],
        print_results=False,
        _send_results=False,
    )

    fetch.assert_awaited_once()
    assert captured == [TRACE.to_datapoint().inputs['recorded_output']]
    assert results[0].job_results is not None
    assert results[0].job_results[0].output == TRACE.output_messages


@pytest.mark.asyncio
async def test_trace_input_rejects_empty_recorded_output(monkeypatch: pytest.MonkeyPatch) -> None:
    evaluatorq_module = importlib.import_module('evaluatorq.evaluatorq')
    fetch = AsyncMock(return_value=[INPUT_ONLY_TRACE])
    monkeypatch.setattr(evaluatorq_module, 'fetch_traces', fetch)

    results = await evaluatorq(
        'trace-eval',
        data=TraceInput(trace_id='trace-input-only'),
        inference=False,
        evaluators=[],
        print_results=False,
        _send_results=False,
    )

    fetch.assert_awaited_once()
    assert results[0].job_results is not None
    job_result = results[0].job_results[0]
    assert job_result.output is None
    assert job_result.error is not None
    assert 'no messages' in job_result.error


@pytest.mark.asyncio
async def test_trace_input_does_not_replay_earlier_assistant_when_output_is_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluatorq_module = importlib.import_module('evaluatorq.evaluatorq')
    fetch = AsyncMock(return_value=[TRACE_WITH_EMPTY_SELECTED_OUTPUT])
    monkeypatch.setattr(evaluatorq_module, 'fetch_traces', fetch)

    results = await evaluatorq(
        'trace-eval',
        data=TraceInput(trace_id='trace-empty-selected-output'),
        inference=False,
        evaluators=[],
        print_results=False,
        _send_results=False,
    )

    fetch.assert_awaited_once()
    assert results[0].job_results is not None
    job_result = results[0].job_results[0]
    assert job_result.output is None
    assert job_result.error is not None
    assert 'no messages' in job_result.error


@pytest.mark.asyncio
async def test_trace_input_requires_no_inference(monkeypatch: pytest.MonkeyPatch):
    evaluatorq_module = importlib.import_module('evaluatorq.evaluatorq')
    fetch = AsyncMock()
    monkeypatch.setattr(evaluatorq_module, 'fetch_traces', fetch)

    async def unused_job(_data: DataPoint, _row: int) -> dict[str, object]:
        return {"name": "unused", "output": None}

    with pytest.raises(ValueError, match="TraceInput"):
        await evaluatorq(
            "trace-eval",
            data=TraceInput(trace_id='trace-1'),
            jobs=[unused_job],
            print_results=False,
            _send_results=False,
        )

    fetch.assert_not_awaited()
