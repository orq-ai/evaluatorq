"""RES-993: no-inference mode replays pre-recorded responses to the evaluators."""

import pytest

from evaluatorq import evaluatorq
from evaluatorq.evaluatorq import extract_recorded_response
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
        _exit_on_failure=False,
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
        _exit_on_failure=False,
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
        _exit_on_failure=False,
    )

    assert len(results) == 1
    assert results[0].job_results is not None
    job_result = results[0].job_results[0]
    assert job_result.output is None
    assert job_result.error is not None
    assert "messages" in job_result.error


@pytest.mark.asyncio
async def test_inference_true_without_jobs_raises():
    with pytest.raises(ValueError, match="jobs"):
        await evaluatorq(
            "test-needs-jobs",
            data=[DataPoint(inputs={"text": "x"})],
            evaluators=[],
            print_results=False,
            _send_results=False,
            _exit_on_failure=False,
        )
