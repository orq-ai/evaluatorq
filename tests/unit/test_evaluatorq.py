"""Test that evaluatorq runs correctly with various scenarios."""

import asyncio
import random

import pytest

from evaluatorq import evaluatorq
from evaluatorq.types import DataPoint, EvaluationResult, ScorerParameter

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
