"""
Simple local evaluation example with tracing.

This example runs a simple evaluation without datasets or deployments,
just local data and a local evaluator, with OTEL tracing enabled.
"""

import asyncio

from evaluatorq import DataPoint, evaluatorq, job, string_contains_evaluator


@job("uppercase-converter")
async def uppercase_job(data: DataPoint, _row: int) -> str:
    """Simple job that converts text to uppercase."""
    text = str(data.inputs.get("text", ""))
    return text.upper()


async def run():
    """Run a simple local evaluation."""
    print("\n🧪 Simple Local Evaluation with Tracing\n")
    print("------------------------------------------\n")

    # Simple local data
    data = [
        DataPoint(inputs={"text": "hello world"}, expected_output="HELLO"),
        DataPoint(inputs={"text": "python is great"}, expected_output="PYTHON"),
        DataPoint(inputs={"text": "evaluatorq rocks"}, expected_output="EVALUATORQ"),
    ]

    results = await evaluatorq(
        "simple-local-eval",
        data=data,
        jobs=[uppercase_job],
        evaluators=[string_contains_evaluator()],
        parallelism=3,
        print_results=True,
        description="Simple local evaluation to test tracing",
    )

    return results


if __name__ == "__main__":
    _ = asyncio.run(run())
