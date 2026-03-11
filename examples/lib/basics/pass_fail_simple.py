"""
Simple pass/fail example - all tests pass.

This is the simplest example showing basic evaluation with a calculator job.

Usage:
    python pass_fail_simple.py
"""

import asyncio
from typing import Any

from evaluatorq import DataPoint, ScorerParameter, evaluatorq, job


@job("calculator")
async def calculator_job(data: DataPoint, _row: int = 0) -> int | float:
    """Simple calculator that performs basic operations."""
    inputs = data.inputs
    a = inputs.get("a", 0)
    b = inputs.get("b", 0)
    op = inputs.get("op", "+")

    if op == "+":
        return a + b
    elif op == "-":
        return a - b
    elif op == "*":
        return a * b
    elif op == "/":
        return a / b
    else:
        return 0


async def matches_expected_scorer(input_data: ScorerParameter) -> dict[str, Any]:
    """Evaluator that checks if output matches expected."""
    output = input_data["output"]
    data = input_data["data"]
    matches = output == data.expected_output

    return {
        "value": 1.0 if matches else 0.0,
        "pass": matches,
        "explanation": "Correct!" if matches else f"Expected {data.expected_output}",
    }


async def main():
    """Run simple pass/fail evaluation."""
    print("\n🧮 Running calculator evaluation...\n")

    data_points = [
        DataPoint(inputs={"a": 2, "b": 3, "op": "+"}, expected_output=5),
        DataPoint(inputs={"a": 10, "b": 4, "op": "-"}, expected_output=6),
        DataPoint(inputs={"a": 7, "b": 8, "op": "*"}, expected_output=56),
        DataPoint(inputs={"a": 20, "b": 4, "op": "/"}, expected_output=5),
    ]

    _ = await evaluatorq(
        "calculator-test",
        {
            "data": data_points,
            "jobs": [calculator_job],
            "evaluators": [
                {"name": "matches-expected", "scorer": matches_expected_scorer}
            ],
            "print": True,
        },
    )

    print("\n✅ All tests passed!")


if __name__ == "__main__":
    asyncio.run(main())
