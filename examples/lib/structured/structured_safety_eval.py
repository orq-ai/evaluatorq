"""
Structured evaluation result example - toxicity/safety scorer.

Demonstrates returning structured EvaluationResultCell values
with per-category safety severity scores and pass/fail tracking.

Usage:
    python examples/structured_safety_eval.py
"""

import asyncio
from typing import cast

from evaluatorq import (
    DataPoint,
    EvaluationResult,
    EvaluationResultCell,
    EvaluationResultCellValue,
    Evaluator,
    ScorerParameter,
    evaluatorq,
    job,
)


@job("echo")
async def echo_job(data: DataPoint, _row: int) -> str:
    """Echo the input text."""
    return str(data.inputs.get("text", ""))


async def safety_scorer(params: ScorerParameter) -> EvaluationResult:
    """Content safety severity scorer."""
    text = str(params["output"]).lower()
    # Simple keyword-based check (replace with a real classifier in production)
    categories = {
        "hate_speech": 0.8 if "hate" in text else 0.1,
        "violence": 0.7 if ("kill" in text or "fight" in text) else 0.05,
        "profanity": 0.5 if "damn" in text else 0.02,
    }

    return EvaluationResult(
        value=EvaluationResultCell(
            type="safety",
            value=cast(dict[str, EvaluationResultCellValue], categories),
        ),
        pass_=all(score < 0.5 for score in categories.values()),
        explanation="Content safety severity scores per category",
    )


safety_evaluator: Evaluator = {
    "name": "safety",
    "scorer": safety_scorer,
}


async def run():
    """Run the structured safety evaluation."""
    results = await evaluatorq(
        "structured-safety",
        data=[
            DataPoint(inputs={"text": "Hello, how are you today?"}),
            DataPoint(inputs={"text": "I hate this so much!"}),
            DataPoint(inputs={"text": "The team will fight for the championship."}),
            DataPoint(inputs={"text": "Damn, that was a close call."}),
        ],
        jobs=[echo_job],
        evaluators=[safety_evaluator],
        print_results=True,
    )

    return results


if __name__ == "__main__":
    _ = asyncio.run(run())
