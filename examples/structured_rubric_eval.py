"""
Structured evaluation result example - multi-criteria rubric scorer.

Demonstrates returning structured EvaluationResultCell values with
multiple sub-scores per evaluator.

Usage:
    python examples/structured_rubric_eval.py
"""

import asyncio

from evaluatorq import (
    DataPoint,
    EvaluationResult,
    EvaluationResultCell,
    Evaluator,
    ScorerParameter,
    evaluatorq,
    job,
)


@job("echo")
async def echo_job(data: DataPoint, _row: int) -> str:
    """Echo the input text."""
    return str(data.inputs.get("text", ""))


async def rubric_scorer(params: ScorerParameter) -> EvaluationResult:
    """Multi-criteria quality rubric scorer."""
    text = str(params["output"])
    return EvaluationResult(
        value=EvaluationResultCell(
            type="rubric",
            value={
                "relevance": min(len(text) / 100, 1),
                "coherence": 0.9 if "." in text else 0.4,
                "fluency": 0.85 if len(text.split(" ")) > 5 else 0.5,
            },
        ),
        explanation="Multi-criteria quality rubric",
    )


rubric_evaluator: Evaluator = {
    "name": "rubric",
    "scorer": rubric_scorer,
}


async def run():
    """Run the structured rubric evaluation."""
    results = await evaluatorq(
        "structured-rubric",
        data=[
            DataPoint(inputs={"text": "The quick brown fox jumps over the lazy dog."}),
            DataPoint(inputs={"text": "Hi"}),
            DataPoint(
                inputs={
                    "text": "This is a well-structured sentence that demonstrates good fluency and coherence in natural language."
                }
            ),
        ],
        jobs=[echo_job],
        evaluators=[rubric_evaluator],
        print_results=True,
    )

    return results


if __name__ == "__main__":
    _ = asyncio.run(run())
