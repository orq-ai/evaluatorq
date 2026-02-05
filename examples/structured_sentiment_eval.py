"""
Structured evaluation result example - sentiment breakdown scorer.

Demonstrates returning structured EvaluationResultCell values
with sentiment distribution across categories.

Usage:
    python examples/structured_sentiment_eval.py
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


async def sentiment_scorer(params: ScorerParameter) -> EvaluationResult:
    """Sentiment distribution scorer."""
    text = str(params["output"]).lower()
    positive_words = ["good", "great", "excellent", "happy", "love"]
    negative_words = ["bad", "terrible", "awful", "sad", "hate"]
    pos_count = sum(1 for w in positive_words if w in text)
    neg_count = sum(1 for w in negative_words if w in text)
    total = max(pos_count + neg_count, 1)

    return EvaluationResult(
        value=EvaluationResultCell(
            type="sentiment",
            value={
                "positive": pos_count / total,
                "negative": neg_count / total,
                "neutral": 1 - (pos_count + neg_count) / total,
            },
        ),
        explanation="Sentiment distribution across categories",
    )


sentiment_evaluator: Evaluator = {
    "name": "sentiment",
    "scorer": sentiment_scorer,
}


async def run():
    """Run the structured sentiment evaluation."""
    results = await evaluatorq(
        "structured-sentiment",
        data=[
            DataPoint(inputs={"text": "This is a great and excellent product!"}),
            DataPoint(inputs={"text": "Terrible experience, very bad service."}),
            DataPoint(inputs={"text": "The package arrived on Tuesday."}),
            DataPoint(inputs={"text": "I love this but hate the price."}),
        ],
        jobs=[echo_job],
        evaluators=[sentiment_evaluator],
        print_results=True,
    )

    return results


if __name__ == "__main__":
    _ = asyncio.run(run())
