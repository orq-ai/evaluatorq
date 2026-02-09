"""
Example demonstrating LLM-based evaluation with multiple jobs.

This example shows how to:
- Create multiple LLM-powered jobs with different system prompts
- Use local and Orq platform evaluators to assess output quality
- Evaluate multiple data points in parallel
"""

import asyncio
import os
from typing import Any

from anthropic import AsyncAnthropic
from orq_ai_sdk import Orq
from orq_ai_sdk.models.invokeevalop import BERTScore, RougeN

from evaluatorq import DataPoint, Evaluator, ScorerParameter, evaluatorq, job

# Initialize clients
claude = AsyncAnthropic()

orq = Orq(
    api_key=os.environ.get("ORQ_API_KEY"),
    server_url=os.environ.get("ORQ_BASE_URL", "https://my.orq.ai"),
)

ROUGE_N_EVALUATOR_ID = "<your-rouge-n-evaluator-id>"

BERT_SCORE_EVALUATOR_ID = "<your-bert-score-evaluator-id>"


# Job 1: Polite greeter (lazy and sarcastic for testing)
@job("greet")
async def greet(data: DataPoint, _row: int = 0) -> str:
    """Generate a greeting response using Claude."""
    name = data.inputs.get("name", "")

    response = await claude.messages.create(
        stream=False,
        max_tokens=100,
        model="claude-3-5-haiku-latest",
        system="For testing purposes please be really lazy and sarcastic in your response, not polite at all.",
        messages=[
            {
                "role": "user",
                "content": f"Hello My name is {name}",
            }
        ],
    )

    return response.content[0].text if response.content[0].type == "text" else ""


# Job 2: Joker personality
@job("joker")
async def joker(data: DataPoint, _row: int = 0) -> str:
    """Generate a funny/sarcastic response using Claude."""
    name = data.inputs.get("name", "")

    response = await claude.messages.create(
        stream=False,
        max_tokens=100,
        model="claude-3-5-haiku-latest",
        system="You are a joker. You are funny and sarcastic. You are also a bit of a smartass. and make fun of the name of the user",
        messages=[
            {
                "role": "user",
                "content": f"Hello My name is {name}",
            }
        ],
    )

    return response.content[0].text if response.content[0].type == "text" else ""


# Job 3: Mathematician personality
@job("calculator")
async def calculator(data: DataPoint, _row: int = 0) -> str:
    """Generate a mathematical response using Claude."""
    name = data.inputs.get("name", "")

    response = await claude.messages.create(
        stream=False,
        max_tokens=100,
        model="claude-3-5-haiku-latest",
        system="You are a mathematician. You bring up a relating theory or a recent discovery when somebody talks to you.",
        messages=[
            {
                "role": "user",
                "content": f"Hello My name is {name}",
            }
        ],
    )

    return response.content[0].text if response.content[0].type == "text" else ""


# Evaluator 1: Length similarity (local)
length_similarity_evaluator: Evaluator = {
    "name": "length-similarity",
    "scorer": lambda params: _length_similarity_scorer(params),
}


async def _length_similarity_scorer(params: ScorerParameter) -> dict[str, Any]:
    data: DataPoint = params["data"]
    output = params["output"]
    expected = str(data.expected_output or "")
    actual = str(output)
    max_len = max(len(expected), len(actual), 1)
    score = 1 - abs(len(expected) - len(actual)) / max_len
    return {
        "value": round(score * 100) / 100,
        "explanation": f"Length similarity: expected {len(expected)} chars, got {len(actual)} chars",
    }


# Evaluator 2: ROUGE-N (via Orq platform)
rouge_n_evaluator: Evaluator = {
    "name": "rouge_n",
    "scorer": lambda params: _rouge_n_scorer(params),
}


async def _rouge_n_scorer(params: ScorerParameter) -> dict[str, Any]:
    data: DataPoint = params["data"]
    output = params["output"]
    result = await orq.evals.invoke_async(
        id=ROUGE_N_EVALUATOR_ID,
        output=str(output),
        reference=str(data.expected_output or ""),
    )
    if isinstance(result, RougeN):
        val = result.value
        return {
            "value": {
                "type": "rouge_n",
                "value": {
                    "rouge_1": {"precision": val.rouge_1.precision, "recall": val.rouge_1.recall, "f1": val.rouge_1.f1},
                    "rouge_2": {"precision": val.rouge_2.precision, "recall": val.rouge_2.recall, "f1": val.rouge_2.f1},
                    "rouge_l": {"precision": val.rouge_l.precision, "recall": val.rouge_l.recall, "f1": val.rouge_l.f1},
                },
            },
            "explanation": "ROUGE-N similarity scores between output and reference",
        }
    return {"value": 0, "explanation": "Unexpected response format"}


# Evaluator 3: BERTScore (via Orq platform)
bert_score_evaluator: Evaluator = {
    "name": "bert-score",
    "scorer": lambda params: _bert_score_scorer(params),
}


async def _bert_score_scorer(params: ScorerParameter) -> dict[str, Any]:
    data: DataPoint = params["data"]
    output = params["output"]
    result = await orq.evals.invoke_async(
        id=BERT_SCORE_EVALUATOR_ID,
        output=str(output),
        reference=str(data.expected_output or ""),
    )
    if isinstance(result, BERTScore):
        val = result.value
        return {
            "value": {
                "type": "bert_score",
                "value": {
                    "precision": val.precision,
                    "recall": val.recall,
                    "f1": val.f1,
                },
            },
            "explanation": "BERTScore semantic similarity between output and reference",
        }
    return {"value": 0, "explanation": "Unexpected response format"}


async def main():
    """Run evaluation with multiple LLM jobs and evaluators."""
    _ = await evaluatorq(
        "llm-eval-with-results",
        {
            "data": [
                DataPoint(
                    inputs={"name": "Alice"},
                    expected_output="Hello Alice, nice to meet you!",
                ),
                DataPoint(
                    inputs={"name": "Bob"},
                    expected_output="Hello Bob, nice to meet you!",
                ),
                DataPoint(
                    inputs={"name": "Márk"},
                    expected_output="Hello Márk, nice to meet you!",
                ),
            ],
            "jobs": [greet, joker, calculator],
            "evaluators": [
                length_similarity_evaluator,
                bert_score_evaluator,
                rouge_n_evaluator,
            ],
            "parallelism": 4,
            "print": True,
        },
    )


if __name__ == "__main__":
    asyncio.run(main())
