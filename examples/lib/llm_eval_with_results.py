"""
Example demonstrating LLM-based evaluation with multiple jobs.

This example shows how to:
- Create multiple LLM-powered jobs with different system prompts
- Use LLM-based evaluators to assess output quality
- Evaluate multiple data points in parallel
"""

import asyncio

from anthropic import AsyncAnthropic

from evaluatorq import DataPoint, evaluatorq, job
from .evals import (
    contains_name_validator,
    is_it_polite_llm_eval,
    min_length_validator,
)

# Initialize Anthropic client
claude = AsyncAnthropic()


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


async def main():
    """Run evaluation with multiple LLM jobs and evaluators."""
    _ = await evaluatorq(
        "llm-eval-with-results",
        {
            "data": [
                DataPoint(inputs={"name": "Alice"}),
                DataPoint(inputs={"name": "Bob"}),
                DataPoint(inputs={"name": "Márk"}),
            ],
            "jobs": [greet, joker, calculator],
            "evaluators": [
                contains_name_validator,
                is_it_polite_llm_eval,
                min_length_validator(70),
            ],
            "parallelism": 4,
            "print": True,
        },
    )


if __name__ == "__main__":
    asyncio.run(main())
