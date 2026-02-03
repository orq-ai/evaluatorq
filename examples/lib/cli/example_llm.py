"""
CLI Example: LLM-based evaluation.

This example demonstrates using Claude for job execution
and LLM-based evaluators from the command line.
"""

import asyncio

from anthropic import AsyncAnthropic

from evaluatorq import DataPoint, evaluatorq, job

from ..evals import contains_name_validator, is_it_polite_llm_eval

# Initialize Anthropic client
claude = AsyncAnthropic()


@job("greet")
async def greet(data: DataPoint, _row: int = 0) -> str:
    """Generate a greeting response using Claude (lazy and sarcastic)."""
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


async def main():
    """Run the LLM evaluation example from CLI."""
    _ = await evaluatorq(
        "dataset-evaluation",
        {
            "data": [
                DataPoint(inputs={"name": "Alice"}),
                DataPoint(inputs={"name": "Bob"}),
                DataPoint(inputs={"name": "Márk"}),
            ],
            "jobs": [greet],
            "evaluators": [contains_name_validator, is_it_polite_llm_eval],
            "parallelism": 2,
            "print": True,
        },
    )


if __name__ == "__main__":
    asyncio.run(main())
