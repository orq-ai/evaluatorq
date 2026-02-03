"""
CLI Example 1: Simple text analysis evaluation.

This example demonstrates running evaluatorq from the command line
with a simple text analysis job and output validator.
"""

import asyncio
import re
from typing import Any

from evaluatorq import DataPoint, ScorerParameter, evaluatorq, job


@job("text-analyzer")
async def text_analyzer(data: DataPoint, _row: int = 0) -> dict[str, Any]:
    """Analyze text input and return statistics."""
    text = data.inputs.get("text") or data.inputs.get("input") or ""
    text_str = str(text)

    analysis = {
        "length": len(text_str),
        "wordCount": len([w for w in text_str.split() if w]),
        "hasNumbers": bool(re.search(r"\d", text_str)),
        "hasSpecialChars": bool(re.search(r"[^a-zA-Z0-9\s]", text_str)),
    }

    return analysis


async def output_validator(input_data: ScorerParameter):
    """Validate that output matches expected structure."""
    data = input_data["data"]
    output = input_data["output"]

    # Check if output is valid (not null/undefined)
    if output is None:
        return {
            "value": False,
            "explanation": "Output is null or undefined",
        }

    # If there's an expected output, compare
    if data.expected_output is not None:
        # For objects, check if they have the expected structure
        if isinstance(output, dict) and isinstance(data.expected_output, dict):
            import json

            matches = json.dumps(output, sort_keys=True) == json.dumps(
                data.expected_output, sort_keys=True
            )
            return {
                "value": matches,
                "explanation": (
                    "Output matches expected structure"
                    if matches
                    else "Output does not match expected structure"
                ),
            }

        # For primitives, direct comparison
        matches = output == data.expected_output
        return {
            "value": matches,
            "explanation": (
                "Output matches expected value"
                if matches
                else f"Expected {data.expected_output}, got {output}"
            ),
        }

    # No expected output, just validate the output exists
    return {
        "value": True,
        "explanation": "Output exists (no expected output to compare)",
    }


async def main():
    """Run the CLI evaluation example."""
    _ = await evaluatorq(
        "dataset-evaluation",
        {
            "data": [
                DataPoint(
                    inputs={"text": "Hello joke"},
                    expected_output={
                        "length": 10,
                        "wordCount": 2,
                        "hasNumbers": False,
                        "hasSpecialChars": False,
                    },
                ),
            ],
            "jobs": [text_analyzer],
            "evaluators": [
                {"name": "output-validator", "scorer": output_validator},
            ],
            "parallelism": 2,
            "print": True,
        },
    )


if __name__ == "__main__":
    asyncio.run(main())
