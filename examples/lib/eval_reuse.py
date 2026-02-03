"""
Example demonstrating job and evaluator reuse.

This example shows how to define jobs and evaluators that can be
reused across multiple evaluations, promoting code modularity.
"""

import asyncio
import re
from typing import Any

from evaluatorq import DataPoint, evaluatorq, job

from .evals import max_length_validator


# Define a reusable job for text analysis
@job("text-analyzer")
async def text_analysis_job(data: DataPoint, _row: int = 0) -> dict[str, Any]:
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


async def main():
    """Run evaluation with reusable job and evaluator."""
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
            "jobs": [text_analysis_job],
            "evaluators": [max_length_validator(10)],
            "parallelism": 2,
            "print": True,
        },
    )


if __name__ == "__main__":
    asyncio.run(main())
