"""
Example demonstrating dataset-based evaluation using Orq AI platform.

This example shows how to:
- Load data from an Orq dataset using datasetId
- Process data with multiple jobs
- Evaluate outputs with custom evaluators
"""

import asyncio
import re
from typing import Any

from evaluatorq import DataPoint, ScorerParameter, evaluatorq, job


async def main():
    # Job 1: Text analysis job
    @job("text-analyzer")
    async def text_analyzer(data: DataPoint, _row: int) -> dict[str, Any]:
        text = data.inputs.get("text") or data.inputs.get("input") or ""
        text_str = str(text)

        analysis = {
            "length": len(text_str),
            "wordCount": len([w for w in text_str.split() if w]),
            "hasNumbers": bool(re.search(r"\d", text_str)),
            "hasSpecialChars": bool(re.search(r"[^a-zA-Z0-9\s]", text_str)),
        }

        return analysis

    # Job 2: Simple transformation job
    @job("text-normalizer")
    async def text_normalizer(data: DataPoint, _row: int) -> str:
        input_text = data.inputs.get("text") or data.inputs.get("input") or ""
        transformed = str(input_text).lower()
        # Replace non-alphanumeric chars with space
        transformed = re.sub(r"[^a-z0-9]", " ", transformed)
        # Replace multiple spaces with single space
        transformed = re.sub(r"\s+", " ", transformed)
        transformed = transformed.strip()

        return transformed

    # Evaluator 1: Output validator
    async def output_validator(input_data: ScorerParameter):
        data = input_data["data"]
        output = input_data["output"]

        # Check if output is valid (not null/undefined)
        if output is None:
            return {
                "value": 0,
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
                    "value": 1 if matches else 0.5,
                    "explanation": (
                        "Output exactly matches expected structure"
                        if matches
                        else "Output structure partially matches expected"
                    ),
                }

            # For primitives, direct comparison
            matches = output == data.expected_output
            return {
                "value": 1 if matches else 0,
                "explanation": (
                    "Output matches expected value"
                    if matches
                    else f"Expected {data.expected_output}, got {output}"
                ),
            }

        # No expected output, just validate the output exists
        return {
            "value": 1,
            "explanation": "Output exists (no expected output to compare)",
        }

    # Evaluator 2: Performance scorer
    async def performance_scorer(input_data: ScorerParameter):
        output = input_data["output"]

        # Simple performance score based on output characteristics
        if isinstance(output, dict):
            # For object outputs (like from text-analyzer)
            key_count = len(output.keys())
            import random

            score = (0.8 if key_count > 0 else 0.2) + random.random() * 0.2
            return {
                "value": score,
                "explanation": f"Object with {key_count} properties analyzed",
            }
        elif isinstance(output, str):
            # For string outputs (like from text-normalizer)
            score = 0.9 if len(output) > 0 else 0.1
            return {
                "value": score,
                "explanation": "Non-empty string output"
                if len(output) > 0
                else "Empty string",
            }

        return {
            "value": 0.5,
            "explanation": "Neutral performance score",
        }

    # Evaluator 3: Contains the word joke
    async def contains_joke(input_data: ScorerParameter):
        output = input_data["output"]
        data = input_data["data"]

        output_str = str(output) if output else ""
        expected_str = (
            str(data.expected_output) if data.expected_output is not None else ""
        )

        has_joke = "joke" in expected_str.lower() or "joke" in output_str.lower()

        return {
            "value": has_joke,
            "explanation": (
                "Contains the word 'joke'"
                if has_joke
                else "Does not contain the word 'joke'"
            ),
        }

    # Run evaluation with dataset from Orq AI platform
    _ = await evaluatorq(
        "dataset-evaluation",
        {
            "data": {
                "dataset_id": "01KAX39A6Z7KZ72TH9QMDB5SNJ",
            },
            "jobs": [text_analyzer, text_normalizer],
            "evaluators": [
                {"name": "output-validator", "scorer": output_validator},
                {"name": "performance-scorer", "scorer": performance_scorer},
                {"name": "contains the word joke", "scorer": contains_joke},
            ],
            "parallelism": 2,
            "print": True,
        },
    )


if __name__ == "__main__":
    asyncio.run(main())
