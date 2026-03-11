"""
Reusable evaluator functions for the evaluatorq-py examples.

This module contains various evaluator implementations that can be used
across different examples, including length validators, content validators,
and LLM-based evaluators.
"""

import json
from typing import Any, Sized, cast

from anthropic import AsyncAnthropic

from evaluatorq import Evaluator, ScorerParameter


def max_length_validator(max_length: int) -> Evaluator:
    """
    Create an evaluator that checks if output length does not exceed a maximum.

    Args:
            max_length: Maximum allowed length

    Returns:
            Evaluator that validates length constraint
    """

    async def scorer(input_data: ScorerParameter) -> dict[str, Any]:
        output = input_data["output"]

        if output is None:
            return {
                "value": False,
                "explanation": "Output is null or undefined",
            }

        # Check if output has a length attribute
        if not hasattr(output, "__len__"):
            return {
                "value": False,
                "explanation": f"Output type {type(output).__name__} has no length",
            }
        has_valid_length = len(cast(Sized, output)) <= max_length

        return {
            "value": has_valid_length,
            "explanation": (
                f"Length is within limit (≤{max_length})"
                if has_valid_length
                else f"Length exceeds limit of {max_length}"
            ),
        }

    return Evaluator(name=f"max-length-{max_length}", scorer=scorer)


def min_length_validator(min_length: int) -> Evaluator:
    """
    Create an evaluator that checks if string output meets a minimum length.

    Args:
            min_length: Minimum required length

    Returns:
            Evaluator that validates minimum length constraint
    """

    async def scorer(input_data: ScorerParameter) -> dict[str, Any]:
        output = input_data["output"]

        if output is None:
            return {
                "value": False,
                "explanation": "Output is null or undefined",
            }

        meets_min_length = isinstance(output, str) and len(output) >= min_length

        if meets_min_length:
            explanation = f"String length meets minimum (≥{min_length})"
        elif isinstance(output, str):
            explanation = f"String length {len(output)} is below minimum {min_length}"
        else:
            explanation = "Output is not a string"

        return {
            "value": meets_min_length,
            "explanation": explanation,
        }

    return Evaluator(name=f"min-length-{min_length}", scorer=scorer)


async def contains_name_scorer(input_data: ScorerParameter) -> dict[str, Any]:
    """
    Scorer function that checks if output contains the name from input data.

    Args:
            input_data: Scorer input containing data and output

    Returns:
            Score result with boolean value and explanation
    """
    output = input_data["output"]
    data = input_data["data"]

    if output is None:
        return {
            "value": False,
            "explanation": "Output is null or undefined",
        }

    name = str(data.inputs.get("name", ""))
    contains_name = name in str(output)

    return {
        "value": contains_name,
        "explanation": (
            f'Output contains the name "{name}"'
            if contains_name
            else f'Output does not contain the name "{name}"'
        ),
    }


# Pre-configured evaluator instance
contains_name_validator = Evaluator(name="contains-name", scorer=contains_name_scorer)


# Initialize Anthropic client for LLM-based evaluation
claude = AsyncAnthropic()


async def is_it_polite_scorer(input_data: ScorerParameter) -> dict[str, Any]:
    """
    LLM-based scorer that evaluates politeness of the output.

    Uses Claude to score how polite a response is on a scale from 0 to 1.

    Args:
            input_data: Scorer input containing the output to evaluate

    Returns:
            Score result with float value (0-1) and explanation
    """
    output = input_data["output"]

    response = await claude.messages.create(
        stream=False,
        max_tokens=200,
        model="claude-3-5-haiku-latest",
        messages=[
            {
                "role": "user",
                "content": f"""Evaluate how polite the following response is on a scale from 0 to 1, where 0 is extremely rude and 1 is extremely polite.

<response_to_evaluate>
{output}
</response_to_evaluate>

Return ONLY valid JSON in this format:
{{"score": 0.85, "explanation": "Brief explanation of the score"}}

The score must be a float between 0 and 1.""",
            }
        ],
    )

    try:
        text = response.content[0].text if response.content[0].type == "text" else "{}"
        result = json.loads(text)
        score = result.get("score", 0)
        explanation = result.get("explanation", f"Politeness score: {score:.2f}")

        return {
            "value": score,
            "explanation": explanation,
        }
    except Exception as error:
        print(f"Failed to parse politeness score: {error}")
        print(response)
        return {
            "value": 0,
            "explanation": "Failed to evaluate politeness (LLM response parsing error)",
        }


# Pre-configured LLM evaluator instance
is_it_polite_llm_eval = Evaluator(name="is-it-polite", scorer=is_it_polite_scorer)
