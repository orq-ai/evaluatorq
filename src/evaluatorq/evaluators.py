"""
Built-in evaluators for evaluatorq.

This module provides commonly used evaluators that can be used with the evaluatorq framework.
"""

from __future__ import annotations

from typing import Any

from .types import Evaluator, ScorerParameter


def string_contains_evaluator(
    case_insensitive: bool = True,
    name: str = "string-contains",
) -> Evaluator:
    """
    Creates an evaluator that checks if the output contains the expected output.
    Uses the data.expected_output from the dataset to compare against.

    Args:
        case_insensitive: Whether the comparison should be case-insensitive
        name: Optional name for the evaluator

    Returns:
        An Evaluator that checks if output contains expected output

    Example:
        # Basic usage
        evaluator = string_contains_evaluator()

        # With case-sensitive matching
        strict_evaluator = string_contains_evaluator(case_insensitive=False)

        # With custom name
        my_evaluator = string_contains_evaluator(name="my-contains-check")
    """

    async def scorer(params: ScorerParameter) -> dict[str, Any]:
        data = params["data"]
        output = params["output"]

        # Convert to strings for comparison - this is intentional to allow flexible
        # matching of various output types (dicts, objects, etc.) as their string repr
        expected = str(data.expected_output) if data.expected_output is not None else ""
        actual = str(output) if output is not None else ""

        if not expected:
            return {
                "value": 0,
                "pass_": False,
                "explanation": "No expected output defined",
            }

        expected_normalized = expected.casefold() if case_insensitive else expected
        actual_normalized = actual.casefold() if case_insensitive else actual

        contains = expected_normalized in actual_normalized

        # Truncate strings for readable explanations
        truncated_expected = expected[:100] + "..." if len(expected) > 100 else expected
        truncated_actual = actual[:100] + "..." if len(actual) > 100 else actual

        if contains:
            return {
                "value": 1.0,
                "pass_": True,
                "explanation": f'Output contains "{truncated_expected}"',
            }
        else:
            return {
                "value": 0.0,
                "pass_": False,
                "explanation": f'Expected "{truncated_expected}" not found in: "{truncated_actual}"',
            }

    return {
        "name": name,
        "scorer": scorer,
    }


def exact_match_evaluator(
    *,
    case_insensitive: bool = False,
    name: str = "exact-match",
) -> Evaluator:
    """
    Creates an evaluator that checks if the output exactly matches the expected output.
    Uses the data.expected_output from the dataset to compare against.

    Args:
        case_insensitive: Whether the comparison should be case-insensitive
        name: Optional name for the evaluator

    Returns:
        An Evaluator that checks if output exactly matches expected output

    Example:
        # Basic usage (case-sensitive)
        evaluator = exact_match_evaluator()

        # With case-insensitive matching
        loose_evaluator = exact_match_evaluator(case_insensitive=True)
    """

    async def scorer(params: ScorerParameter) -> dict[str, Any]:
        data = params["data"]
        output = params["output"]

        # Convert to strings for comparison - this is intentional to allow flexible
        # matching of various output types (dicts, objects, etc.) as their string repr
        expected = str(data.expected_output) if data.expected_output is not None else ""
        actual = str(output) if output is not None else ""

        if not expected:
            return {
                "value": 0,
                "pass_": False,
                "explanation": "No expected output defined",
            }

        expected_normalized = expected.casefold() if case_insensitive else expected
        actual_normalized = actual.casefold() if case_insensitive else actual

        matches = expected_normalized == actual_normalized

        # Truncate strings for readable explanations
        truncated_expected = expected[:100] + "..." if len(expected) > 100 else expected
        truncated_actual = actual[:100] + "..." if len(actual) > 100 else actual

        if matches:
            return {
                "value": 1.0,
                "pass_": True,
                "explanation": "Output exactly matches expected output",
            }
        else:
            return {
                "value": 0.0,
                "pass_": False,
                "explanation": f'Expected "{truncated_expected}" but got "{truncated_actual}"',
            }

    return {
        "name": name,
        "scorer": scorer,
    }
