"""
Example runner with simulated delays for realistic async operations.

This module demonstrates parallel job execution with simulated processing times
to show how evaluatorq handles concurrent operations.
"""

import asyncio
import random
from typing import Any

from evaluatorq import DataPoint, ScorerParameter, evaluatorq, job


async def run_simulated_delay_example():
    """
    Run an example evaluation with simulated delays for jobs and evaluators.

    This demonstrates:
    - Multiple parallel jobs with different processing times
    - Async evaluator execution
    - Handling different types of outputs (strings and numbers)
    """
    # Create data points with various queries
    data_points = [
        DataPoint(
            inputs={"query": "What is the capital of France?", "userId": "user-123"},
            expected_output="Paris",
        ),
        DataPoint(
            inputs={"query": "Calculate 42 * 17", "userId": "user-456"},
            expected_output=714,
        ),
        DataPoint(
            inputs={"query": "What year was JavaScript created?", "userId": "user-789"},
            expected_output=1995,
        ),
        DataPoint(
            inputs={"query": "Name the largest planet", "userId": "user-012"},
            expected_output="Jupiter",
        ),
    ]

    # Job 1: Simulated LLM response (takes 500-1500ms)
    @job("llm-response")
    async def llm_response(data: DataPoint, _row: int) -> str | int:
        processing_time = 0.5 + random.random()
        await asyncio.sleep(processing_time)

        # Simulate different responses based on query
        query = str(data.inputs.get("query", ""))

        if "capital of France" in query:
            return "Paris"
        elif "42 * 17" in query:
            return 714
        elif "JavaScript created" in query:
            return 1995
        elif "largest planet" in query:
            return "Jupiter"
        else:
            return "Unknown query"

    # Job 2: Simulated context retrieval (takes 200-800ms)
    @job("context-retrieval")
    async def context_retrieval(data: DataPoint, _row: int) -> str:
        processing_time = 0.2 + random.random() * 0.6
        await asyncio.sleep(processing_time)

        user_id = data.inputs.get("userId", "unknown")
        return f"Retrieved context for user {user_id}"

    # Evaluator 1: Accuracy checker
    async def accuracy_checker(input_data: ScorerParameter) -> dict[str, Any]:
        # Simulate evaluator processing time (100-400ms)
        await asyncio.sleep(0.1 + random.random() * 0.3)

        output = input_data["output"]
        data = input_data["data"]

        # Check if the output matches expected
        is_correct = output == data.expected_output
        return {
            "value": 1.0 if is_correct else 0.0,
            "explanation": (
                "Output matches expected result"
                if is_correct
                else f"Expected {data.expected_output}, got {output}"
            ),
        }

    # Evaluator 2: Response validator
    async def response_validator(input_data: ScorerParameter) -> dict[str, Any]:
        # Simulate validation processing time (150-350ms)
        await asyncio.sleep(0.15 + random.random() * 0.2)

        output = input_data["output"]

        # Simple validation: check if output is not empty
        if isinstance(output, str) and len(output) > 0:
            return {
                "value": True,
                "explanation": "Valid string response",
            }
        elif isinstance(output, (int, float)):
            return {
                "value": True,
                "explanation": "Valid numeric response",
            }

        return {
            "value": False,
            "explanation": "Invalid or empty response",
        }

    # Evaluator 3: Latency scorer
    async def latency_scorer(_input_data: ScorerParameter) -> dict[str, Any]:
        # Simulate scoring based on response time (50-150ms)
        await asyncio.sleep(0.05 + random.random() * 0.1)

        # Return a simulated latency score
        score = 0.85 + random.random() * 0.15  # Score between 0.85 and 1.0
        return {
            "value": score,
            "explanation": "Excellent latency" if score >= 0.95 else "Good latency",
        }

    # Run evaluation
    result = await evaluatorq(
        "simulated-llm-evaluation",
        {
            "data": data_points,
            "jobs": [llm_response, context_retrieval],
            "evaluators": [
                {"name": "accuracy-checker", "scorer": accuracy_checker},
                {"name": "response-validator", "scorer": response_validator},
                {"name": "latency-scorer", "scorer": latency_scorer},
            ],
            "parallelism": 2,  # Process 2 data points at a time
            "print": True,  # Display the table
        },
    )

    return result
