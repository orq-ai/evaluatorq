"""
LangChain Agent — Dataset-Driven Research Evaluation Example

Demonstrates a dataset-driven evaluation scenario with:
  - LangChain createReactAgent with multiple tools
  - Dataset with structured inputs: { city, data }, messages, expected_output
  - System instructions built from dataset inputs (city + data)
  - User prompt extracted from dataset messages
  - OpenResponses output with input: [system, user] messages
  - Multiple evaluators: correctness, tool-usage, quality rubric,
    completeness, and city-relevance
  - Path-based organization for the Orq dashboard
  - Parallel processing

Prerequisites:
  - Set OPENAI_API_KEY and ORQ_API_KEY environment variables
  - Upload a dataset to Orq with columns:
      "city"            — city name (string)
      "data"            — contextual data about the city (string)
      "messages"        — conversation messages (the user prompt as a message)
      "expected_output" — the expected answer (string, optional)

Usage:
  ORQ_API_KEY=... OPENAI_API_KEY=... DATASET_ID=... python examples/lib/integrations/langchain/langgraph_research_eval.py
"""

import asyncio
import os
import re
from typing import Any
from urllib.parse import quote_plus

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from evaluatorq import DataPoint, ScorerParameter, evaluatorq
from evaluatorq.integrations.langchain_integration import wrap_langgraph_agent



# ────────────────────────────────────────────────
# Helpers — extract text and tool calls from OpenResponses output
# ────────────────────────────────────────────────
def extract_text(output: Any) -> str:
    if not isinstance(output, dict):
        return ""
    items: list[dict[str, Any]] = output.get("output", [])
    message = next((item for item in items if item.get("type") == "message"), None)
    if not message:
        return ""
    content_array: list[dict[str, Any]] = message.get("content", [])
    text_content = next((c for c in content_array if c.get("type") == "output_text"), None)
    return text_content.get("text", "") if text_content else ""


def extract_tool_calls(output: Any) -> list[dict[str, Any]]:
    if not isinstance(output, dict):
        return []
    items: list[dict[str, Any]] = output.get("output", [])
    return [item for item in items if item.get("type") == "function_call"]


# ────────────────────────────────────────────────
# Build system instructions from dataset inputs
# ────────────────────────────────────────────────
def build_system_instructions(city: str, data: str) -> str:
    return "\n\n".join([
        f"You are an expert analyst for the city of {city}.",
        f"Use the following context data to inform your answers:\n{data}",
        "Always ground your response in the provided data.",
        "You MUST use your tools (search, calculator, or fact_check) at least once before answering. Search for additional information to supplement the provided data, verify claims with the fact-checker, or use the calculator for any numerical analysis.",
    ])


# ────────────────────────────────────────────────
# Tools
# ────────────────────────────────────────────────


@tool
def search(query: str) -> dict[str, Any]:
    """Search the web for information on a topic."""
    return {
        "results": [
            {
                "title": f"Top result for: {query}",
                "snippet": (
                    f"Comprehensive information about {query}. According to recent studies, "
                    "this topic has significant implications in multiple domains."
                ),
                "url": f"https://example.com/search?q={quote_plus(query)}",
            },
            {
                "title": f"Academic paper: {query}",
                "snippet": (
                    f"A peer-reviewed analysis of {query} published in 2024 found that the "
                    "key factors include scalability, reliability, and cost-effectiveness."
                ),
                "url": f"https://example.com/papers/{quote_plus(query)}",
            },
        ],
    }


@tool
def calculator(expression: str) -> dict[str, Any]:
    """Evaluate a mathematical expression."""
    # NOTE: Uses a hard-coded lookup for demo purposes.
    # In production, use a dedicated math expression library instead.
    known_expressions: dict[str, float] = {
        "2 + 2": 4,
        "10 * 5": 50,
        "100 / 4": 25,
        "3.14 * 2": 6.28,
        "2 ** 10": 1024,
        "(5 + 3) * 2": 16,
        "1000 - 750": 250,
    }
    result = known_expressions.get(expression.strip())
    if result is not None:
        return {"expression": expression, "result": result, "error": None}
    return {"expression": expression, "result": None, "error": "Expression not in demo lookup table"}


@tool
def fact_check(claim: str) -> dict[str, Any]:
    """Verify a factual claim against known sources."""
    confidence = 0.85
    return {
        "claim": claim,
        "verdict": "supported" if confidence >= 0.85 else "partially_supported",
        "confidence": round(confidence, 2),
        "sources": [f"https://example.com/fact-check/{quote_plus(claim[:30])}"],
    }


# ────────────────────────────────────────────────
# LangChain agent — createReactAgent
# ────────────────────────────────────────────────

tools = [search, calculator, fact_check]

model = ChatOpenAI(model="gpt-4o", temperature=0)

agent = create_react_agent(model, tools)


# ────────────────────────────────────────────────
# Evaluators
# ────────────────────────────────────────────────


async def correctness_scorer(params: ScorerParameter) -> dict[str, Any]:
    """Checks correctness against expected output when available."""
    text = extract_text(params["output"]).lower()
    expected = params["data"].expected_output

    if not expected:
        return {
            "value": 1 if len(text) > 20 else 0.5,
            "explanation": "No expected output — scored on response substance",
        }

    expected_str = str(expected).lower()
    contains = expected_str in text
    return {
        "value": 1 if contains else 0,
        "pass": contains,
        "explanation": (
            f'Output contains expected answer "{expected}"'
            if contains
            else f'Expected "{expected}" not found in output'
        ),
    }


async def tool_usage_scorer(params: ScorerParameter) -> dict[str, Any]:
    """Validates that the agent actually used its tools."""
    calls = extract_tool_calls(params["output"])
    tool_names = list(set(c.get("name", "") for c in calls))
    score = min(len(tool_names) / 2, 1.0)
    return {
        "value": round(score, 2),
        "explanation": (
            f"Used {len(calls)} tool call(s) across {len(tool_names)} "
            f"distinct tool(s): {', '.join(tool_names) or 'none'}"
        ),
    }


async def quality_rubric_scorer(params: ScorerParameter) -> dict[str, Any]:
    """Multi-criteria quality rubric (structured result)."""
    text = extract_text(params["output"])
    words = [w for w in text.split() if w]
    sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]

    completeness = min(len(words) / 50, 1.0)

    avg_sentence_len = len(words) / len(sentences) if sentences else 0
    if 10 <= avg_sentence_len <= 25:
        clarity = 0.95
    elif avg_sentence_len > 0:
        clarity = 0.5
    else:
        clarity = 0.1

    has_structure = 0.9 if re.search(r"(\n[-•*]|\n\d+\.|\n\n)", text) else 0.5

    return {
        "value": {
            "type": "rubric",
            "value": {
                "completeness": round(completeness, 2),
                "clarity": round(clarity, 2),
                "structure": round(has_structure, 2),
            },
        },
        "explanation": "Multi-criteria quality rubric (completeness, clarity, structure)",
    }


async def completeness_scorer(params: ScorerParameter) -> dict[str, Any]:
    """Boolean pass/fail — the response must not be empty or a refusal."""
    text = extract_text(params["output"])
    words = [w for w in text.split() if w]
    is_refusal = bool(re.search(r"i (can't|cannot|am unable to)", text, re.IGNORECASE))
    is_complete = len(words) >= 10 and not is_refusal

    return {
        "value": is_complete,
        "pass": is_complete,
        "explanation": (
            f"Complete response ({len(words)} words)"
            if is_complete
            else (
                "Agent refused to answer"
                if is_refusal
                else f"Incomplete response (only {len(words)} words)"
            )
        ),
    }


async def city_relevance_scorer(params: ScorerParameter) -> dict[str, Any]:
    """Checks that the response references the city from the dataset input."""
    text = extract_text(params["output"]).lower()
    city = str(params["data"].inputs.get("city", ""))
    mentions_city = city.lower() in text
    return {
        "value": 1 if mentions_city else 0,
        "pass": mentions_city,
        "explanation": (
            f'Response references the target city "{city}"'
            if mentions_city
            else f'Response does not mention "{city}"'
        ),
    }


# ────────────────────────────────────────────────
# Run the evaluation
# ────────────────────────────────────────────────

DATASET_ID = os.environ.get("DATASET_ID")


async def main() -> None:
    if not DATASET_ID:
        raise ValueError("DATASET_ID environment variable is required")

    await evaluatorq(
        "langchain-research-eval",
        description=(
            "LangChain research agent evaluation with structured dataset input "
            "(city + data), custom instructions, and OpenResponses output"
        ),
        path="Integrations/LangChain",
        parallelism=3,
        data={"dataset_id": DATASET_ID, "include_messages": True},
        jobs=[
            wrap_langgraph_agent(
                agent,
                name="langchain-research-agent",
                instructions=lambda data: build_system_instructions(
                    str(data.inputs.get("city", "")),
                    str(data.inputs.get("data", "")),
                ),
            ),
        ],
        evaluators=[
            {"name": "correctness", "scorer": correctness_scorer},
            {"name": "tool-usage", "scorer": tool_usage_scorer},
            {"name": "quality-rubric", "scorer": quality_rubric_scorer},
            {"name": "completeness", "scorer": completeness_scorer},
            {"name": "city-relevance", "scorer": city_relevance_scorer},
        ],
    )


if __name__ == "__main__":
    asyncio.run(main())
