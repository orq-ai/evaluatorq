"""
Cosine Similarity Evaluator Example

This example demonstrates how to use cosine similarity evaluators
to compare semantic similarity between outputs and expected text
using OpenAI embeddings.

Prerequisites:
- Install dependencies: anthropic, openai
- Set ANTHROPIC_API_KEY environment variable for Claude
- Set either ORQ_API_KEY or OPENAI_API_KEY for embeddings

Run with: python example_cosine_similarity.py
"""

import asyncio
import math
import os
from typing import Any

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from evaluatorq import DataPoint, ScorerParameter, evaluatorq, job

# Initialize Anthropic client
claude = AsyncAnthropic()


def create_openai_client() -> AsyncOpenAI:
    """Create OpenAI client configured for Orq proxy or direct API."""
    orq_api_key = os.environ.get("ORQ_API_KEY")
    openai_api_key = os.environ.get("OPENAI_API_KEY")

    if orq_api_key:
        base_url = os.environ.get("ORQ_BASE_URL", "https://api.orq.ai")
        return AsyncOpenAI(
            base_url=f"{base_url}/v2/proxy",
            api_key=orq_api_key,
        )
    if openai_api_key:
        return AsyncOpenAI(api_key=openai_api_key)

    raise ValueError(
        "Cosine similarity evaluator requires either ORQ_API_KEY or "
        + "OPENAI_API_KEY environment variable to be set for embeddings"
    )


def get_embedding_model() -> str:
    """Get the appropriate embedding model based on the environment."""
    if os.environ.get("ORQ_API_KEY"):
        return "openai/text-embedding-3-small"
    return "text-embedding-3-small"


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    if len(vec_a) != len(vec_b):
        raise ValueError(f"Vector dimensions don't match: {len(vec_a)} vs {len(vec_b)}")

    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    magnitude_a = math.sqrt(sum(a * a for a in vec_a))
    magnitude_b = math.sqrt(sum(b * b for b in vec_b))

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)


async def get_embedding(client: AsyncOpenAI, text: str, model: str) -> list[float]:
    """Get embedding vector for text using OpenAI API."""
    response = await client.embeddings.create(input=text, model=model)
    return response.data[0].embedding


def simple_cosine_similarity(expected_text: str) -> dict[str, Any]:
    """
    Create a cosine similarity evaluator that returns the raw similarity score.

    Args:
        expected_text: The expected text to compare against the output.

    Returns:
        An evaluator dict with name and scorer function.
    """
    # Lazy initialization of client
    client: AsyncOpenAI | None = None

    async def scorer(input_data: ScorerParameter) -> dict[str, Any]:
        nonlocal client
        output = input_data["output"]

        if output is None:
            return {
                "value": 0,
                "explanation": "Output is null or undefined",
            }

        output_text = str(output)

        if client is None:
            client = create_openai_client()

        model = get_embedding_model()

        # Get embeddings for both texts
        output_embedding, expected_embedding = await asyncio.gather(
            get_embedding(client, output_text, model),
            get_embedding(client, expected_text, model),
        )

        # Calculate cosine similarity
        similarity = cosine_similarity(output_embedding, expected_embedding)

        return {
            "value": similarity,
            "explanation": f"Cosine similarity: {similarity:.3f}",
        }

    return {"name": "cosine-similarity", "scorer": scorer}


def cosine_similarity_threshold_evaluator(
    expected_text: str,
    threshold: float,
    name: str = "cosine-similarity-threshold",
) -> dict[str, Any]:
    """
    Create a cosine similarity evaluator that returns pass/fail based on threshold.

    Args:
        expected_text: The expected text to compare against the output.
        threshold: Similarity threshold (0-1). Returns True if similarity >= threshold.
        name: Optional name for the evaluator.

    Returns:
        An evaluator dict with name and scorer function.
    """
    # Lazy initialization of client
    client: AsyncOpenAI | None = None

    async def scorer(input_data: ScorerParameter) -> dict[str, Any]:
        nonlocal client
        output = input_data["output"]

        if output is None:
            return {
                "value": False,
                "explanation": "Output is null or undefined",
            }

        output_text = str(output)

        if client is None:
            client = create_openai_client()

        model = get_embedding_model()

        # Get embeddings for both texts
        output_embedding, expected_embedding = await asyncio.gather(
            get_embedding(client, output_text, model),
            get_embedding(client, expected_text, model),
        )

        # Calculate cosine similarity
        similarity = cosine_similarity(output_embedding, expected_embedding)
        meets_threshold = similarity >= threshold

        return {
            "value": meets_threshold,
            "explanation": (
                f"Similarity ({similarity:.3f}) meets threshold ({threshold})"
                if meets_threshold
                else f"Similarity ({similarity:.3f}) below threshold ({threshold})"
            ),
        }

    return {"name": name, "scorer": scorer}


@job("translate-to-french")
async def translate_to_french(data: DataPoint, _row: int = 0) -> str:
    """Translate text to French using Claude."""
    text = str(data.inputs.get("text", ""))

    response = await claude.messages.create(
        model="claude-3-5-haiku-latest",
        max_tokens=100,
        system="You are a translator. Translate the given text to French. Respond only with the translation.",
        messages=[
            {
                "role": "user",
                "content": text,
            }
        ],
    )

    return response.content[0].text if response.content[0].type == "text" else ""


@job("describe-capital")
async def describe_capital(data: DataPoint, _row: int = 0) -> str:
    """Generate capital city descriptions using Claude."""
    country = str(data.inputs.get("country", ""))

    response = await claude.messages.create(
        model="claude-3-5-haiku-latest",
        max_tokens=50,
        system="You are a geography expert. Provide a one-sentence description of the capital city of the given country.",
        messages=[
            {
                "role": "user",
                "content": f"What is the capital of {country}?",
            }
        ],
    )

    return response.content[0].text if response.content[0].type == "text" else ""


async def main():
    """Run cosine similarity evaluation examples."""
    print("🌍 Running translation evaluation...\n")

    # Create evaluators
    french_translation_similarity = simple_cosine_similarity(
        "Bonjour, comment allez-vous?"
    )
    exact_translation_threshold = cosine_similarity_threshold_evaluator(
        expected_text="Le ciel est bleu",
        threshold=0.85,
        name="exact-translation-match",
    )

    # Run evaluation with translation examples
    _ = await evaluatorq(
        "translation-evaluation",
        {
            "data": [
                DataPoint(inputs={"text": "Hello, how are you?"}),
                DataPoint(inputs={"text": "The sky is blue"}),
                DataPoint(inputs={"text": "Good morning"}),
            ],
            "jobs": [translate_to_french],
            "evaluators": [french_translation_similarity, exact_translation_threshold],
            "parallelism": 2,
            "print": True,
        },
    )

    print("\n🗺️ Running capital city evaluation...\n")

    # Create evaluator for capital descriptions
    capital_description_threshold = cosine_similarity_threshold_evaluator(
        expected_text="The capital of France is Paris",
        threshold=0.7,
        name="capital-semantic-match",
    )

    # Run evaluation with capital city descriptions
    _ = await evaluatorq(
        "capital-evaluation",
        {
            "data": [
                DataPoint(inputs={"country": "France"}),
                DataPoint(inputs={"country": "Germany"}),
                DataPoint(inputs={"country": "Japan"}),
            ],
            "jobs": [describe_capital],
            "evaluators": [capital_description_threshold],
            "parallelism": 2,
            "print": True,
        },
    )

    print("\n✅ Cosine similarity evaluation examples completed!")


if __name__ == "__main__":
    asyncio.run(main())
