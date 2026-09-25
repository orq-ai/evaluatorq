"""Cluster description: an LLM name + description for each cluster, contrasted against its neighbours.

One retry layer: `generate_structured`'s own four-rung ladder (see
`common/structured_output.py`'s module docstring) — do not add a second
`with_retry` around this call path, mirroring `summarize.py`.

Per-cluster failure never raises: an exception from `generate_structured`, or a
reply that fails to parse (`result.parsed is None`), is logged and reported
back as an error string for that cluster only, per the "per-trace failures
never fail a run" house rule (here: per-cluster).
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from loguru import logger
from pydantic import BaseModel

from evaluatorq.common.sanitize import delimit
from evaluatorq.common.structured_output import generate_structured
from evaluatorq.common.template_engine import render_template

if TYPE_CHECKING:
    from openai import AsyncOpenAI

    from evaluatorq.insights.models import DimensionName


class ClusterName(BaseModel):
    """An LLM-generated name and description for one cluster."""

    name: str
    description: str


# Ported from `~/Developer/orq/orq-traces-intelligence`'s
# `trace_intelligence/dimensions/prompts.py`, condensed to the `ClusterName`
# schema (name + description only — no `slug`) and adapted from the upstream
# `.format()`-style `{positive_examples}` placeholders to `render_template`'s
# `{{...}}` syntax. The example/contrast text blocks are rendered through
# `delimit` before substitution (never the raw upstream string interpolation),
# so an attacker-controlled trace summary cannot break out of the prompt.
_INTENT_PROMPT = """You are tasked with describing a group of related user intents with a short, precise, and accurate description and name. Your goal is to capture what users in this group were trying to accomplish, and to distinguish this group from other nearby groups.

Describe the group in 3-4 clear sentences in the past tense. Focus exclusively on the user's goal or task — not on the outcome, quality of the response, or the user's emotional state. Your description should distinguish this group from the contrastive examples.

After creating the description, generate a short name for the group. The name must be an imperative sentence that describes the user's intent (e.g., 'Configure CI/CD pipelines for Python projects', 'Debug memory leaks in Node.js applications', or 'Generate product descriptions for an e-commerce site'). The name should be at most ten words long, specific enough to be meaningful, but broad enough to represent the whole group.

The name should distinguish this group from the contrastive examples by the specific intent or goal, not by success, failure, or sentiment. NEVER start the name with 'Summarize' — you are describing a user intent, not performing summarization.

Provide your response as structured JSON with these fields:
- "name": the imperative cluster name (max 10 words)
- "description": the 3-4 sentence description in past tense

Below are the user intent examples in this group:
{{examples}}

For context, here are examples from nearby groups that are NOT in this group:
{{contrastive}}

Analyze the examples carefully and focus on the user's intent and task, not on sentiment, frustration, or whether the assistant succeeded."""

_SENTIMENT_PROMPT = """You are tasked with describing a group of related user sentiment patterns. Your goal is to capture the root cause shared by users in this group, and to distinguish this group from other nearby groups.

The examples below are sentiment descriptions — each describes how a user felt and why. Your job is to describe the cause of the sentiment pattern, NOT to summarize conversations.

Describe the group in 3-4 clear sentences in the past tense. Focus on the emotional pattern and WHY it occurred, not on what the user was doing. Your description should distinguish this group from the contrastive examples.

After creating the description, generate a short name for the group. The name must describe the root cause (e.g., 'Frustrated by repeated incorrect code suggestions', 'Satisfied after completing a complex migration in one exchange', or 'Neutral toward verbose but technically correct responses'). The name should be at most ten words long.

The name should distinguish this group from the contrastive examples by the specific cause, not by the task or topic. NEVER start the name with 'Summarize' — you are describing a sentiment pattern, not performing summarization.

Provide your response as structured JSON with these fields:
- "name": the sentiment pattern name (max 10 words), describing the root cause
- "description": the 3-4 sentence description in past tense

Below are the sentiment description examples in this group:
{{examples}}

For context, here are examples from nearby groups that are NOT in this group:
{{contrastive}}

Analyze the examples carefully and focus on the root cause of the sentiment."""

_FAILURE_PROMPT = """You are tasked with describing a group of related assistant failure patterns with a short, precise, and accurate description and name. Your goal is to capture the specific failure mechanism shared across this group, and to distinguish it from other failure modes in nearby groups.

Describe the group in 3-4 clear sentences in the past tense. Focus on WHAT went wrong and HOW — the specific failure mechanism — not on the user's sentiment or the topic of the conversation. Your description should distinguish this group from the contrastive examples.

After creating the description, generate a short name for the group. The name must describe the specific failure mode (e.g., 'Hallucinated non-existent API endpoints', 'Truncated code completions without warning', or 'Produced syntactically valid but logically incorrect SQL queries'). The name should be at most ten words long.

Special case: if this group consists of conversations with no assistant failures, name it 'No failures detected' and describe what successful interactions in this group looked like.

The name should distinguish this group from the contrastive examples by the specific failure mechanism — not by the topic, user sentiment, or severity. NEVER start the name with 'Summarize' — you are describing a failure pattern, not performing summarization.

Provide your response as structured JSON with these fields:
- "name": the failure mode name (max 10 words), describing what went wrong
- "description": the 3-4 sentence description in past tense

Below are the failure description examples in this group:
{{examples}}

For context, here are examples from nearby groups that are NOT in this group:
{{contrastive}}

Analyze the examples carefully and focus on the specific failure mechanism."""

_DIMENSION_PROMPTS: dict[DimensionName, str] = {
    'intent': _INTENT_PROMPT,
    'failure': _FAILURE_PROMPT,
    'sentiment': _SENTIMENT_PROMPT,
}

_TOP_LEVEL_PROMPT = """You are tasked with naming a broader group of related clusters, given the name and description of each of its child clusters. Your goal is to capture the shared theme across all the children in a short, precise, and accurate name and description.

Describe the group in 2-3 clear sentences in the past tense, capturing what unifies the children. After creating the description, generate a short name for the group, at most ten words long, that reflects the shared theme of its children rather than any single one of them.

Provide your response as structured JSON with these fields:
- "name": the group name (max 10 words)
- "description": the 2-3 sentence description in past tense

Below are the child clusters of this group:
{{children}}

Analyze the children carefully and focus on what unifies them."""

_MAX_MEMBER_EXAMPLES = 10
_MAX_NEIGHBOUR_CLUSTERS = 3
_MAX_NEIGHBOUR_EXAMPLES = 3


def _build_describe_prompt(dimension: DimensionName, member_texts: list[str], contrastive_texts: list[str]) -> str:
    examples = delimit('\n'.join(member_texts) if member_texts else '(none)', tag='examples')
    contrastive = delimit('\n'.join(contrastive_texts) if contrastive_texts else '(none)', tag='contrastive')
    return render_template(_DIMENSION_PROMPTS[dimension], {'examples': examples, 'contrastive': contrastive})


def _contrastive_texts(members: dict[int, list[str]], neighbour_ids: list[int]) -> list[str]:
    texts: list[str] = []
    for neighbour_id in neighbour_ids[:_MAX_NEIGHBOUR_CLUSTERS]:
        texts.extend(members.get(neighbour_id, [])[:_MAX_NEIGHBOUR_EXAMPLES])
    return texts


async def _describe_one(
    cluster_id: int,
    *,
    members: dict[int, list[str]],
    neighbours: dict[int, list[int]],
    dimension: DimensionName,
    client: AsyncOpenAI,
    model: str,
    semaphore: asyncio.Semaphore,
) -> tuple[int, ClusterName | str]:
    member_texts = members.get(cluster_id, [])[:_MAX_MEMBER_EXAMPLES]
    contrastive_texts = _contrastive_texts(members, neighbours.get(cluster_id, []))
    prompt = _build_describe_prompt(dimension, member_texts, contrastive_texts)
    messages = [{'role': 'user', 'content': prompt}]

    async with semaphore:
        try:
            result = await generate_structured(
                client,
                model=model,
                messages=messages,
                response_format=ClusterName,
                max_tokens=400,
                label='insights.describe',
            )
        except Exception as exc:  # noqa: BLE001 - a per-cluster failure must never fail the run
            message = str(exc)
            logger.warning('Insights cluster description failed for cluster {}: {}', cluster_id, message)
            return cluster_id, f'describe: {message}'

    if result.parsed is None:
        logger.warning('Insights cluster description for cluster {} produced unparseable model output', cluster_id)
        return cluster_id, 'describe: unparseable model output'

    return cluster_id, result.parsed


async def describe_clusters(
    members: dict[int, list[str]],
    *,
    neighbours: dict[int, list[int]],
    dimension: DimensionName,
    client: AsyncOpenAI,
    model: str,
    parallelism: int = 20,
) -> dict[int, ClusterName | str]:
    """Name and describe every cluster, contrasted against its nearest neighbours.

    Per cluster: up to 10 member texts, plus up to 3 texts from each of up to 3
    neighbour clusters (`cluster.nearest_neighbours`) as contrastive examples. A
    per-cluster failure (an exception, or a reply that does not parse) is
    reported as an error string for that cluster only and never raises, per the
    "per-trace failures never fail a run" house rule.
    """
    semaphore = asyncio.Semaphore(parallelism)
    tasks = [
        asyncio.ensure_future(
            _describe_one(
                cluster_id,
                members=members,
                neighbours=neighbours,
                dimension=dimension,
                client=client,
                model=model,
                semaphore=semaphore,
            )
        )
        for cluster_id in members
    ]
    results = await asyncio.gather(*tasks)
    return dict(results)


def _children_text(children: list[ClusterName]) -> str:
    return '\n'.join(f'- {child.name}: {child.description}' for child in children)


async def _describe_top_one(
    top_id: int,
    *,
    children: list[ClusterName],
    client: AsyncOpenAI,
    model: str,
    semaphore: asyncio.Semaphore,
) -> tuple[int, ClusterName | str]:
    rendered = render_template(_TOP_LEVEL_PROMPT, {'children': delimit(_children_text(children), tag='children')})
    messages = [{'role': 'user', 'content': rendered}]

    async with semaphore:
        try:
            result = await generate_structured(
                client,
                model=model,
                messages=messages,
                response_format=ClusterName,
                max_tokens=400,
                label='insights.describe_top_level',
            )
        except Exception as exc:  # noqa: BLE001 - a per-cluster failure must never fail the run
            message = str(exc)
            logger.warning('Insights top-level cluster description failed for group {}: {}', top_id, message)
            return top_id, f'describe: {message}'

    if result.parsed is None:
        logger.warning('Insights top-level cluster description for group {} produced unparseable model output', top_id)
        return top_id, 'describe: unparseable model output'

    return top_id, result.parsed


async def describe_top_level(
    children: dict[int, list[ClusterName]],
    *,
    client: AsyncOpenAI,
    model: str,
    parallelism: int = 20,
) -> dict[int, ClusterName | str]:
    """Name and describe each top-level group from its children's names and descriptions.

    Same per-cluster failure policy as `describe_clusters`: a failure is
    reported as an error string for that top group only and never raises.
    """
    semaphore = asyncio.Semaphore(parallelism)
    tasks = [
        asyncio.ensure_future(
            _describe_top_one(top_id, children=group_children, client=client, model=model, semaphore=semaphore)
        )
        for top_id, group_children in children.items()
    ]
    results = await asyncio.gather(*tasks)
    return dict(results)
