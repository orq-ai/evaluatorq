"""`LLMConfig.max_objectives_per_llm_call` must drive the objective batching.

The field has exactly one reader — `_call_llm_for_objectives` in
`redteam/adaptive/objective_generator.py` — since the shadowing module constant
`_MAX_PER_LLM_CALL = 8` was deleted. Nothing asserted that the *configured* value
reaches the batching arithmetic, so a regression to a literal would be invisible.

These tests therefore assert the observable consequence of the budget: how many
LLM calls a given objective count is split into, and what per-batch count each
call asks the model for. Asserting "the config object was passed" would pass
against a hardcoded 8; asserting the call count cannot.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

# ruff: noqa: S101
import pytest

from evaluatorq.redteam.adaptive.objective_generator import (
    GeneratedObjective,
    GeneratedObjectives,
    generate_objectives_for_vulnerability,
)
from evaluatorq.redteam.contracts import (
    AgentContext,
    DeliveryMethod,
    LLMConfig,
    TurnType,
    Vulnerability,
)


def _requested_counts(recorded_prompts: list[str]) -> list[int]:
    """Recover the per-call batch size from each rendered prompt.

    `_call_llm_for_objectives_single` fills the template's ``{count}``
    placeholder, so the prompt is where the batching arithmetic becomes
    observable to the model — and the only place a batch size that never left
    Python would show up as wrong.
    """
    counts: list[int] = []
    for prompt in recorded_prompts:
        marker = 'Generate '
        start = prompt.index(marker) + len(marker)
        counts.append(int(prompt[start:].split(' ', 1)[0]))
    return counts


def _recording_client(recorded_prompts: list[str]) -> MagicMock:
    """A client whose `.parse` records the prompt and returns one objective."""

    async def _parse(**params: Any) -> Any:
        recorded_prompts.append(params['messages'][0]['content'])
        parsed = GeneratedObjectives(
            objectives=[
                GeneratedObjective(
                    objective='Redirect the agent goal via injected content',
                    turn_type=TurnType.SINGLE,
                    delivery_method=DeliveryMethod.DIRECT_REQUEST,
                    requires_tools=False,
                    requires_memory=False,
                )
            ]
        )
        choice = MagicMock()
        choice.message.parsed = parsed
        choice.message.content = None
        choice.finish_reason = 'stop'
        response = MagicMock()
        response.choices = [choice]
        response.usage = None
        return response

    client = MagicMock()
    client.base_url = 'https://api.openai.com/v1'
    client.chat.completions.parse = _parse
    return client


async def _generate(count: int, config: LLMConfig | None) -> list[str]:
    """Run the real generator against a recording client; return each prompt."""
    prompts: list[str] = []
    with patch('evaluatorq.redteam.adaptive.objective_generator.with_llm_span') as span:
        span.return_value.__aenter__ = AsyncMock(return_value=MagicMock())
        span.return_value.__aexit__ = AsyncMock(return_value=None)
        await generate_objectives_for_vulnerability(
            vuln=Vulnerability.GOAL_HIJACKING,
            agent_context=AgentContext(key='test_agent'),
            llm_client=_recording_client(prompts),  # pyright: ignore[reportArgumentType]
            model='test-model',
            count=count,
            pipeline_config=config,
        )
    return prompts


@pytest.mark.asyncio
async def test_default_budget_asks_for_all_seven_objectives_in_one_call():
    """The shipped budget is 8, so 7 objectives must not be batched at all."""
    prompts = await _generate(count=7, config=None)
    assert len(prompts) == 1
    assert _requested_counts(prompts) == [7]


@pytest.mark.asyncio
async def test_a_lower_budget_splits_the_same_request_into_batches():
    """max_objectives_per_llm_call=3 must turn one 7-objective request into 3+3+1.

    Three live calls instead of one — the cost knob the field documents. Were the
    reader still using a literal 8, this would make a single call asking for 7 and
    both assertions would fail.
    """
    prompts = await _generate(count=7, config=LLMConfig(max_objectives_per_llm_call=3))
    assert len(prompts) == 3
    assert _requested_counts(prompts) == [3, 3, 1]


@pytest.mark.asyncio
async def test_a_budget_of_one_makes_one_call_per_objective():
    """The `ge=1` boundary: every objective becomes its own single-item call."""
    prompts = await _generate(count=4, config=LLMConfig(max_objectives_per_llm_call=1))
    assert _requested_counts(prompts) == [1, 1, 1, 1]


@pytest.mark.asyncio
async def test_a_budget_above_the_default_collapses_batches_the_default_would_split():
    """Raising the budget must widen batches, not just avoid narrowing them.

    12 objectives are 8+4 under the shipped budget and a single call at 16. A
    reader pinned to the literal 8 would still split, so the one-call assertion
    is what proves the raised value was read.
    """
    default_prompts = await _generate(count=12, config=None)
    assert _requested_counts(default_prompts) == [8, 4]

    raised_prompts = await _generate(count=12, config=LLMConfig(max_objectives_per_llm_call=16))
    assert _requested_counts(raised_prompts) == [12]
