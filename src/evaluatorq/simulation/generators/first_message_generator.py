"""First message generator using LLM."""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from openai import AsyncOpenAI
    from openai.types.chat import ChatCompletionMessageParam

from evaluatorq.common.llm_call import execute_response
from evaluatorq.common.responses import first_responses_refusal, responses_stop_reason
from evaluatorq.common.retry import _is_retryable_error, with_retry
from evaluatorq.common.structured_output import warn_unread_config_fields
from evaluatorq.common.tracing import record_llm_input
from evaluatorq.contracts import LLMCallConfig  # noqa: TC001
from evaluatorq.simulation.tracing import with_llm_span
from evaluatorq.simulation.types import DEFAULT_MODEL, Persona, Scenario
from evaluatorq.simulation.utils.prompt_builders import (
    build_persona_system_prompt,
    build_scenario_user_context,
)

logger = logging.getLogger(__name__)

_MAX_OUTPUT_TOKENS = 500
# One message of one size, so the budget and the endpoint are this call site's.
_READ_CONFIG_FIELDS = frozenset({
    'model',
    'client',
    'temperature',
    'reasoning_effort',
    'extra_body',
    'extra_kwargs',
    'timeout_ms',
})

# An opening line is a small, fast call; a minute is already pathological.
_TIMEOUT_S = 60.0

# Attempts for a reply with no usable text (empty or refused). Transport errors
# are retried separately, inside each attempt, by `with_retry`.
_CONTENT_ATTEMPTS = 3


class FirstMessageGenerationError(RuntimeError):
    """The model produced no usable first message; the datapoint should fail."""


def is_recoverable_first_message_failure(exc: Exception) -> bool:
    """Drop one pair for unusable output or an exhausted transient provider failure.

    Authentication, request/configuration, and unexpected errors must abort the
    batch so callers do not mistake incomplete input for a valid dataset.
    """
    return isinstance(exc, FirstMessageGenerationError) or _is_retryable_error(exc)


_FIRST_MESSAGE_PROMPT = """You are generating the authentic first message a user would type to a support agent.

## Your Task
Create a realistic opening message that sounds like an ACTUAL customer, not a script.

## Guidelines

### Voice Matching (based on persona traits):
- **Communication style "terse"**: Short sentences, minimal pleasantries, gets straight to the point
- **Communication style "verbose"**: Detailed explanations, context, multiple sentences
- **Communication style "formal"**: Professional language, complete sentences, "Dear", "Sincerely"
- **Communication style "casual"**: Contractions, slang, emojis if appropriate, friendly tone

- **Low patience (0-0.3)**: Frustrated tone, urgency indicators ("I've been waiting", "This is ridiculous")
- **High patience (0.7-1.0)**: Calm, understanding, may apologize for bothering

- **Low politeness (0-0.3)**: Direct, potentially demanding, no pleasantries
- **High politeness (0.7-1.0)**: "Please", "Thank you", "I appreciate your help"

- **Low technical level (0-0.3)**: Simple language, may describe problems in non-technical terms
- **High technical level (0.7-1.0)**: Technical terminology, specific error codes, detailed descriptions

### Emotional States:
- **Frustrated**: Caps for emphasis, exclamation marks, expressions of disappointment
- **Confused**: Questions, uncertainty ("I'm not sure if...", "Am I doing something wrong?")
- **Urgent**: Time pressure mentioned, immediate action requested
- **Happy**: Positive tone, compliments, appreciation
- **Neutral**: Matter-of-fact, balanced

### Message Length:
- Keep messages 50-200 characters for "terse" style
- Allow 150-400 characters for "verbose" style
- Target 80-250 characters for "casual" or "formal"

### DO:
- Include specific details from the scenario context
- Sound like a real person typing quickly (minor imperfections are OK)
- Match the emotional intensity to the starting_emotion

### DON'T:
- Start with "Dear Support" unless formal style with high politeness
- Be overly long unless verbose style
- Use robotic language ("I am writing to inquire about...")

Return ONLY the message text. No quotes, no explanations, no labels."""


class FirstMessageGenerator:
    """Generates first messages for simulations."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        client: AsyncOpenAI | None = None,
        api_key: str | None = None,
        config: LLMCallConfig | None = None,
    ) -> None:
        """``config`` carries the sampling settings for this generator's own LLM
        calls; ``model`` is the shorthand for setting just the model on it. When
        both set the model, ``config.model`` wins and the contradiction is
        logged — same rule, same warning, as the public entry points.
        """
        from evaluatorq.simulation._config import resolve_sim_llm_config

        self._config = resolve_sim_llm_config(model=model, llm_config=config, caller=type(self).__name__)
        warn_unread_config_fields(self._config, _READ_CONFIG_FIELDS, caller=type(self).__name__)
        self._model = self._config.model
        from evaluatorq.openresponses.client import build_simulation_client

        # `build_simulation_client` owns the precedence, so the config's client must reach it rather than be checked after.
        self._client, self._client_owned = build_simulation_client(
            client or self._config.client,
            extra_api_key=api_key,
            max_retries=0,
        )

    async def close(self) -> None:
        """Close the HTTP client (only if this generator built it)."""
        if self._client_owned:
            await self._client.close()

    async def generate(self, persona: Persona, scenario: Scenario) -> str:
        """Generate a first message for a simulation.

        Transport errors are retried by ``with_retry`` (client retries are
        disabled); an empty or refused reply is retried up to
        ``_CONTENT_ATTEMPTS`` times.

        Raises:
            FirstMessageGenerationError: no usable message was produced. There is
                no canned fallback: the caller fails that datapoint instead.
            APIStatusError: the provider call failed after retries.
        """
        persona_context = build_persona_system_prompt(persona)
        scenario_context = build_scenario_user_context(scenario)

        user_prompt = f"""PERSONA:
{persona_context}

SCENARIO:
{scenario_context}

Generate the FIRST message this user would send to start the conversation.
The message should immediately convey their goal and emotional state.
Keep it natural - this is how they would actually open a conversation."""

        messages: list[ChatCompletionMessageParam] = cast(
            'list[ChatCompletionMessageParam]',
            [
                {'role': 'system', 'content': _FIRST_MESSAGE_PROMPT},
                {'role': 'user', 'content': user_prompt},
            ],
        )

        async with with_llm_span(
            model=self._model,
            operation='responses',
            max_tokens=_MAX_OUTPUT_TOKENS,
            temperature=self._config.temperature,
            purpose='first_message',
        ) as span:
            record_llm_input(
                span,
                [
                    {'role': str(m['role']), 'content': str(m.get('content', ''))}  # pyright: ignore[reportAttributeAccessIssue]
                    for m in messages
                ],
            )
            # RES-1295: `generate()` returns a bare `str`, so the usage
            # execute_response now prices has nowhere to go — carrying it
            # would mean widening this public return type. See "What the
            # totals do not include" in docs/guides/red-teaming.md.
            failure = 'no attempt made'
            for attempt in range(1, _CONTENT_ATTEMPTS + 1):
                response, _usage = await with_retry(
                    lambda: execute_response(
                        client=self._client,
                        model=self._model,
                        messages=cast('list[dict[str, Any]]', messages),
                        span=span,
                        timeout_s=self._config.timeout_s(_TIMEOUT_S),
                        max_output_tokens=_MAX_OUTPUT_TOKENS,
                        **self._config.set_values('temperature', 'reasoning_effort', 'extra_body', 'extra_kwargs'),
                    ),
                    label='FirstMessageGenerator.generate',
                )

                if responses_stop_reason(response) == 'length':
                    # Retrying at the same budget would truncate identically.
                    raise FirstMessageGenerationError(
                        f'response truncated at max_output_tokens={_MAX_OUTPUT_TOKENS}; '
                        'raise the budget to get a persona-shaped opening'
                    )
                refusal = first_responses_refusal(response)
                message = re.sub(r'^["\']|["\']$', '', (response.output_text or '').strip())
                if refusal is None and message:
                    logger.debug('Generated first message: %s...', message[:100])
                    return message
                failure = f'model refused: {refusal}' if refusal is not None else 'empty content'
                if attempt < _CONTENT_ATTEMPTS:
                    logger.info('FirstMessageGenerator: %s, retrying (attempt %d)', failure, attempt + 1)

        raise FirstMessageGenerationError(f'no usable first message after {_CONTENT_ATTEMPTS} attempts ({failure})')
