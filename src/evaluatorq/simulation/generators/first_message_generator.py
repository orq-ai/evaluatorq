"""First message generator using LLM."""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any, cast

from openai import APIStatusError

if TYPE_CHECKING:
    from openai import AsyncOpenAI
    from openai.types.chat import ChatCompletionMessageParam

from evaluatorq.common.llm_call import execute_response
from evaluatorq.common.responses import first_responses_refusal, responses_stop_reason
from evaluatorq.common.retry import with_retry
from evaluatorq.common.tracing import record_llm_input
from evaluatorq.contracts import LLMCallConfig
from evaluatorq.simulation.tracing import with_llm_span
from evaluatorq.simulation.types import DEFAULT_MODEL, Persona, Scenario
from evaluatorq.simulation.utils.prompt_builders import (
    build_persona_system_prompt,
    build_scenario_user_context,
)

logger = logging.getLogger(__name__)

_MAX_OUTPUT_TOKENS = 500
# An opening line is a small, fast call; a minute is already pathological.
_TIMEOUT_S = 60.0

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
        both are given ``config.model`` wins, because a caller who built a whole
        config said everything they meant to say.
        """
        self._config = config if config is not None else LLMCallConfig(model=model)
        self._model = self._config.model
        from evaluatorq.openresponses.client import build_simulation_client

        self._client, self._client_owned = build_simulation_client(
            client,
            extra_api_key=api_key,
            max_retries=0,
        )

    async def close(self) -> None:
        """Close the HTTP client (only if this generator built it)."""
        if self._client_owned:
            await self._client.close()

    async def generate(self, persona: Persona, scenario: Scenario) -> str:
        """Generate a first message for a simulation.

        Retry is owned by ``with_retry``; client retries are disabled.
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

        try:
            async with with_llm_span(
                model=self._model,
                operation='responses',
                max_tokens=_MAX_OUTPUT_TOKENS,
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
                message = ''
                for attempt in range(2):
                    response, _usage = await with_retry(
                        lambda: execute_response(
                            client=self._client,
                            model=self._model,
                            messages=cast('list[dict[str, Any]]', messages),
                            span=span,
                            timeout_s=self._config.timeout_ms / 1000.0
                            if 'timeout_ms' in self._config.model_fields_set
                            else _TIMEOUT_S,
                            max_output_tokens=_MAX_OUTPUT_TOKENS,
                            temperature=self._config.temperature,
                            reasoning_effort=self._config.reasoning_effort,
                            extra_body=self._config.extra_body or None,
                            extra_kwargs=self._config.extra_kwargs or None,
                        ),
                        label='FirstMessageGenerator.generate',
                    )

                    refusal = first_responses_refusal(response)
                    if refusal is not None:
                        logger.warning('FirstMessageGenerator: model refused first message: %s', refusal)
                        break
                    if responses_stop_reason(response) == 'length':
                        logger.warning(
                            'FirstMessageGenerator: response truncated at max_output_tokens=%s before any text; '
                            'raise the budget to get a persona-shaped opening',
                            _MAX_OUTPUT_TOKENS,
                        )
                        break
                    message = re.sub(r'^["\']|["\']$', '', (response.output_text or '').strip())
                    if message:
                        break
                    # A reasoning model can spend the whole budget before it
                    # answers, so empty text here often means truncation rather
                    # than a lazy model. Retrying at the same budget would
                    # truncate identically — name the cause and stop.
                    if attempt == 0:
                        logger.info('FirstMessageGenerator: LLM returned empty content, retrying once')
                else:
                    message = ''

            if not message:
                logger.warning('FirstMessageGenerator: LLM returned empty content after retry, using generic fallback')
                return f'Hi, I need help with: {scenario.goal}'

            logger.debug('Generated first message: %s...', message[:100])
            return message

        except APIStatusError as e:
            # Re-raise client errors (auth, bad request, model-not-found, …) —
            # those are real misconfigurations, not transient, and a canned
            # message would silently mask them. Only fall back for persistent
            # server (5xx) / rate-limit (429) errors that survived with_retry, so
            # a long run isn't aborted by an infra blip; log loudly so the
            # degraded input is visible.
            if e.status_code < 500 and e.status_code != 429:
                raise
            logger.warning(
                'FirstMessageGenerator: generation failed after retries (HTTP %s); '
                'using a generic first message for this datapoint. Error: %s',
                e.status_code,
                e,
            )
            return f'Hi, I need help with: {scenario.goal}'
