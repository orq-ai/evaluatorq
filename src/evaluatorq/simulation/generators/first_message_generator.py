"""First message generator using LLM."""

from __future__ import annotations

import logging
import os
import re

from openai import APIStatusError, AsyncOpenAI

from evaluatorq.simulation.types import DEFAULT_MODEL, Persona, Scenario
from evaluatorq.simulation.utils.retry import with_retry
from evaluatorq.simulation.utils.prompt_builders import (
    build_persona_system_prompt,
    build_scenario_user_context,
)

logger = logging.getLogger(__name__)

_TEMPERATURE_FIRST_MESSAGE = 0.8

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
    ) -> None:
        self._model = model
        if client is not None:
            self._client = client
        else:
            resolved_key = api_key or os.environ.get("ORQ_API_KEY")
            if not resolved_key:
                raise ValueError(
                    "ORQ_API_KEY environment variable is not set. Set it or pass api_key/client."
                )
            self._client = AsyncOpenAI(
                base_url=f"{os.environ.get('ORQ_BASE_URL', 'https://api.orq.ai')}/v2/router",
                api_key=resolved_key,
            )

    async def generate(self, persona: Persona, scenario: Scenario) -> str:
        """Generate a first message for a simulation."""
        persona_context = build_persona_system_prompt(persona)
        scenario_context = build_scenario_user_context(scenario)

        user_prompt = f"""PERSONA:
{persona_context}

SCENARIO:
{scenario_context}

Generate the FIRST message this user would send to start the conversation.
The message should immediately convey their goal and emotional state.
Keep it natural - this is how they would actually open a conversation."""

        try:
            response = await with_retry(
                lambda: self._client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": _FIRST_MESSAGE_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=_TEMPERATURE_FIRST_MESSAGE,
                    max_tokens=500,
                ),
                label="FirstMessageGenerator.generate",
            )

            message = response.choices[0].message.content if response.choices else ""
            message = re.sub(r'^["\']|["\']$', "", (message or "").strip())

            if not message:
                logger.warning(
                    "FirstMessageGenerator: LLM returned empty content, using generic fallback"
                )
                return f"Hi, I need help with: {scenario.goal}"

            logger.debug("Generated first message: %s...", message[:100])
            return message

        except APIStatusError as e:
            # Re-throw auth errors
            if e.status_code in (401, 403):
                raise
            logger.warning(
                "FirstMessageGenerator: API call failed, using generic fallback. Error: %s",
                e,
            )
            return f"Hi, I need help with: {scenario.goal}"
