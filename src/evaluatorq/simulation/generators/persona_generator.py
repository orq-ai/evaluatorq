"""Persona generator using LLM."""

from __future__ import annotations

import logging
import os
from typing import Any

from openai import AsyncOpenAI
from pydantic import BaseModel

from evaluatorq.simulation.types import DEFAULT_MODEL, CommunicationStyle, Persona
from evaluatorq.simulation.utils.sanitize import delimit
from evaluatorq.simulation.utils.structured_output import generate_structured

logger = logging.getLogger(__name__)

_TEMPERATURE_CREATIVE = 0.8
_TEMPERATURE_BALANCED = 0.7


_PERSONA_GENERATOR_PROMPT = """You are an expert persona designer for AI agent testing. Create realistic, memorable user personas that feel like real people, not stereotypes.

## Persona Structure
Each persona must include:
- **name**: A vivid, specific descriptor (e.g., "Anxious First-Time Buyer", "Retired Engineer Seeking Help")
- **patience**: 0-1 (0=interrupts constantly, 1=waits indefinitely)
- **assertiveness**: 0-1 (0=accepts anything, 1=demands specific outcomes)
- **politeness**: 0-1 (0=rude/demanding, 1=overly polite)
- **technical_level**: 0-1 (0=never used a computer, 1=software developer)
- **communication_style**: "formal", "casual", "terse", or "verbose"
- **background**: DETAILED context (2-3 sentences) explaining WHO they are, WHY they're contacting support, and WHAT their emotional state is

## Quality Guidelines

### DO create personas that are:
- **Realistic**: Based on actual customer archetypes you'd encounter
- **Coherent**: Traits that logically fit together (e.g., high technical_level + formal style for an engineer)
- **Specific**: Unique situations with concrete details (names, specific products, timeframes)
- **Emotionally grounded**: Clear emotional context that explains their behavior

### DON'T create personas that are:
- Generic (e.g., "Customer with a problem")
- Contradictory (e.g., patience=0.1 but described as "patient and understanding")
- Unrealistic trait combinations (e.g., technical_level=0.9 + communication_style="terse" for a "confused elderly person")
- All similar - vary trait values across personas, including some with low values and some with high values

## Example HIGH-QUALITY Persona:
{
  "name": "Overwhelmed Working Parent",
  "patience": 0.3,
  "assertiveness": 0.6,
  "politeness": 0.7,
  "technical_level": 0.4,
  "communication_style": "terse",
  "background": "Sarah is a working mom with two kids under 5. She ordered a birthday gift (a tablet) for her daughter 2 weeks ago but it hasn't arrived. The party is in 3 days. She's stressed, multitasking while on hold, and needs quick answers without lengthy explanations."
}

## Example LOW-QUALITY Persona (AVOID):
{
  "name": "Angry Customer",
  "patience": 0.1,
  "assertiveness": 0.9,
  "politeness": 0.2,
  "technical_level": 0.5,
  "communication_style": "casual",
  "background": "A customer who is angry about something"
}

Return a JSON array of persona objects."""


class PersonaListResponse(BaseModel):
    """Wrapper for structured output parsing."""

    personas: list[Persona]


class PersonaGenerator:
    """Generates personas from agent descriptions."""

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

    @staticmethod
    def _parse_personas(content: str) -> list[Persona]:
        import json

        try:
            parsed = json.loads(content)
        except (json.JSONDecodeError, ValueError):
            logger.warning("Failed to parse personas JSON response")
            return []

        # json_object mode returns a top-level object; find the first list value
        items: list[Any]
        if isinstance(parsed, list):
            items = parsed
        elif isinstance(parsed, dict):
            items = next(
                (v for v in parsed.values() if isinstance(v, list)), []
            )
        else:
            logger.warning("Failed to parse personas: unexpected JSON shape")
            return []

        personas: list[Persona] = []
        for item in items:
            try:
                personas.append(Persona.model_validate(item))
            except Exception as e:
                logger.warning("Failed to parse persona: %s", e)
        return personas

    async def generate(
        self,
        *,
        agent_description: str,
        context: str = "",
        num_personas: int = 5,
        edge_case_percentage: float = 0.2,
    ) -> list[Persona]:
        """Generate personas for agent testing."""
        num_edge_cases = int(num_personas * edge_case_percentage)

        user_prompt = f"""Agent Description: {delimit(agent_description)}

Additional Context: {delimit(context or "None provided")}

Generate {num_personas} diverse personas for testing this agent.
- Include {num_edge_cases} edge case/challenging personas
- Ensure variety in patience, assertiveness, and technical levels
- Create realistic backgrounds relevant to the agent's domain

Return ONLY a JSON array, no other text."""

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": _PERSONA_GENERATOR_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        parsed, raw = await generate_structured(
            self._client,
            model=self._model,
            messages=messages,
            response_format=PersonaListResponse,
            temperature=_TEMPERATURE_CREATIVE,
            max_tokens=4000,
            label="PersonaGenerator.generate",
        )
        personas = parsed.personas if parsed is not None else self._parse_personas(raw or "[]")

        if len(personas) < num_personas:
            logger.warning(
                "PersonaGenerator: requested %d personas but only %d were successfully parsed",
                num_personas,
                len(personas),
            )
        return personas

    async def generate_with_coverage(
        self,
        *,
        agent_description: str,
        context: str = "",
        num_personas: int = 8,
        edge_case_percentage: float = 0.2,
    ) -> list[Persona]:
        """Generate personas with guaranteed trait coverage."""
        styles = ["formal", "casual", "terse", "verbose"]
        trait_targets = [
            {
                "patience": 0.1,
                "assertiveness": 0.1,
                "politeness": 0.1,
                "technical_level": 0.1,
            },
            {
                "patience": 0.9,
                "assertiveness": 0.1,
                "politeness": 0.9,
                "technical_level": 0.9,
            },
            {
                "patience": 0.1,
                "assertiveness": 0.9,
                "politeness": 0.1,
                "technical_level": 0.5,
            },
            {
                "patience": 0.5,
                "assertiveness": 0.9,
                "politeness": 0.9,
                "technical_level": 0.1,
            },
            {
                "patience": 0.5,
                "assertiveness": 0.5,
                "politeness": 0.5,
                "technical_level": 0.5,
            },
            {
                "patience": 0.3,
                "assertiveness": 0.7,
                "politeness": 0.6,
                "technical_level": 0.3,
            },
            {
                "patience": 0.7,
                "assertiveness": 0.3,
                "politeness": 0.8,
                "technical_level": 0.7,
            },
            {
                "patience": 0.2,
                "assertiveness": 0.8,
                "politeness": 0.3,
                "technical_level": 0.8,
            },
        ]
        num_edge_cases = int(num_personas * edge_case_percentage)

        coverage_lines = []
        for i in range(min(num_personas, 8)):
            target = trait_targets[i % len(trait_targets)]
            coverage_lines.append(
                f"- Persona {i + 1}: communication_style='{styles[i % len(styles)]}', "
                f"patience={target['patience']:.1f}, "
                f"assertiveness={target['assertiveness']:.1f}, "
                f"politeness={target['politeness']:.1f}, "
                f"technical_level={target['technical_level']:.1f}"
            )

        user_prompt = f"""Agent Description: {delimit(agent_description)}

Additional Context: {delimit(context or "None provided")}

Generate {num_personas} personas with EXACT trait values as specified below.
CRITICAL: Use the EXACT numeric values provided - do NOT adjust them to be more "balanced".

{chr(10).join(coverage_lines)}

IMPORTANT:
- Use the EXACT trait values shown above (e.g., if it says politeness=0.1, use 0.1, not 0.3 or 0.5)
- Low values (0.1-0.2) represent EXTREME traits - these are intentional, not mistakes
- Include {num_edge_cases} edge case/challenging personas
- Ensure traits span the full 0-1 range across all personas
- Create realistic backgrounds relevant to the agent's domain

Return ONLY a JSON array, no other text."""

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": _PERSONA_GENERATOR_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        parsed, raw = await generate_structured(
            self._client,
            model=self._model,
            messages=messages,
            response_format=PersonaListResponse,
            temperature=_TEMPERATURE_BALANCED,
            max_tokens=4000,
            label="PersonaGenerator.generate_with_coverage",
        )
        personas = parsed.personas if parsed is not None else self._parse_personas(raw or "[]")

        # Validate coverage and fill gaps
        personas = self._ensure_style_coverage(
            personas, [CommunicationStyle(s) for s in styles]
        )
        self._log_trait_coverage_gaps(personas)

        if len(personas) > num_personas:
            personas = personas[:num_personas]

        if len(personas) < num_personas:
            logger.warning(
                "PersonaGenerator: requested %d personas (with coverage) but only %d were successfully parsed",
                num_personas,
                len(personas),
            )
        return personas

    @staticmethod
    def _ensure_style_coverage(
        personas: list[Persona], required_styles: list[CommunicationStyle]
    ) -> list[Persona]:
        existing_styles = {p.communication_style for p in personas}
        missing_styles = [s for s in required_styles if s not in existing_styles]

        if missing_styles and personas:
            for i, style in enumerate(missing_styles):
                if i < len(personas):
                    personas[i] = personas[i].model_copy(
                        update={"communication_style": style}
                    )
                    logger.debug(
                        "Adjusted persona '%s' to style '%s' for coverage",
                        personas[i].name,
                        style.value,
                    )
        return personas

    @staticmethod
    def _log_trait_coverage_gaps(personas: list[Persona]) -> None:
        if not personas:
            return

        def has_low(values: list[float]) -> bool:
            return any(v <= 0.2 for v in values)

        def has_high(values: list[float]) -> bool:
            return any(v >= 0.8 for v in values)

        patience_vals = [p.patience for p in personas]
        assertive_vals = [p.assertiveness for p in personas]
        polite_vals = [p.politeness for p in personas]
        tech_vals = [p.technical_level for p in personas]

        gaps: list[str] = []
        if not has_low(patience_vals):
            gaps.append("low patience (<0.2)")
        if not has_high(patience_vals):
            gaps.append("high patience (>0.8)")
        if not has_low(assertive_vals):
            gaps.append("low assertiveness (<0.2)")
        if not has_high(assertive_vals):
            gaps.append("high assertiveness (>0.8)")
        if not has_low(polite_vals):
            gaps.append("low politeness (<0.2)")
        if not has_high(polite_vals):
            gaps.append("high politeness (>0.8)")
        if not has_low(tech_vals):
            gaps.append("low technical_level (<0.2)")
        if not has_high(tech_vals):
            gaps.append("high technical_level (>0.8)")

        if gaps:
            logger.debug("Trait coverage gaps: %s", ", ".join(gaps))
