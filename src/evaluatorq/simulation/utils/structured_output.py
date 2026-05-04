"""Structured output helper with json_object fallback.

Tries ``client.chat.completions.parse()`` first (schema-enforced structured
output).  When the model doesn't support it the API returns 400; we fall back
to ``response_format={"type": "json_object"}`` and return the raw content for
manual parsing.
"""

from __future__ import annotations

import logging
from typing import Any, TypeVar, cast

from openai import APIStatusError, AsyncOpenAI
from pydantic import BaseModel

from evaluatorq.simulation.utils.retry import with_retry

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


async def generate_structured(
    client: AsyncOpenAI,
    *,
    model: str,
    messages: list[dict[str, Any]],
    response_format: type[T],
    temperature: float,
    max_tokens: int,
    label: str,
) -> tuple[T | None, str]:
    """Generate a chat completion with structured output, falling back to json_object.

    Returns ``(parsed_model, "")`` when structured output succeeds, or
    ``(None, raw_content)`` when the model doesn't support it and we fall back
    to json_object mode.
    """

    # Cast once — the OpenAI SDK accepts dict literals at runtime; the
    # TypedDict union just doesn't type-narrow from dict[str, Any].
    typed_messages = cast("Any", messages)

    # 1. Try structured output via parse()
    try:
        response = await with_retry(
            lambda: client.chat.completions.parse(
                model=model,
                messages=typed_messages,
                response_format=response_format,
                temperature=temperature,
                max_tokens=max_tokens,
            ),
            label=label,
        )
        message = response.choices[0].message
        refusal = getattr(message, "refusal", None)
        if refusal:
            raise RuntimeError(f"{label}: model refused to generate: {refusal}")
        parsed = message.parsed
        if parsed is not None:
            return parsed, ""
        logger.debug("%s: parse() returned None, falling back to json_object", label)
    except APIStatusError as e:
        if e.status_code != 400:
            raise
        # Only fall back if this looks like a schema-support issue
        err_body = str(getattr(e, "body", None) or getattr(e, "message", "") or "").lower()
        schema_keywords = ("structured", "response_format", "json_schema", "not supported")
        if not any(kw in err_body for kw in schema_keywords):
            raise
        logger.debug(
            "%s: structured output not supported by model, falling back to json_object",
            label,
        )

    # 2. Fallback: json_object mode
    fallback_response = await with_retry(
        lambda: client.chat.completions.create(
            model=model,
            messages=typed_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        ),
        label=f"{label} (fallback)",
    )
    content = fallback_response.choices[0].message.content if fallback_response.choices else ""
    return None, content or ""
