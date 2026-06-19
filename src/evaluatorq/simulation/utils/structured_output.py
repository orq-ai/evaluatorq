"""Structured output helper with json_object fallback.

Tries ``client.chat.completions.parse()`` first (schema-enforced structured
output).  When the model doesn't support it the API returns 400; we fall back
to ``response_format={"type": "json_object"}`` and return the raw content for
manual parsing.
"""

from __future__ import annotations

import logging
from typing import Any, TypeVar, cast

from openai import APIStatusError, AsyncOpenAI, LengthFinishReasonError
from pydantic import BaseModel

from evaluatorq.common.retry import with_retry
from evaluatorq.common.tracing import get_trace_context_headers, record_llm_input, record_llm_response
from evaluatorq.simulation.tracing import with_llm_span

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

    async with with_llm_span(
        model=model,
        operation="chat",
        temperature=temperature,
        max_tokens=max_tokens,
        purpose=label,
    ) as span:
        record_llm_input(
            span,
            [{"role": str(m["role"]), "content": str(m["content"])} for m in messages],
        )
        trace_headers = await get_trace_context_headers()
        extra: dict[str, Any] = {"extra_headers": trace_headers} if trace_headers else {}

        # 1. Try structured output via parse()
        try:
            response = await with_retry(
                lambda: client.chat.completions.parse(
                    model=model,
                    messages=typed_messages,
                    response_format=response_format,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **extra,
                ),
                label=label,
            )
            record_llm_response(span, response)
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
            logger.warning(
                "%s: structured output not supported by model, falling back to json_object",
                label,
            )
            # TODO: annotate the active OTel span with
            #   orq.simulation.structured_output.fallback=true
            # once a get_current_span() helper is wired into this module.
            if span is not None:
                fallback = True
                span.set_attribute("orq.simulation.structured_output.fallback", fallback)
        except LengthFinishReasonError as exc:
            # Length-truncated structured output is unusable — the JSON is cut
            # off mid-string. Falling back to json_object would truncate at the
            # same budget, so fail loudly with an actionable message instead.
            logger.error("%s: structured output truncated at the token limit (max_tokens=%s)", label, max_tokens)
            raise RuntimeError(
                f"{label}: the model hit the token limit (max_tokens={max_tokens}) and the "
                f"structured output was truncated, so the result is unusable. Raise the budget "
                f"via EVALUATORQ_LLM_MAX_TOKENS and retry."
            ) from exc

        # 2. Fallback: json_object mode
        fallback_response = await with_retry(
            lambda: client.chat.completions.create(  # pyright: ignore[reportUnknownLambdaType]
                model=model,
                messages=typed_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
                **extra,
            ),
            label=f"{label} (fallback)",
        )
        record_llm_response(span, fallback_response)
        content = fallback_response.choices[0].message.content if fallback_response.choices else ""
        return None, content or ""
