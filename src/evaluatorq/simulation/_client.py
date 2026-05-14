"""Shared AsyncOpenAI client construction and response extraction for simulation components."""

from __future__ import annotations

import json as _json
import os
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from openai import AsyncOpenAI

    from evaluatorq.simulation.types import TokenUsage


def build_simulation_client(
    config_client: AsyncOpenAI | None = None,
    *,
    extra_api_key: str | None = None,
) -> tuple[AsyncOpenAI, bool]:
    """Build AsyncOpenAI client.

    Returns (client, owned) where owned=False means caller must not close it.

    Resolution order:
    1. ``config_client`` — injected client, used as-is (not owned).
    2. ``extra_api_key`` argument, treated as an ORQ key and routed through
       the Orq router.
    3. ``ORQ_API_KEY`` env var — routes through
       ``ORQ_BASE_URL/v2/router`` (default: ``https://api.orq.ai/v2/router``).
    4. ``OPENAI_API_KEY`` env var — uses the OpenAI SDK default base URL so
       traffic goes to OpenAI directly, not to the Orq router.
    """
    from openai import AsyncOpenAI

    if config_client is not None:
        return config_client, False

    orq_key = extra_api_key or os.environ.get("ORQ_API_KEY")
    resolved = orq_key or os.environ.get("OPENAI_API_KEY")

    if not resolved:
        raise ValueError(
            "No API key found. Set ORQ_API_KEY or OPENAI_API_KEY, "
            "or pass a pre-built client."
        )

    base_url: str | None = (
        f"{os.environ.get('ORQ_BASE_URL', 'https://api.orq.ai')}/v2/router"
        if orq_key
        else None
    )

    return AsyncOpenAI(base_url=base_url, api_key=resolved), True


def extract_responses_output(response: object) -> tuple[list[Any], TokenUsage | None]:
    """Extract output items and usage from a Responses API response object.

    Returns ``(output_messages, token_usage)``. ``token_usage`` is ``None``
    when the response carries no ``usage`` block — callers must distinguish
    "no usage reported" from "zero tokens used" so cost reports stay honest.

    Uses the correct ``item.type`` discriminator (``"message"`` for text,
    ``"function_call"`` for tool calls) rather than fragile ``hasattr`` checks.
    """
    from evaluatorq.contracts import TextOutputItem, ToolCallOutputItem
    from evaluatorq.simulation.types import TokenUsage

    items: list[Any] = []
    for item in getattr(response, "output", None) or []:
        item_type = getattr(item, "type", None)

        if item_type == "message":
            for part in getattr(item, "content", None) or []:
                part_type = getattr(part, "type", None)
                text = getattr(part, "text", None)
                if part_type == "output_text" and text:
                    items.append(
                        TextOutputItem(
                            type="output_text",
                            text=text,
                            annotations=[],
                            logprobs=[],
                        )
                    )

        elif item_type == "function_call":
            name = getattr(item, "name", None) or ""
            raw_args = getattr(item, "arguments", None) or "{}"
            call_id = (
                getattr(item, "call_id", None)
                or getattr(item, "id", None)
                or ""
            )
            result = getattr(item, "result", None)
            items.append(
                ToolCallOutputItem(
                    type="function_call",
                    name=str(name),
                    call_id=str(call_id),
                    arguments=raw_args if isinstance(raw_args, str) else _json.dumps(raw_args),
                    result=str(result) if result is not None else None,
                )
            )

        elif item_type == "reasoning":
            pass  # reasoning/thinking steps (o1/o3/o4-mini) intentionally excluded from output

        else:
            logger.warning("extract_responses_output: skipping unknown item type={!r}", item_type)

    usage_obj = getattr(response, "usage", None)
    if usage_obj is None:
        logger.warning(
            "extract_responses_output: response.usage is None; returning None "
            "so cost reports do not record fake-zero usage for billed calls"
        )
        return items, None
    input_toks = int(getattr(usage_obj, "input_tokens", 0) or 0)
    output_toks = int(getattr(usage_obj, "output_tokens", 0) or 0)
    usage = TokenUsage(
        prompt_tokens=input_toks,
        completion_tokens=output_toks,
        total_tokens=input_toks + output_toks,
    )

    return items, usage


__all__ = ["build_simulation_client", "extract_responses_output"]
