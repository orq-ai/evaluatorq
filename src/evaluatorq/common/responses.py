"""Shared Responses API metadata, schema, refusal, and parsing helpers."""

from __future__ import annotations

import logging
from importlib import import_module
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable

    from pydantic import BaseModel

logger = logging.getLogger(__name__)


def responses_text_config(response_model: type[BaseModel]) -> dict[str, Any]:
    """Build the strict ``text.format`` config used by ``responses.create``."""
    try:
        to_strict_json_schema = import_module('openai.lib._pydantic').to_strict_json_schema
    except (ImportError, AttributeError):
        logger.warning('OpenAI strict-schema helper unavailable; using the Pydantic JSON schema fallback')
        schema = response_model.model_json_schema()
    else:
        schema = to_strict_json_schema(response_model)
    return {
        'format': {
            'type': 'json_schema',
            'strict': True,
            'name': response_model.__name__,
            'schema': schema,
        }
    }


def first_responses_refusal(response: Any) -> str | None:
    """Return the first refusal in a Responses output, if present."""
    for item in getattr(response, 'output', None) or []:
        for part in getattr(item, 'content', None) or []:
            if getattr(part, 'type', None) == 'refusal':
                return getattr(part, 'refusal', '') or ''
    return None


def responses_stop_reason(response: Any) -> str | None:
    """Return a normalized provider completion reason.

    OpenAI Responses uses ``incomplete_details.reason=max_output_tokens``;
    compatible routers may expose the equivalent directly as ``length``.
    Nested max-token metadata wins over a conflicting non-length top-level
    value because it is the more specific truncation signal.
    """
    nested_reason = getattr(getattr(response, 'incomplete_details', None), 'reason', None)
    if nested_reason == 'max_output_tokens':
        return 'length'
    stop_reason = getattr(response, 'stop_reason', None)
    if isinstance(stop_reason, str):
        return 'length' if stop_reason == 'max_output_tokens' else stop_reason
    return nested_reason if isinstance(nested_reason, str) else None


def parse_responses_response(
    response: Any,
    response_model: type[BaseModel],
    *,
    input_tools: Iterable[Any] | None = None,
) -> Any:
    """Apply the SDK's per-output-item Responses parser to a raw response."""
    try:
        parse_response = import_module('openai.lib._parsing._responses').parse_response
    except (ImportError, AttributeError):
        logger.warning('OpenAI Responses parser unavailable; validating output_text directly')
        raw_content = getattr(response, 'output_text', '') or ''
        parsed = response_model.model_validate_json(raw_content) if raw_content else None
        return SimpleNamespace(output_parsed=parsed)
    return parse_response(text_format=response_model, input_tools=input_tools or (), response=response)
