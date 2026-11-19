"""Keep a generator's real items when a model answers a JSON-schema response format with placeholders.

Claude Sonnet 5.5 and Opus 5.5 often answer a list schema with one schema-valid item whose text is empty, ``x`` or
``placeholder``. Asking the same way again mostly returns another, so the re-ask goes through a forced tool call.
"""

from __future__ import annotations

import contextlib
import json
import logging
from typing import TYPE_CHECKING, TypeVar

from pydantic import BaseModel, ValidationError

from evaluatorq.common.llm_errors import BAD_REQUEST_ERRORS, CONNECTION_ERRORS, STATUS_ERRORS
from evaluatorq.common.structured_output import StructuredGenerationError, usage_from_exception

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from evaluatorq.common.structured_output import StructuredResult
    from evaluatorq.contracts import TokenUsage

logger = logging.getLogger(__name__)

MIN_TEXT_CHARS = 3

# What a failed re-ask may raise and still leave the first answer standing; anything else is a bug and propagates.
_REASK_ERRORS: tuple[type[BaseException], ...] = (
    *STATUS_ERRORS,
    *CONNECTION_ERRORS,
    *BAD_REQUEST_ERRORS,
    StructuredGenerationError,
    json.JSONDecodeError,
    ValidationError,
)

Answer = TypeVar('Answer', bound=BaseModel)
Item = TypeVar('Item')


def is_stub_item(*, name: str, body: str) -> bool:
    """Whether either text is blank, or both are under `MIN_TEXT_CHARS` (a persona called "Li" is real)."""
    name, body = name.strip(), body.strip()
    return not name or not body or (len(name) < MIN_TEXT_CHARS and len(body) < MIN_TEXT_CHARS)


def too_few(*, returned: int, expected: int) -> bool:
    """Whether a non-empty list has fewer than half the items asked for; one short is a miscount, not a stub."""
    return 0 < returned < (expected + 1) // 2


async def generate_real_items(
    *,
    call: Callable[..., Awaitable[StructuredResult[Answer]]],
    items_of: Callable[[Answer], list[Item]],
    texts_of: Callable[[Item], tuple[str, str]],
    salvage: Callable[[str], list[Item]],
    expected: int,
    label: str,
    record_usage: Callable[[TokenUsage | None], None],
) -> list[Item]:
    """Run ``call`` (a partial `generate_structured`) and return its real items, re-asking once on placeholders.

    An answer holding a stub, or too few items, is asked again with ``skip_schema_rungs=True`` under the label
    ``{label}.reask``; the attempt with more real items wins, the first on a tie. ``salvage`` parses raw text when an
    attempt did not validate: on the first attempt a `json.JSONDecodeError` it raises propagates (the generators catch
    it and return ``[]``), on the re-ask it is suppressed. Every attempt's usage goes to ``record_usage``, including
    one that raised.
    """
    reask_label = f'{label}.reask'

    def real(items: list[Item], *, attempt: str) -> list[Item]:
        kept: list[Item] = []
        for item in items:
            name, body = texts_of(item)
            if not is_stub_item(name=name, body=body):
                kept.append(item)
        if len(kept) < len(items):
            logger.warning('%s: dropped %d placeholder item(s)', attempt, len(items) - len(kept))
        return kept

    try:
        first = await call(skip_schema_rungs=False, label=label)
    except Exception as exc:
        record_usage(usage_from_exception(exc))
        raise
    record_usage(first.usage)
    answered = items_of(first.parsed) if first.parsed is not None else salvage(first.raw or '[]')
    items = real(answered, attempt=label)
    if len(items) == len(answered) and not too_few(returned=len(items), expected=expected):
        return items

    logger.warning(
        '%s: %d real item(s) for %d asked; asking once more through a tool call', label, len(items), expected
    )
    try:
        second = await call(skip_schema_rungs=True, label=reask_label)
    except _REASK_ERRORS as exc:
        logger.warning('%s: the call failed (%s); keeping the first answer', reask_label, exc)
        record_usage(usage_from_exception(exc))
        return items
    record_usage(second.usage)
    retry: list[Item] = []
    if second.parsed is not None:
        retry = real(items_of(second.parsed), attempt=reask_label)
    elif second.raw:
        with contextlib.suppress(json.JSONDecodeError):
            retry = real(salvage(second.raw), attempt=reask_label)
    return retry if len(retry) > len(items) else items
