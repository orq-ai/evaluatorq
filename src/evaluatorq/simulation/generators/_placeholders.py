"""Keep a generator's real items when a model answers a JSON-schema response format with placeholders.

Claude Sonnet 5.5 and Opus 5.5 often answer a list schema with one schema-valid item whose text is empty, ``x`` or
``placeholder``. Asking the same way again mostly returns another, so the re-ask goes through a forced tool call.
"""

from __future__ import annotations

import contextlib
import json
import logging
from typing import TYPE_CHECKING, Any, TypeVar

from evaluatorq.common.structured_output import usage_from_exception

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from evaluatorq.common.structured_output import StructuredResult
    from evaluatorq.contracts import TokenUsage

logger = logging.getLogger(__name__)

MIN_TEXT_CHARS = 3

Item = TypeVar('Item')


def is_stub_item(name: str, body: str) -> bool:
    """Whether either text is blank, or both are under `MIN_TEXT_CHARS` (a persona called "Li" is real)."""
    name, body = name.strip(), body.strip()
    return not name or not body or (len(name) < MIN_TEXT_CHARS and len(body) < MIN_TEXT_CHARS)


def too_few(returned: int, expected: int) -> bool:
    """Whether a non-empty list has fewer than half the items asked for; one short is a miscount, not a stub."""
    return 0 < returned < (expected + 1) // 2


async def generate_real_items(
    call: Callable[..., Awaitable[StructuredResult[Any]]],
    *,
    items_of: Callable[[Any], list[Item]],
    texts_of: Callable[[Item], tuple[str, str]],
    salvage: Callable[[str], list[Item]],
    expected: int,
    label: str,
    record_usage: Callable[[TokenUsage | None], None],
) -> list[Item]:
    """Run ``call`` (a partial `generate_structured`) and return its real items, re-asking once on placeholders.

    A parsed answer holding a stub, or too few items, is asked again with ``skip_schema_rungs=True``; the attempt with
    more real items wins, the first on a tie. ``salvage`` parses raw text when an attempt did not validate. Every
    attempt's usage goes to ``record_usage``, including a second attempt that raised.
    """

    def real(items: list[Item]) -> list[Item]:
        kept = [item for item in items if not is_stub_item(*texts_of(item))]
        if len(kept) < len(items):
            logger.warning('%s: dropped %d placeholder item(s)', label, len(items) - len(kept))
        return kept

    first = await call(skip_schema_rungs=False)
    record_usage(first.usage)
    if first.parsed is None:
        return real(salvage(first.raw or '[]'))
    answered = items_of(first.parsed)
    items = real(answered)
    if len(items) == len(answered) and not too_few(len(items), expected):
        return items

    logger.warning(
        '%s: %d real item(s) for %d asked; asking once more through a tool call', label, len(items), expected
    )
    try:
        second = await call(skip_schema_rungs=True)
    except Exception as exc:  # noqa: BLE001 - the optional re-ask never costs the first answer
        logger.warning('%s: the second call failed (%s); keeping the first answer', label, exc)
        record_usage(usage_from_exception(exc))
        return items
    record_usage(second.usage)
    retry: list[Item] = []
    if second.parsed is not None:
        retry = real(items_of(second.parsed))
    elif second.raw:
        with contextlib.suppress(json.JSONDecodeError):
            retry = real(salvage(second.raw))
    return retry if len(retry) > len(items) else items
