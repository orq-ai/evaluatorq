"""Redact untrusted trace text with the Orq PII endpoint before it reaches an external classifier.

Shell commands, tool arguments and command output can carry API keys, tokens, passwords and personal data. A hand-rolled
credential regex misses most of them, so every caller that sends such text to a model goes through `redact_texts`, which
asks the Orq `POST /v2/pii/redact` endpoint and returns the text with placeholders such as `<API_KEY_1>` in place of the
secrets.

The helpers fail closed. Text the endpoint could not redact comes back as `None`, never as the raw text, and the caller
leaves it out of the request it was going to build. Every such failure logs a warning that names its cause.

Retry is the SDK's own (`RetryConfig` on the request); there is no `with_retry` around it.
"""

from __future__ import annotations

import asyncio
import re
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from loguru import logger

from evaluatorq.common.llm_client import client_routes_through_orq, resolve_results_base_url
from evaluatorq.common.orq_client import close_orq_client, resolve_orq_client

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Sequence

    from openai import AsyncOpenAI
    from orq_ai_sdk import Orq

# Several texts travel in one request, joined by this line. The redacted reply is split on it again, and a reply that
# splits into another number of parts is treated as a failed redaction.
SEPARATOR = '\n\u241e\u241e\u241e\n'
# Characters per request. The endpoint takes about 15 s per million characters, so one request stays near a second.
BATCH_CHARS = 50_000
# Redaction requests in flight at once for one `redact_texts` call.
_CONCURRENCY = 4
_TIMEOUT_MS = 30_000
# Characters kept from each end of a text too long to redact whole; see `redact_cut`.
EDGE_WINDOW = 8_000
_PLACEHOLDER = re.compile(r'<([A-Z][A-Z0-9_]*)_\d+>')


def normalize_placeholders(text: str) -> str:
    """Drop the numbering of redaction placeholders: `<UUID_3>` becomes `<UUID>`.

    Two texts that differ only in which id, email, URL or key they carry normalize to the same string, so a caller can
    ask one question for all of them.
    """
    return _PLACEHOLDER.sub(r'<\1>', text)


@asynccontextmanager
async def redaction_client(client: AsyncOpenAI | None) -> AsyncIterator[Orq | None]:
    """Yield an Orq client for redaction built from the LLM client's credentials, or `None` when it cannot be built.

    The LLM client has to route through the Orq router: its API key and host then address the same Orq deployment that
    serves `/v2/pii/redact`. A client that does not (direct OpenAI, none) yields `None`, and so does a missing SDK. The
    Orq client is closed when the block exits.

    Yields:
        The Orq client, or `None` when none can be built from `client`.
    """
    api_key = getattr(client, 'api_key', None)
    if not client_routes_through_orq(client) or not isinstance(api_key, str) or not api_key:
        yield None
        return
    try:
        orq = resolve_orq_client(api_key=api_key, base_url=resolve_results_base_url(client))
    except (ImportError, ValueError) as exc:
        logger.warning('Could not build an Orq client for PII redaction: {}', exc)
        yield None
        return
    try:
        yield orq
    finally:
        await close_orq_client(orq)


async def redact_texts(texts: Sequence[str], *, orq: Orq | None) -> list[str | None]:
    """Redact every text with the Orq PII endpoint; the result at index `i` is `texts[i]` redacted.

    Texts are deduplicated and packed into requests of at most `BATCH_CHARS` characters (a text alone over that, or one
    containing the separator, gets a request of its own), at most `_CONCURRENCY` in flight. A text that is empty or only
    whitespace is returned as is. A text whose request failed (exception, non-200, a reply that does not split back
    into the parts sent) comes back as `None`, and so does every text when `orq` is `None`; each failure logs a warning
    naming its cause. Raw text is never returned in place of a redaction.

    Retry is the SDK's: each request carries a `RetryConfig` for rate limits, 5xx and connection errors.
    """
    redacted: dict[str, str | None] = {}
    pending: list[str] = []
    for text in dict.fromkeys(texts):
        if text.strip():
            pending.append(text)
        else:
            redacted[text] = text
    if pending and orq is None:
        logger.warning(
            'PII redaction is unavailable (no Orq client); withholding {} text(s) from the classifier', len(pending)
        )
        redacted.update(dict.fromkeys(pending))
    elif pending:
        semaphore = asyncio.Semaphore(_CONCURRENCY)
        batches = _batches(pending)
        replies = await asyncio.gather(*(_redact_batch(orq, batch, semaphore) for batch in batches))
        for batch, reply in zip(batches, replies, strict=True):
            redacted.update(zip(batch, reply if reply is not None else [None] * len(batch), strict=True))
    return [redacted[text] for text in texts]


async def redact_cut(
    texts: Sequence[str],
    *,
    keep: int,
    join: Callable[[str, str, int], str],
    slack: int = 0,
    orq: Orq | None,
) -> list[str | None]:
    """Redact each text, then cut it to its first and last `keep` characters; `None` where redaction failed.

    A text of at most `2 * keep + slack` characters after redaction is returned whole. A longer one is cut with
    `join(head, tail, omitted)`, where `omitted` counts the characters between them.

    To bound the cost of a very long text, only its first and last `EDGE_WINDOW` characters are redacted, as two
    separate parts, once the text is longer than twice that. The cut is taken from those redacted parts, so a secret
    would have to be over `EDGE_WINDOW - keep` characters long to straddle the cut and leave a readable fragment.
    """
    if keep >= EDGE_WINDOW:
        raise ValueError(f'keep must be below {EDGE_WINDOW}, got {keep}')
    parts: list[str] = []
    spans: list[tuple[int, int]] = []  # per text: index of its first part and how many parts it has
    for text in texts:
        windows = [text] if len(text) <= 2 * EDGE_WINDOW else [text[:EDGE_WINDOW], text[-EDGE_WINDOW:]]
        spans.append((len(parts), len(windows)))
        parts.extend(windows)
    redacted = await redact_texts(parts, orq=orq)
    cut: list[str | None] = []
    for text, (start, count) in zip(texts, spans, strict=True):
        pieces = redacted[start : start + count]
        if any(piece is None for piece in pieces):
            cut.append(None)
        elif count == 1:
            whole = pieces[0] or ''
            cut.append(
                whole if len(whole) <= 2 * keep + slack else join(whole[:keep], whole[-keep:], len(whole) - 2 * keep)
            )
        else:
            head, tail = pieces[0] or '', pieces[1] or ''
            cut.append(join(head[:keep], tail[-keep:], len(text) - 2 * keep))
    return cut


def _batches(texts: Sequence[str]) -> list[list[str]]:
    """Group texts into requests of at most `BATCH_CHARS`; a text that cannot be split back out travels alone."""
    batches: list[list[str]] = []
    current: list[str] = []
    used = 0
    for text in texts:
        if SEPARATOR in text or len(text) > BATCH_CHARS:
            batches.append([text])
            continue
        if current and used + len(SEPARATOR) + len(text) > BATCH_CHARS:
            batches.append(current)
            current, used = [], 0
        used += len(text) + (len(SEPARATOR) if current else 0)
        current.append(text)
    if current:
        batches.append(current)
    return batches


def _retry_config() -> object:
    from orq_ai_sdk.utils.retries import BackoffStrategy, RetryConfig

    return RetryConfig(
        strategy='backoff',
        backoff=BackoffStrategy(initial_interval=500, max_interval=5_000, exponent=2, max_elapsed_time=60_000),
        retry_connection_errors=True,
    )


async def _redact_batch(orq: Orq | None, batch: list[str], semaphore: asyncio.Semaphore) -> list[str] | None:
    """One request for a batch; its parts redacted in order, or `None` (logged) when the request failed."""
    if orq is None:
        return None
    try:
        async with semaphore:
            response = await orq.pii.redact_async(
                text=SEPARATOR.join(batch), retries=_retry_config(), timeout_ms=_TIMEOUT_MS
            )
    except Exception as exc:  # noqa: BLE001 - one batch failing must not lose the others
        # The exception text is left out: an API error body can quote the request it rejected.
        status = getattr(exc, 'status_code', None)
        return _withheld(batch, f'{type(exc).__name__}{f" (HTTP {status})" if status else ""}')
    text = getattr(response, 'redacted_text', None)
    if not isinstance(text, str):
        return _withheld(batch, f'the reply carried {type(text).__name__} instead of redacted text')
    parts = text.split(SEPARATOR) if len(batch) > 1 else [text]
    if len(parts) != len(batch):
        return _withheld(batch, f'the redacted text split into {len(parts)} parts, expected {len(batch)}')
    return parts


def _withheld(batch: list[str], cause: str) -> None:
    logger.warning(
        'PII redaction failed for a batch of {} text(s), withholding them from the classifier: {}', len(batch), cause
    )
