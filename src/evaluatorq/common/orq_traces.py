"""One reader for Orq's ``/v3/traces`` routes, shared by the trace importer and the trace finder.

Both surfaces read the same routes through the Orq SDK client. The SDK's
generated models drop most of what those routes return (span attributes, a
trace summary's attributes, a span summary's ``level`` and ``has_children``,
conversation items), so every call here also keeps the raw JSON body through an
SDK after-success hook. Paging goes through `PageCursor`, which owns the page cap
and the repeated-token guard; each caller decides what a paging failure means for
it.
"""

from __future__ import annotations

import asyncio
import json
import re
import weakref
from collections import defaultdict, deque
from contextlib import suppress
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from orq_ai_sdk import Orq

T = TypeVar('T')

SPAN_PAGE_SIZE = 200
MAX_SPAN_PAGES = 10
MAX_SPANS_PER_TRACE = SPAN_PAGE_SIZE * MAX_SPAN_PAGES
_MAX_CAPTURED_RESPONSES = 128
_CAPTURE_REGISTRATION_ATTR = '_evaluatorq_trace_finder_capture_registration'
CAPTURE_REQUEST: ContextVar[object | None] = ContextVar('trace_finder_capture_request', default=None)
"""Marks one SDK call so its raw body is popped by that call, not by a concurrent one."""

_API_VERSION = re.compile(r'^/v\d+(?=/)')


class TracePageError(Exception):
    """A paged trace route could not be read to its end."""


class PageTokenError(TracePageError):
    """The route said more pages follow but gave no usable token for them."""

    def __init__(self, message: str, *, token: str | None) -> None:
        super().__init__(message)
        self.token = token


class PageLimitError(TracePageError):
    """The route needed more pages (or items) than the caller allows."""


class PageCursor:
    """The page token for the next request, with one page cap and one repeated-token guard.

    Call `start_page` before each request and `advance` after it. A server stuck
    on one token, or one that promises more pages without a token, raises
    `PageTokenError` instead of being paged forever.
    """

    def __init__(self, *, max_pages: int) -> None:
        self.max_pages = max_pages
        self.pages = 0
        self.token: str | None = None
        self._seen: set[str] = set()

    def start_page(self) -> None:
        """Count one request. Raises `PageLimitError` once ``max_pages`` were already requested."""
        if self.pages >= self.max_pages:
            raise PageLimitError(f'exceeded {self.max_pages} pages')
        self.pages += 1

    def advance(self, *, has_more: Any, next_token: Any) -> bool:
        """Take the next token from a page. Returns False on the last page.

        Raises:
            PageTokenError: more pages follow but the token is missing or was already used.
        """
        if not has_more:
            return False
        token = str(next_token) if next_token else None
        if token is None:
            raise PageTokenError('reports more pages but returns no page token', token=None)
        if token in self._seen:
            raise PageTokenError(f'repeated page token {token!r}', token=token)
        self._seen.add(token)
        self.token = token
        return True


def _field(value: Any, name: str) -> Any:
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


async def collect_span_summaries(fetch_page: Callable[[str | None], Awaitable[Any]], *, trace_id: str) -> list[Any]:
    """Every span summary of one trace, paged through ``fetch_page(page_token)``.

    The caller owns the request (its client, timeout, semaphore and deadline);
    this owns the cap of `MAX_SPAN_PAGES` pages and `MAX_SPANS_PER_TRACE` spans,
    so a trace's span list is either complete or an error, never truncated.

    Raises:
        TracePageError: the list needs more pages or spans than the cap, or its page token is missing or repeated.
    """
    cursor = PageCursor(max_pages=MAX_SPAN_PAGES)
    spans: list[Any] = []
    while True:
        try:
            cursor.start_page()
        except PageLimitError as exc:
            raise PageLimitError(f'span pagination exceeded {MAX_SPAN_PAGES} pages for trace {trace_id!r}') from exc
        page = await fetch_page(cursor.token)
        data = _field(page, 'data')
        spans.extend(data if isinstance(data, (list, tuple)) else [])
        if len(spans) > MAX_SPANS_PER_TRACE:
            raise PageLimitError(f'span pagination exceeded {MAX_SPANS_PER_TRACE} spans for trace {trace_id!r}')
        if not cursor.advance(has_more=_field(page, 'has_more'), next_token=_field(page, 'next_page_token')):
            return spans


class RawResponseCapture:
    """Retain raw trace payloads whose generated SDK models discard fields."""

    _OPERATIONS = frozenset({
        'TracesGetConversation',
        'TracesGetSpan',
        'TracesListSpans',
        'TracesQueryOql',
        'TracesSearch',
    })

    def __init__(self) -> None:
        self._responses: dict[tuple[str, object | None], deque[Any]] = defaultdict(
            lambda: deque(maxlen=_MAX_CAPTURED_RESPONSES)
        )

    @staticmethod
    def key(path: str) -> str:
        """Capture key for a request path: the API version segment is dropped so SDK upgrades keep matching."""
        return _API_VERSION.sub('', path, count=1)

    def after_success(self, hook_context: Any, response: Any) -> Any:
        """Capture supported JSON responses while leaving the SDK response unchanged."""
        if getattr(hook_context, 'operation_id', None) not in self._OPERATIONS:
            return response
        try:
            payload = response.json()
        except (json.JSONDecodeError, UnicodeDecodeError, AttributeError):
            return response
        path = getattr(getattr(getattr(response, 'request', None), 'url', None), 'path', None)
        if payload is not None and isinstance(path, str):
            self._responses[self.key(path), CAPTURE_REQUEST.get()].append(payload)
        return response

    def pop(self, path: str) -> Any:
        """Return the oldest captured response for ``path`` (any API version), or None."""
        key = (self.key(path), CAPTURE_REQUEST.get())
        responses = self._responses.get(key)
        if responses:
            payload = responses.popleft()
            if not responses:
                del self._responses[key]
            return payload
        return None

    def clear(self) -> None:
        """Discard all captured responses."""
        self._responses.clear()


class CaptureRegistration:
    """Share raw capture state and per-event-loop locks across readers of one client.

    An asyncio lock is tied to the event loop that first waits on it, so locks
    are kept per running loop while the response queue remains shared per SDK
    client.
    """

    def __init__(self) -> None:
        self.capture = RawResponseCapture()
        self.sources = 0
        self.registered = False
        self.locks: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Lock] = weakref.WeakKeyDictionary()

    def lock_for(self, loop: asyncio.AbstractEventLoop) -> asyncio.Lock:
        """Return the capture lock associated with ``loop``."""
        lock = self.locks.get(loop)
        if lock is None:
            lock = asyncio.Lock()
            self.locks[loop] = lock
        return lock


class OrqTraceReader:
    """Raw-JSON access to one Orq SDK client's trace routes.

    Registers one shared after-success hook per client and releases it when the
    last reader on that client closes. A client without hooks (a test double)
    gets no registration; `raw_call` then returns no raw body and the caller
    falls back to the SDK model.
    """

    def __init__(self, client: Orq) -> None:
        self.client = client
        self.capture = RawResponseCapture()
        self.registration: CaptureRegistration | None = None
        self._hooks: Any | None = None
        self._closed = False
        configuration = getattr(client, 'sdk_configuration', None)
        hooks = getattr(configuration, '_hooks', None)
        register = getattr(hooks, 'register_after_success_hook', None)
        if not callable(register):
            return
        registration = getattr(hooks, _CAPTURE_REGISTRATION_ATTR, None)
        if not isinstance(registration, CaptureRegistration):
            registration = CaptureRegistration()
            try:
                setattr(hooks, _CAPTURE_REGISTRATION_ATTR, registration)
            except (AttributeError, TypeError):
                return
        if not registration.registered:
            register(registration.capture)
            registration.registered = True
        registration.sources += 1
        self.capture = registration.capture
        self._hooks = hooks
        self.registration = registration

    def close(self) -> None:
        """Release this reader's capture registration without closing the client."""
        if self._closed:
            return
        self._closed = True
        registration = self.registration
        if registration is None:
            self.capture.clear()
            return
        registration.sources = max(0, registration.sources - 1)
        if registration.sources:
            return
        self.capture.clear()
        unregister = getattr(self._hooks, 'unregister_after_success_hook', None)
        if callable(unregister):
            unregister(self.capture)
            registration.registered = False
            with suppress(AttributeError):
                delattr(self._hooks, _CAPTURE_REGISTRATION_ATTR)

    async def raw_call(
        self,
        call: Callable[[], Awaitable[T]],
        *,
        path: str,
        serialize: bool = False,
        raw_over_model: bool = False,
    ) -> tuple[T | None, Any]:
        """Run one SDK call and return ``(sdk_result, raw_json)``.

        ``raw_json`` is None when nothing was captured, as on a client without hooks. ``serialize`` holds the
        per-loop capture lock for the call, for routes whose concurrent responses
        must not interleave. With ``raw_over_model``, a body the SDK's generated
        model rejects but the hook already captured comes back as
        ``(None, raw_json)`` instead of raising, for a caller that reads only the
        raw body and checks its shape itself. The captured body is always popped,
        so a failed call leaves nothing behind in the shared capture.
        """
        registration = self.registration
        if registration is None:
            # Nothing is registered, so the capture only holds what was put there by hand.
            return await call(), self.capture.pop(path)
        marker = CAPTURE_REQUEST.set(object())
        try:
            if serialize:
                async with registration.lock_for(asyncio.get_running_loop()):
                    return await self._call_and_pop(call, path=path, raw_over_model=raw_over_model)
            return await self._call_and_pop(call, path=path, raw_over_model=raw_over_model)
        finally:
            CAPTURE_REQUEST.reset(marker)

    async def _call_and_pop(
        self, call: Callable[[], Awaitable[T]], *, path: str, raw_over_model: bool
    ) -> tuple[T | None, Any]:
        try:
            result = await call()
        except Exception:
            raw = self.capture.pop(path)
            if raw is None or not raw_over_model:
                raise
            return None, raw
        return result, self.capture.pop(path)


def plain_json(value: Any) -> Any:
    """An SDK model (or a list of them) as the JSON the API sent, keyed by alias."""
    if isinstance(value, (list, tuple)):
        return [plain_json(item) for item in value]
    dump = getattr(value, 'model_dump', None)
    if callable(dump):
        return dump(by_alias=True, mode='json', exclude_none=True)
    return value


__all__ = [
    'CAPTURE_REQUEST',
    'MAX_SPANS_PER_TRACE',
    'MAX_SPAN_PAGES',
    'SPAN_PAGE_SIZE',
    'CaptureRegistration',
    'OrqTraceReader',
    'PageCursor',
    'PageLimitError',
    'PageTokenError',
    'RawResponseCapture',
    'TracePageError',
    'collect_span_summaries',
    'plain_json',
]
