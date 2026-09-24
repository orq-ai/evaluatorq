"""Bounded live OQL acquisition normalized into the trace finder snapshot contract."""

from __future__ import annotations

import asyncio
import json
import re
import weakref
from collections import defaultdict, deque
from collections.abc import Mapping
from contextlib import suppress
from contextvars import ContextVar
from datetime import datetime, timedelta, timezone
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from loguru import logger

from .facets import _project_names, project_labels
from .models import FacetSelection, NumericFilters, Snapshot, TraceRecord

if TYPE_CHECKING:
    from orq_ai_sdk import Orq

BASE_FILTER = 'operation not_in ("generate_content")'
FACET_OQL_FIELDS: Mapping[str, str] = MappingProxyType({
    'model': 'model',
    'provider': 'provider',
    'status': 'status',
    'product': 'product',
    'trace_type': 'attributes.orq.leading_span.span_type',
    'agent_name': 'agent_name',
    'tool_name': 'tool_name',
})
MAX_LIVE_TRACES = 5000
PAGE_SIZE = 200
MAX_SPAN_PAGES = 10
MAX_SPANS_PER_TRACE = PAGE_SIZE * MAX_SPAN_PAGES
SDK_TIMEOUT_MS = 30_000
DEFAULT_LOOKBACK = timedelta(days=7)
CONVERSATION_SPAN_TYPES = frozenset({
    'span.chat_completion',
    'span.completion',
    'span.deployment',
    'span.responses',
})
EVALUATOR_SPAN_TYPES = frozenset({'span.evaluation_engine', 'span.evaluator'})
_MAX_CAPTURED_RESPONSES = 128
_CAPTURE_REGISTRATION_ATTR = '_evaluatorq_trace_finder_capture_registration'
_CAPTURE_REQUEST: ContextVar[object | None] = ContextVar('trace_finder_capture_request', default=None)


class OrqSourceError(ValueError):
    """Live trace acquisition could not produce a valid snapshot."""


_API_VERSION = re.compile(r'^/v\d+(?=/)')


class _RawResponseCapture:
    """Retain raw trace payloads whose generated SDK models discard attributes."""

    _OPERATIONS = frozenset({'TracesGetSpan', 'TracesQueryOql'})

    def __init__(self) -> None:
        self._responses: dict[tuple[str, object | None], deque[dict[str, Any]]] = defaultdict(
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
        if isinstance(payload, dict) and isinstance(path, str):
            self._responses[self.key(path), _CAPTURE_REQUEST.get()].append(payload)
        return response

    def pop(self, path: str) -> dict[str, Any] | None:
        """Return the oldest captured response for ``path`` (any API version)."""

        key = (self.key(path), _CAPTURE_REQUEST.get())
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


class _CaptureRegistration:
    """Share raw capture state and per-event-loop locks across client sources.

    An asyncio lock is tied to the event loop that first waits on it, so locks
    are kept per running loop while the response queue remains shared per SDK
    client.
    """

    def __init__(self) -> None:
        self.capture = _RawResponseCapture()
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


class OrqTraceSource:
    """Load recent live traces once, returning the immutable local snapshot contract.

    This shared source is loop-affine: its capture lock serialises calls within
    one event loop only, so build one source per event loop.
    """

    def __init__(self, client: Orq, *, hydration_concurrency: int = 20, owns_client: bool = False) -> None:
        if hydration_concurrency < 1:
            raise ValueError('hydration_concurrency must be at least 1')
        self._client = client
        self._hydration_concurrency = hydration_concurrency
        self._owns_client = owns_client
        self._capture = _RawResponseCapture()
        self._hooks: Any | None = None
        self._registration: _CaptureRegistration | None = None
        self._closed = False
        configuration = getattr(client, 'sdk_configuration', None)
        hooks = getattr(configuration, '_hooks', None)
        register = getattr(hooks, 'register_after_success_hook', None)
        if callable(register):
            registration = getattr(hooks, _CAPTURE_REGISTRATION_ATTR, None)
            if not isinstance(registration, _CaptureRegistration):
                registration = _CaptureRegistration()
                try:
                    setattr(hooks, _CAPTURE_REGISTRATION_ATTR, registration)
                except (AttributeError, TypeError):
                    registration = None
            if registration is not None:
                if not registration.registered:
                    register(registration.capture)
                    registration.registered = True
                registration.sources += 1
                self._capture = registration.capture
                self._hooks = hooks
                self._registration = registration

    def close(self) -> None:
        """Release this source's capture registration without closing the caller's client."""

        if self._closed:
            return
        self._closed = True
        registration = self._registration
        if registration is None:
            self._capture.clear()
            return
        registration.sources = max(0, registration.sources - 1)
        if registration.sources:
            return
        self._capture.clear()
        unregister = getattr(self._hooks, 'unregister_after_success_hook', None)
        if callable(unregister):
            unregister(self._capture)
            registration.registered = False
            with suppress(AttributeError):
                delattr(self._hooks, _CAPTURE_REGISTRATION_ATTR)

    async def aclose(self) -> None:
        """Async alias for ``close`` for callers that manage async SDK resources."""

        self.close()

    async def load_async(
        self,
        start: datetime | None,
        end: datetime | None,
        limit: int,
        *,
        facets: FacetSelection,
        numeric: NumericFilters,
    ) -> Snapshot:
        """Load at most 5000 usable traces in deterministic newest-first order."""

        try:
            return await self._load_with_lifecycle(start, end, limit, facets, numeric)
        except OrqSourceError:
            raise
        except Exception as error:
            raise OrqSourceError(f'live Orq trace loading failed: {error}') from error

    async def _load_with_lifecycle(
        self,
        start: datetime | None,
        end: datetime | None,
        limit: int,
        facets: FacetSelection,
        numeric: NumericFilters,
    ) -> Snapshot:
        if self._owns_client:
            async with self._client:
                return await self._validated_load(start, end, limit, facets, numeric)
        return await self._validated_load(start, end, limit, facets, numeric)

    async def _validated_load(
        self,
        start: datetime | None,
        end: datetime | None,
        limit: int,
        facets: FacetSelection,
        numeric: NumericFilters,
    ) -> Snapshot:
        if limit < 1:
            raise OrqSourceError('limit must be at least 1')
        resolved_end = _aware_bound(end, 'end') if end is not None else datetime.now(timezone.utc)
        resolved_start = _aware_bound(start, 'start') if start is not None else resolved_end - DEFAULT_LOOKBACK
        if resolved_start > resolved_end:
            raise OrqSourceError('start must not be after end')
        return await self._load(resolved_start, resolved_end, min(limit, MAX_LIVE_TRACES), facets, numeric)

    async def _load(
        self,
        start: datetime,
        end: datetime,
        limit: int,
        facets: FacetSelection,
        numeric: NumericFilters,
    ) -> Snapshot:
        project_names = await _project_names(self._client)
        oql = build_oql(facets, numeric, project_names)
        semaphore = asyncio.Semaphore(self._hydration_concurrency)
        records: list[TraceRecord] = []
        seen_trace_ids: set[str] = set()
        used_tokens: set[str] = set()
        dropped_count = 0
        wrong_project_count = 0
        fallback_count = 0
        scanned_count = 0
        max_scanned = max(PAGE_SIZE, limit * 5)
        pages_scanned = 0
        max_pages = max(20, (max_scanned + PAGE_SIZE - 1) // PAGE_SIZE * 2)
        page_token: str | None = None

        while len(records) < limit:
            if scanned_count >= max_scanned or pages_scanned >= max_pages:
                logger.warning(
                    'stopped trace search after scanning {} summaries across {} pages for {} usable traces',
                    scanned_count,
                    pages_scanned,
                    limit,
                )
                break
            pages_scanned += 1
            if page_token is not None:
                if page_token in used_tokens:
                    raise OrqSourceError(f'repeated OQL page token {page_token!r}')
                used_tokens.add(page_token)
            page_limit = min(PAGE_SIZE, limit - len(records))
            registration = self._registration
            marker = _CAPTURE_REQUEST.set(object()) if registration is not None else None
            try:
                if registration is None:
                    response = await self._client.traces.query_async(
                        from_=start,
                        to=end,
                        oql=oql,
                        limit=page_limit,
                        page_token=page_token,
                        timeout_ms=SDK_TIMEOUT_MS,
                    )
                    raw_page = self._capture.pop('/traces/query')
                else:
                    async with registration.lock_for(asyncio.get_running_loop()):
                        response = await self._client.traces.query_async(
                            from_=start,
                            to=end,
                            oql=oql,
                            limit=page_limit,
                            page_token=page_token,
                            timeout_ms=SDK_TIMEOUT_MS,
                        )
                        raw_page = self._capture.pop('/traces/query')
            finally:
                if marker is not None:
                    _CAPTURE_REQUEST.reset(marker)
            search = _field(response, 'search') or response
            summaries = _as_list(_field(search, 'data'))
            scanned_count += len(summaries)
            raw_summaries = _raw_trace_summaries(raw_page)
            raw_by_id = {
                str(trace_id): payload
                for payload in raw_summaries
                if (trace_id := _field(payload, 'trace_id') or _field(payload, 'id'))
            }
            unique_summaries, rejected = _select_summaries(summaries, raw_by_id, seen_trace_ids, facets.project_id)
            wrong_project_count += rejected

            hydrated = await self._hydrate_page(unique_summaries, project_names, semaphore)
            fallback_count += sum(result[1] for result in hydrated)
            dropped_count += sum(record is None for record, _ in hydrated)
            records.extend(record for record, _ in hydrated if record is not None)
            if len(records) >= limit:
                break

            has_more = bool(_field(search, 'has_more'))
            next_token = _field(search, 'next_page_token')
            if not has_more:
                break
            if not next_token:
                raise OrqSourceError('OQL response reported more pages without a page token')
            if str(next_token) in used_tokens:
                raise OrqSourceError(f'repeated OQL page token {next_token!r}')
            page_token = str(next_token)

        if wrong_project_count:
            logger.warning(
                'discarded {} trace summary(s) outside selected project {} despite the OQL filter',
                wrong_project_count,
                facets.project_id,
            )
            if not records:
                raise OrqSourceError('Orq returned only traces outside the selected project. No traces were loaded.')
        if fallback_count:
            logger.warning(
                'used SDK-model fallback for {} trace response(s) because raw-response capture was unavailable; '
                'generated SDK models may omit conversation messages, agent_name, tool_name, total_tokens, and '
                'duration_ms',
                fallback_count,
            )
        if dropped_count:
            logger.warning('dropped {} trace(s) because messages or timestamps were unusable', dropped_count)
        records.sort(key=lambda record: (record.timestamp, record.trace_id), reverse=True)
        return Snapshot(
            traces=tuple(records[:limit]),
            capture_metadata={
                'source': 'orq-oql',
                'start': start.isoformat(),
                'end': end.isoformat(),
            },
        )

    async def _hydrate_page(
        self,
        summaries: list[tuple[Any, Any, bool]],
        project_names: Mapping[str, str],
        semaphore: asyncio.Semaphore,
    ) -> list[tuple[TraceRecord | None, int]]:
        tasks = [
            asyncio.create_task(
                self._hydrate_trace(
                    summary,
                    raw_summary,
                    project_names,
                    semaphore,
                    raw_capture_fallback=raw_capture_fallback,
                )
            )
            for summary, raw_summary, raw_capture_fallback in summaries
        ]
        try:
            return await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _hydrate_trace(
        self,
        summary: Any,
        raw_summary: Any,
        project_names: Mapping[str, str],
        semaphore: asyncio.Semaphore,
        *,
        raw_capture_fallback: bool,
    ) -> tuple[TraceRecord | None, int]:
        fallback_count = int(raw_capture_fallback)
        messages = _conversation_messages(raw_summary)
        if _usable(messages):
            return _record(summary, raw_summary, summary, raw_summary, messages, project_names), fallback_count

        trace_id = str(_field(summary, 'trace_id') or _field(summary, 'id') or '')
        if not trace_id:
            return None, fallback_count
        spans = await self._list_spans(trace_id, semaphore)
        for span in _eligible_spans(spans):
            span_id = str(_field(span, 'span_id') or _field(span, 'id') or '')
            if not span_id:
                continue
            async with semaphore:
                marker = _CAPTURE_REQUEST.set(object()) if self._registration is not None else None
                try:
                    response = await self._client.traces.get_span_async(
                        trace_id=trace_id,
                        span_id=span_id,
                        timeout_ms=SDK_TIMEOUT_MS,
                    )
                    raw_response = self._capture.pop(f'/traces/{trace_id}/spans/{span_id}')
                finally:
                    if marker is not None:
                        _CAPTURE_REQUEST.reset(marker)
            detail = _field(response, 'span') or response
            raw_detail = _field(raw_response, 'span') if raw_response else None
            detail_fallback = raw_detail is None
            if detail_fallback:
                raw_detail = _plain(detail)
            fallback_count += int(detail_fallback)
            messages = _conversation_messages(raw_detail)
            if _usable(messages):
                return _record(summary, raw_summary, detail, raw_detail, messages, project_names), fallback_count
        return None, fallback_count

    async def _list_spans(self, trace_id: str, semaphore: asyncio.Semaphore) -> list[Any]:
        spans: list[Any] = []
        page_token: str | None = None
        used_tokens: set[str] = set()
        pages = 0
        while True:
            if pages >= MAX_SPAN_PAGES:
                raise OrqSourceError(f'span pagination exceeded {MAX_SPAN_PAGES} pages for trace {trace_id!r}')
            pages += 1
            async with semaphore:
                response = await self._client.traces.list_spans_async(
                    trace_id=trace_id,
                    limit=PAGE_SIZE,
                    page_token=page_token,
                    timeout_ms=SDK_TIMEOUT_MS,
                )
            spans.extend(_as_list(_field(response, 'data')))
            if len(spans) > MAX_SPANS_PER_TRACE:
                raise OrqSourceError(f'span pagination exceeded {MAX_SPANS_PER_TRACE} spans for trace {trace_id!r}')
            if not _field(response, 'has_more'):
                return spans
            next_token = _field(response, 'next_page_token')
            if not next_token or str(next_token) in used_tokens:
                raise OrqSourceError(f'repeated span page token for trace {trace_id!r}')
            used_tokens.add(str(next_token))
            page_token = str(next_token)


def _select_summaries(
    summaries: list[Any],
    raw_by_id: Mapping[str, Any],
    seen_trace_ids: set[str],
    project_id: str | None,
) -> tuple[list[tuple[Any, Any, bool]], int]:
    """Enforce the selected project even when the trace query returns other projects."""

    selected: list[tuple[Any, Any, bool]] = []
    rejected = 0
    for summary in summaries:
        trace_id = _field(summary, 'trace_id') or _field(summary, 'id')
        if not trace_id or str(trace_id) in seen_trace_ids:
            continue
        seen_trace_ids.add(str(trace_id))
        raw_summary = raw_by_id.get(str(trace_id))
        raw = raw_summary if raw_summary is not None else _plain(summary)
        if project_id and str(_metadata_value(summary, raw, 'project_id') or '') != project_id:
            rejected += 1
            continue
        selected.append((summary, raw, raw_summary is None))
    return selected, rejected


def build_oql(facets: FacetSelection, numeric: NumericFilters, project_names: Mapping[str, str]) -> str:
    """Compile selected categorical and numeric filters into deterministic OQL."""

    clauses = [BASE_FILTER]
    if facets.project_id:
        if facets.project_id not in project_names:
            raise OrqSourceError(f'cannot resolve selected project id: {facets.project_id}')
        clauses.append(f'project_id in ({_oql_values([facets.project_id])})')
    elif facets.project:
        labels = project_labels(project_names)
        project_ids = sorted(project_id for project_id, label in labels.items() if label in facets.project)
        missing = set(facets.project) - set(labels.values())
        if missing:
            raise OrqSourceError(f'cannot resolve selected project names to ids: {sorted(missing)}')
        clauses.append(f'project_id in ({_oql_values(project_ids)})')
    for facet_name, field in FACET_OQL_FIELDS.items():
        values = sorted(getattr(facets, facet_name))
        if values:
            clauses.append(f'{field} in ({_oql_values(values)})')
    # Separate filter stages compose correctly. With ``and``, the traces API can
    # silently ignore a later categorical clause (including ``project_id``).
    # Numeric comparisons also require their own stages to avoid parser errors.
    stages = [f'filter {clause}' for clause in clauses]
    for field, minimum, maximum in (
        ('total_tokens', numeric.tokens_min, numeric.tokens_max),
        ('duration_ms', numeric.duration_ms_min, numeric.duration_ms_max),
    ):
        if minimum is not None:
            stages.append(f'filter {field} >= {minimum}')
        if maximum is not None:
            stages.append(f'filter {field} <= {maximum}')
    return f'fetch traces | {" | ".join(stages)} | sort end_time desc'


def _oql_values(values: list[str]) -> str:
    return ', '.join(json.dumps(value, ensure_ascii=False) for value in values)


def _eligible_spans(spans: list[Any]) -> list[Any]:
    excluded: set[str] = set()
    children: dict[str, list[str]] = defaultdict(list)
    by_id: dict[str, Any] = {}
    for span in spans:
        span_id = str(_field(span, 'span_id') or _field(span, 'id') or '')
        if not span_id:
            continue
        by_id[span_id] = span
        parent_id = _field(span, 'parent_span_id') or _field(span, 'parent_id')
        if parent_id:
            children[str(parent_id)].append(span_id)
        if str(_field(span, 'type') or '') in EVALUATOR_SPAN_TYPES:
            excluded.add(span_id)
    queue = deque(excluded)
    while queue:
        for child_id in children.get(queue.popleft(), []):
            if child_id not in excluded:
                excluded.add(child_id)
                queue.append(child_id)

    eligible = [
        span
        for span_id, span in by_id.items()
        if span_id not in excluded and str(_field(span, 'type') or '') in CONVERSATION_SPAN_TYPES
    ]
    eligible.sort(key=lambda span: (_recorded_time(span), str(_field(span, 'span_id') or '')), reverse=True)
    return eligible


def _record(
    trace_summary: Any,
    raw_trace: Any,
    selected: Any,
    raw_selected: Any,
    messages: list[dict[str, Any]],
    project_names: Mapping[str, str],
) -> TraceRecord | None:
    selected_summary = _field(selected, 'summary') or selected
    trace_id = str(_field(trace_summary, 'trace_id') or _field(trace_summary, 'id'))
    span_id = str(
        _field(selected_summary, 'span_id')
        or _field(selected_summary, 'id')
        or _field(trace_summary, 'leading_span_id')
        or _field(trace_summary, 'root_span_id')
        or trace_id
    )
    project_id = _metadata_value(trace_summary, raw_trace, 'project_id')
    project = project_names.get(str(project_id), str(project_id or 'unknown'))
    timestamp = _first_time(trace_summary, raw_trace) or _first_time(selected_summary, raw_selected)
    if timestamp is None:
        return None
    model = _metadata_label(
        _metadata_value(selected_summary, raw_selected, 'model') or _first_list_value(trace_summary, 'models')
    )
    provider = _metadata_label(
        _metadata_value(selected_summary, raw_selected, 'provider') or _first_list_value(trace_summary, 'providers')
    )
    status = _metadata_value(trace_summary, raw_trace, 'status')
    product = _metadata_value(trace_summary, raw_trace, 'product')
    trace_type = _leading_span_type(trace_summary, raw_trace)
    agent_name = _metadata_label(
        _metadata_value(trace_summary, raw_trace, 'agent_name')
        or _metadata_value(selected_summary, raw_selected, 'agent_name')
    )
    total_tokens = _numeric_metadata(trace_summary, raw_trace, selected_summary, raw_selected, name='total_tokens')
    duration_ms = _numeric_metadata(trace_summary, raw_trace, selected_summary, raw_selected, name='duration_ms')
    return TraceRecord(
        schema_version=1,
        trace_id=trace_id,
        span_id=span_id,
        timestamp=timestamp,
        messages=tuple(messages),
        project=project,
        model=str(model or 'unknown'),
        provider=str(provider or 'unknown'),
        status=str(status or 'unknown'),
        product=str(product or 'unknown'),
        trace_type=str(trace_type or 'unknown'),
        agent_name=str(agent_name or ''),
        tool_names=_tool_names(trace_summary, raw_trace, selected_summary, raw_selected),
        total_tokens=total_tokens,
        duration_ms=duration_ms,
        capture_metadata={'source': 'orq-oql', 'project_id': str(project_id or '')},
    )


def _conversation_messages(payload: Any) -> list[dict[str, Any]]:
    payload = _plain(payload)
    if not isinstance(payload, dict):
        return []
    attributes = _mapping(payload.get('attributes'))
    gen_ai = _mapping(attributes.get('gen_ai'))
    direct = _message_list(payload.get('messages'))
    if _usable(direct):
        return direct

    inputs = [gen_ai.get('input'), attributes.get('gen_ai.input'), payload.get('input')]
    outputs = [gen_ai.get('output'), attributes.get('gen_ai.output'), payload.get('output')]
    messages = next((parsed for value in inputs if (parsed := _message_list(value, default_role='user'))), [])
    output_messages = next(
        (parsed for value in outputs if (parsed := _message_list(value, default_role='assistant'))),
        [],
    )
    for message in output_messages:
        if not messages or message != messages[-1]:
            messages.append(message)
    return messages


def _message_list(value: Any, *, default_role: str | None = None) -> list[dict[str, Any]]:
    if isinstance(value, str):
        original = value
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return [{'role': default_role, 'content': value}] if default_role and value.strip() else []
        if isinstance(decoded, (dict, list)):
            decoded_messages = _structured_message_list(decoded, default_role=default_role)
            if decoded_messages is not None:
                return decoded_messages
        return [{'role': default_role, 'content': original}] if default_role and original.strip() else []
    messages = _structured_message_list(value, default_role=default_role)
    return messages if messages is not None else []


def _structured_message_list(value: Any, *, default_role: str | None = None) -> list[dict[str, Any]] | None:
    if isinstance(value, Mapping):
        return _structured_mapping_message(value, default_role=default_role)
    if not isinstance(value, list):
        return None
    messages = []
    recognized = not value
    for item in value:
        if not isinstance(item, Mapping):
            continue
        parsed = _structured_message_list(item, default_role=default_role)
        if parsed is not None:
            recognized = True
            messages.extend(parsed)
    return messages if recognized else None


def _structured_mapping_message(value: Mapping[str, Any], *, default_role: str | None) -> list[dict[str, Any]] | None:
    for key in ('messages', 'input'):
        if key in value:
            return _message_list(value[key], default_role=default_role)
    if 'prompt' in value:
        return _message_list(value['prompt'], default_role=default_role)
    choices = value.get('choices')
    if isinstance(choices, list):
        return _choice_messages(choices, default_role=default_role)
    if any(key in value for key in ('content', 'parts', 'role', 'tool_calls')):
        message = dict(value)
        if not message.get('role') and default_role:
            message['role'] = default_role
        return [message] if message.get('role') else []
    return None


def _choice_messages(choices: list[Any], *, default_role: str | None) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for choice in choices:
        if not isinstance(choice, Mapping):
            continue
        if isinstance(choice.get('message'), Mapping):
            parsed = _structured_message_list(choice['message'], default_role=default_role)
            if parsed is not None:
                messages.extend(parsed)
        elif 'text' in choice:
            messages.extend(_structured_message_list({'content': choice.get('text')}, default_role=default_role) or [])
    return messages


def _usable(messages: list[dict[str, Any]]) -> bool:
    return any(_message_has_content(message) for message in messages)


def _message_has_content(message: dict[str, Any]) -> bool:
    content = message.get('content')
    if _content_has_value(content):
        return True
    if message.get('tool_calls'):
        return True
    parts = message.get('parts')
    return isinstance(parts, list) and any(_content_block_has_value(part) for part in parts)


def _content_has_value(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if value is None:
        return False
    if isinstance(value, (list, tuple)):
        return any(_content_block_has_value(item) for item in value)
    if isinstance(value, Mapping):
        return _content_block_has_value(value)
    return True


def _content_block_has_value(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return _content_has_value(value)
    content_keys = ('content', 'image_url', 'input_text', 'output_text', 'text')
    if any(key in value for key in content_keys):
        return any(_content_has_value(value.get(key)) for key in content_keys if key in value)
    return any(_content_has_value(item) for key, item in value.items() if key != 'type')


def _metadata_value(typed: Any, raw: Any, *names: str) -> Any:
    for source in (typed, raw):
        source = _field(source, 'summary') or source
        for name in names:
            value = _field(source, name)
            if value not in (None, ''):
                return value
        attributes = _mapping(_field(source, 'attributes'))
        gen_ai = _mapping(attributes.get('gen_ai'))
        request = _mapping(gen_ai.get('request'))
        for name in names:
            value = request.get(name) or gen_ai.get(name) or attributes.get(name)
            if value not in (None, ''):
                return value
    return None


def _numeric_metadata(*sources: Any, name: str) -> int | None:
    value = None
    for typed, raw in zip(sources[::2], sources[1::2], strict=True):
        value = _metadata_value(typed, raw, name)
        if value not in (None, ''):
            break
        for source in (typed, raw):
            for candidate in (source, _field(source, 'summary')):
                usage = _field(candidate, 'usage')
                value = _field(usage, name)
                if value not in (None, ''):
                    break
            if value not in (None, ''):
                break
        if value not in (None, ''):
            break
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if result >= 0 else None


def _tool_names(*sources: Any) -> tuple[str, ...]:
    values: list[str] = []
    for typed, raw in zip(sources[::2], sources[1::2], strict=True):
        for source in (typed, raw):
            for candidate_source in (source, _field(source, 'summary')):
                for key in ('tool_names', 'tool_name'):
                    value = _field(candidate_source, key)
                    candidates = value if isinstance(value, (list, tuple, set, frozenset)) else (value,)
                    for candidate in candidates:
                        if isinstance(candidate, Mapping):
                            candidate = candidate.get('name') or candidate.get('id')
                        if isinstance(candidate, str) and candidate and candidate not in values:
                            values.append(candidate)
    return tuple(values)


def _leading_span_type(typed: Any, raw: Any) -> Any:
    for source in (typed, raw):
        attributes = _mapping(_field(source, 'attributes'))
        orq = _mapping(attributes.get('orq'))
        for leading in (_mapping(orq.get('leading_span')), _mapping(attributes.get('leading_span'))):
            if span_type := leading.get('span_type'):
                return span_type
    return _metadata_value(typed, raw, 'type', 'operation')


def _metadata_label(value: Any) -> Any:
    if isinstance(value, Mapping):
        return value.get('name') or value.get('id')
    return value


def _first_list_value(value: Any, name: str) -> Any:
    values = _field(value, name)
    return values[0] if isinstance(values, (list, tuple)) and values else None


def _recorded_time(value: Any) -> datetime:
    return _first_time(value, value) or datetime.min.replace(tzinfo=timezone.utc)


def _first_time(typed: Any, raw: Any) -> datetime | None:
    for source in (typed, raw):
        source = _field(source, 'summary') or source
        for name in ('ended_at', 'end_time', 'started_at', 'start_time'):
            parsed = _parse_time(_field(source, name))
            if parsed is not None:
                return parsed
    return None


def _parse_time(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            return None
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            parsed = datetime.fromtimestamp(value / 1000 if value > 10_000_000_000 else value, tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    else:
        return None
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)


def _aware_bound(value: datetime, name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise OrqSourceError(f'{name} must include a timezone offset')
    return value.astimezone(timezone.utc)


def _raw_trace_summaries(payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not payload:
        return []
    search = payload.get('search') if isinstance(payload.get('search'), Mapping) else payload
    data = search.get('data') if isinstance(search, Mapping) else None
    return [value for value in data or [] if isinstance(value, dict)]


def _as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, (list, tuple)) else []


def _field(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _mapping(value: Any) -> dict[str, Any]:
    plain = _plain(value)
    return plain if isinstance(plain, dict) else {}


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if hasattr(value, 'model_dump'):
        return value.model_dump(exclude_none=True)
    if hasattr(value, '__dict__'):
        return {key: _plain(item) for key, item in vars(value).items() if not key.startswith('_')}
    return value
