"""Bounded live OQL acquisition normalized into the trace finder snapshot contract."""

from __future__ import annotations

import asyncio
import json
import math
import re
import time
import weakref
from collections import defaultdict, deque
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from contextlib import suppress
from contextvars import ContextVar
from dataclasses import dataclass
from dataclasses import field as dc_field
from datetime import datetime, timedelta, timezone
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal

from loguru import logger

from evaluatorq.common.trace_input import detect_message_format, parse_messages
from evaluatorq.openresponses.otel_messages import is_responses_item

from .facets import _project_names, project_labels
from .models import FacetSelection, NumericFilters, Snapshot, TraceRecord
from .progress import report_load_progress
from .rows import TraceRow, _models, _response_models, row_from_summary
from .rows import parse_time as _parse_time
from .span_status import is_error_span

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
TARGET_RELOAD_PAGE_BUDGET_SECONDS = 30
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


class _TargetDeadlineExceeded(asyncio.TimeoutError):
    """The local targeted-reload budget expired, distinct from a provider timeout."""


def _remaining_target_seconds(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise _TargetDeadlineExceeded
    return remaining


def _request_timeout_ms(deadline: float | None) -> int:
    if deadline is None:
        return SDK_TIMEOUT_MS
    return max(1, min(SDK_TIMEOUT_MS, int(_remaining_target_seconds(deadline) * 1000)))


async def _await_target_deadline(awaitable: Awaitable[Any], deadline: float | None) -> Any:
    if deadline is None:
        return await awaitable
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        close = getattr(awaitable, 'close', None)
        if callable(close):
            close()
        raise _TargetDeadlineExceeded
    task = asyncio.ensure_future(awaitable)
    try:
        done, _ = await asyncio.wait({task}, timeout=remaining)
        if done:
            return task.result()
        raise _TargetDeadlineExceeded
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


def _empty_target_snapshot(start: datetime, end: datetime) -> Snapshot:
    return Snapshot(
        traces=(),
        capture_metadata={
            'source': 'orq-oql',
            'start': start.isoformat(),
            'end': end.isoformat(),
            'incomplete_reason': 'target_deadline',
        },
    )


class OrqSourceError(ValueError):
    """Live trace acquisition could not produce a valid snapshot."""


@dataclass
class _Scan:
    """Counters and completion state one paging pass shares with its consumer."""

    scanned: int = 0
    wrong_project: int = 0
    incomplete_reason: str | None = None


@dataclass
class _HydrationIssues:
    """Per-batch failures a hydration pass reports as one warning each, not one per trace."""

    system_prompt_errors: list[str] = dc_field(default_factory=list)
    missing_reply: list[str] = dc_field(default_factory=list)
    span_errors: list[str] = dc_field(default_factory=list)

    def log(self, total: int) -> None:
        if self.system_prompt_errors:
            logger.warning(
                'system prompt lookup failed for {} of {} trace(s); first error: {}',
                len(self.system_prompt_errors),
                total,
                self.system_prompt_errors[0],
            )
        if self.span_errors:
            logger.warning(
                'span lookup failed for {} of {} trace(s); kept summary messages where usable; first error: {}',
                len(self.span_errors),
                total,
                self.span_errors[0],
            )
        if self.missing_reply:
            logger.warning(
                '{} of {} trace(s) report output tokens but their hydrated spans contain no visible reply; first: {}',
                len(self.missing_reply),
                total,
                self.missing_reply[0],
            )


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
        target_trace_ids: set[str] | frozenset[str] | None = None,
    ) -> Snapshot:
        """Load at most 5000 traces, optionally scanning for specific IDs, newest first."""

        try:
            return await self._load_with_lifecycle(start, end, limit, facets, numeric, target_trace_ids)
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
        target_trace_ids: set[str] | frozenset[str] | None,
    ) -> Snapshot:
        if self._owns_client:
            async with self._client:
                return await self._validated_load(start, end, limit, facets, numeric, target_trace_ids)
        return await self._validated_load(start, end, limit, facets, numeric, target_trace_ids)

    async def _validated_load(
        self,
        start: datetime | None,
        end: datetime | None,
        limit: int,
        facets: FacetSelection,
        numeric: NumericFilters,
        target_trace_ids: set[str] | frozenset[str] | None,
    ) -> Snapshot:
        resolved_start, resolved_end = _check_window(start, end, limit)
        return await self._load(
            resolved_start, resolved_end, min(limit, MAX_LIVE_TRACES), facets, numeric, target_trace_ids
        )

    async def _load(
        self,
        start: datetime,
        end: datetime,
        limit: int,
        facets: FacetSelection,
        numeric: NumericFilters,
        target_trace_ids: set[str] | frozenset[str] | None = None,
    ) -> Snapshot:
        target_ids = set(target_trace_ids) if target_trace_ids is not None else None
        target_deadline = time.monotonic() + TARGET_RELOAD_PAGE_BUDGET_SECONDS if target_ids is not None else None
        try:
            project_names = await _await_target_deadline(_project_names(self._client), target_deadline)
        except _TargetDeadlineExceeded:
            logger.warning(
                'targeted trace reload reached its {} second deadline during project lookup',
                TARGET_RELOAD_PAGE_BUDGET_SECONDS,
            )
            return _empty_target_snapshot(start, end)
        oql = build_oql(facets, numeric, project_names)
        semaphore = asyncio.Semaphore(self._hydration_concurrency)
        records: list[TraceRecord] = []
        dropped_count = 0
        fallback_count = 0
        scan = _Scan()
        try:
            async for selected in self._pages(
                start,
                end,
                oql,
                facets.project_id,
                limit,
                lambda: 1 if target_ids is not None else limit - len(records),
                scan,
                target_ids=target_ids,
                deadline=target_deadline,
            ):
                issues = _HydrationIssues()
                hydrated, hydration_timed_out = await self._hydrate_page(
                    selected, project_names, semaphore, issues, deadline=target_deadline
                )
                issues.log(len(selected))
                fallback_count += sum(result[1] for result in hydrated)
                dropped_count += sum(record is None for record, _ in hydrated)
                records.extend(record for record, _ in hydrated if record is not None)
                report_load_progress(min(len(records), limit), limit)
                if hydration_timed_out:
                    logger.warning(
                        'targeted trace reload reached its {} second deadline during trace hydration',
                        TARGET_RELOAD_PAGE_BUDGET_SECONDS,
                    )
                    scan.incomplete_reason = 'target_deadline'
                    break
                if target_ids is not None and target_ids.issubset({record.trace_id for record in records}):
                    break
        except _TargetDeadlineExceeded:
            logger.warning(
                'targeted trace reload reached its {} second deadline during trace acquisition',
                TARGET_RELOAD_PAGE_BUDGET_SECONDS,
            )
            scan.incomplete_reason = 'target_deadline'

        wrong_project_count = scan.wrong_project
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
        capture_metadata = {'source': 'orq-oql', 'start': start.isoformat(), 'end': end.isoformat()}
        incomplete_reason = scan.incomplete_reason
        if (
            incomplete_reason is None
            and target_ids is not None
            and not target_ids.issubset({record.trace_id for record in records})
            and scan.scanned >= MAX_LIVE_TRACES
        ):
            incomplete_reason = 'scan_limit'
        if incomplete_reason is not None:
            capture_metadata['incomplete_reason'] = incomplete_reason
        return Snapshot(traces=tuple(records[:limit]), capture_metadata=capture_metadata)

    async def _query_page(
        self,
        start: datetime,
        end: datetime,
        oql: str,
        page_limit: int,
        page_token: str | None,
        *,
        deadline: float | None = None,
    ) -> tuple[Any, Any]:
        registration = self._registration
        marker = _CAPTURE_REQUEST.set(object()) if registration is not None else None
        try:

            async def query() -> tuple[Any, Any]:
                if registration is None:
                    response = await self._client.traces.query_async(
                        from_=start,
                        to=end,
                        oql=oql,
                        limit=page_limit,
                        page_token=page_token,
                        timeout_ms=_request_timeout_ms(deadline),
                    )
                    return response, self._capture.pop('/traces/query')
                async with registration.lock_for(asyncio.get_running_loop()):
                    response = await self._client.traces.query_async(
                        from_=start,
                        to=end,
                        oql=oql,
                        limit=page_limit,
                        page_token=page_token,
                        timeout_ms=_request_timeout_ms(deadline),
                    )
                    return response, self._capture.pop('/traces/query')

            return await _await_target_deadline(query(), deadline)
        finally:
            if marker is not None:
                _CAPTURE_REQUEST.reset(marker)

    async def _pages(
        self,
        start: datetime,
        end: datetime,
        oql: str,
        project_id: str | None,
        limit: int,
        remaining: Callable[[], int],
        scan: _Scan,
        *,
        target_ids: set[str] | None = None,
        deadline: float | None = None,
    ) -> AsyncIterator[list[tuple[Any, Any, bool]]]:
        """Yield bounded, unique, project-checked OQL pages to the loading surface.

        Yields:
            Unique summaries for one page after project and target filtering.
        """
        seen_trace_ids: set[str] = set()
        used_tokens: set[str] = set()
        max_scanned = MAX_LIVE_TRACES if target_ids is not None else max(PAGE_SIZE, limit * 5)
        max_pages = max(20, (max_scanned + PAGE_SIZE - 1) // PAGE_SIZE * 2)
        pages_scanned = 0
        page_token: str | None = None
        while remaining() > 0:
            if deadline is not None and time.monotonic() >= deadline:
                raise _TargetDeadlineExceeded
            if scan.scanned >= max_scanned or pages_scanned >= max_pages:
                logger.warning(
                    'stopped trace search after scanning {} summaries across {} pages for {} usable traces',
                    scan.scanned,
                    pages_scanned,
                    limit,
                )
                if target_ids is not None:
                    scan.incomplete_reason = 'scan_limit'
                return
            pages_scanned += 1
            if page_token is not None:
                if page_token in used_tokens:
                    raise OrqSourceError(f'repeated OQL page token {page_token!r}')
                used_tokens.add(page_token)
            page_limit = PAGE_SIZE if target_ids is not None else min(PAGE_SIZE, remaining())
            response, raw_page = await self._query_page(start, end, oql, page_limit, page_token, deadline=deadline)
            search = _field(response, 'search') or response
            summaries = _as_list(_field(search, 'data'))
            scan.scanned += len(summaries)
            raw_by_id = {
                str(trace_id): payload
                for payload in _raw_trace_summaries(raw_page)
                if (trace_id := _field(payload, 'trace_id') or _field(payload, 'id'))
            }
            selected, rejected = _select_summaries(summaries, raw_by_id, seen_trace_ids, project_id)
            scan.wrong_project += rejected
            if target_ids is not None:
                selected = [
                    item for item in selected if str(_field(item[0], 'trace_id') or _field(item[0], 'id')) in target_ids
                ]
            yield selected
            if remaining() <= 0:
                return
            if not _field(search, 'has_more'):
                return
            next_token = _field(search, 'next_page_token')
            if not next_token:
                raise OrqSourceError('OQL response reported more pages without a page token')
            if str(next_token) in used_tokens:
                raise OrqSourceError(f'repeated OQL page token {next_token!r}')
            page_token = str(next_token)

    async def search(
        self,
        start: datetime,
        end: datetime,
        limit: int,
        *,
        facets: FacetSelection,
        numeric: NumericFilters,
        on_page: Callable[[tuple[TraceRow, ...]], None] | None = None,
    ) -> tuple[TraceRow, ...]:
        """Return up to *limit* table rows, newest first, from summaries alone (no span fetches)."""
        start, end = _check_window(start, end, limit)
        limit = min(limit, MAX_LIVE_TRACES)
        project_names = await _project_names(self._client)
        oql = build_oql(facets, numeric, project_names)
        rows: list[TraceRow] = []
        scan = _Scan()
        fallback_count = 0
        async for selected in self._pages(start, end, oql, facets.project_id, limit, lambda: limit - len(rows), scan):
            fallback_count += sum(1 for _, _, fallback in selected if fallback)
            rows.extend(row for summary, raw, _ in selected if (row := row_from_summary(summary, raw)) is not None)
            if on_page is not None:
                on_page(tuple(rows[:limit]))
        if scan.wrong_project:
            logger.warning(
                'discarded {} trace summary(s) outside selected project {} despite the OQL filter',
                scan.wrong_project,
                facets.project_id,
            )
            if not rows:
                raise OrqSourceError('Orq returned only traces outside the selected project. No traces were loaded.')
        if fallback_count:
            logger.warning(
                'used SDK-model fallback for {} trace summary response(s) because raw-response capture was unavailable; '
                'some optional explorer fields may render as —',
                fallback_count,
            )
        exclusive = sum(row.usage_exclusive for row in rows)
        if exclusive:
            logger.warning(
                '{} trace(s) reported cache tokens outside prompt_tokens (native provider usage); '
                'counted them into tokens in',
                exclusive,
            )
        return tuple(rows[:limit])

    async def hydrate_rows(self, rows: Sequence[TraceRow]) -> dict[str, TraceRecord | None]:
        """Fetch messages; successful empty traces map to ``None``, failed traces are omitted."""
        project_names = await _project_names(self._client)
        semaphore = asyncio.Semaphore(self._hydration_concurrency)
        errors: list[str] = []
        failed_trace_ids: set[str] = set()
        fallbacks: list[str] = []
        issues = _HydrationIssues()

        done = 0

        async def one(row: TraceRow) -> TraceRecord | None:
            nonlocal done
            try:
                record, fallback_count = await self._hydrate_trace(
                    row, row.raw, project_names, semaphore, issues, raw_capture_fallback=False
                )
                if fallback_count:
                    fallbacks.append(row.trace_id)
            except Exception as error:  # noqa: BLE001 — one trace's failure must not blank the page
                errors.append(f'{type(error).__name__}: {error}')
                failed_trace_ids.add(row.trace_id)
                return None
            finally:
                done += 1
                report_load_progress(done, len(rows))
            return record

        records = await asyncio.gather(*(one(row) for row in rows))
        issues.log(len(rows))
        if fallbacks:
            logger.warning('used SDK-model fallback for span details of {} trace(s)', len(fallbacks))
        empty = sum(record is None for record in records) - len(errors)
        if errors:
            logger.warning(
                'fetching messages failed for {} of {} trace(s); first error: {}', len(errors), len(rows), errors[0]
            )
        if empty:
            logger.warning('{} of {} trace(s) have no usable messages; drawing them as empty bars', empty, len(rows))
        return {
            row.trace_id: record
            for row, record in zip(rows, records, strict=True)
            if row.trace_id not in failed_trace_ids
        }

    async def _hydrate_page(
        self,
        summaries: list[tuple[Any, Any, bool]],
        project_names: Mapping[str, str],
        semaphore: asyncio.Semaphore,
        issues: _HydrationIssues,
        *,
        deadline: float | None = None,
    ) -> tuple[list[tuple[TraceRecord | None, int]], bool]:
        tasks = [
            asyncio.create_task(
                self._hydrate_trace(
                    summary,
                    raw_summary,
                    project_names,
                    semaphore,
                    issues,
                    deadline=deadline,
                    raw_capture_fallback=raw_capture_fallback,
                )
            )
            for summary, raw_summary, raw_capture_fallback in summaries
        ]
        if not tasks:
            return [], False
        try:
            if deadline is None:
                return await asyncio.gather(*tasks), False
            remaining = max(0.0, deadline - time.monotonic())
            done, pending = await asyncio.wait(tasks, timeout=remaining, return_when=asyncio.FIRST_EXCEPTION)
            completed: list[tuple[TraceRecord | None, int]] = []
            hydration_timed_out = bool(pending)
            for task in tasks:
                if task not in done:
                    continue
                try:
                    completed.append(task.result())
                except _TargetDeadlineExceeded:
                    hydration_timed_out = True
            if not pending:
                return completed, hydration_timed_out
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            return completed, True
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
        issues: _HydrationIssues,
        *,
        deadline: float | None = None,
        raw_capture_fallback: bool,
    ) -> tuple[TraceRecord | None, int]:
        fallback_count = int(raw_capture_fallback)
        messages = _conversation_messages(raw_summary)
        output_tokens = _field(_field(raw_summary, 'usage'), 'completion_tokens')
        if output_tokens is None:
            output_tokens = _summary_output_tokens(summary)
        missing_reply = (
            isinstance(output_tokens, (int, float))
            and not isinstance(output_tokens, bool)
            and output_tokens > 0
            and not _has_reply(messages)
        )
        # A trace-level agent span can contain a transcript but no model; the model
        # belongs to its child chat-completion span. In that case keep scanning spans
        # so the hydrated record can supply the model as well as the transcript.
        if _usable(messages) and not missing_reply and _models(summary, raw_summary):
            messages = await self._with_system_prompt(
                summary, raw_summary, messages, semaphore, issues, deadline=deadline
            )
            return _record(summary, raw_summary, summary, raw_summary, messages, project_names), fallback_count

        trace_id = str(_field(summary, 'trace_id') or _field(summary, 'id') or '')
        if not trace_id:
            return None, fallback_count
        fallback: TraceRecord | None = None
        span_lookup_failed = False
        span_models: tuple[str, ...] = ()
        try:
            spans = await self._list_spans(trace_id, semaphore, deadline=deadline)
            span_models = tuple(dict.fromkeys(model for span in spans for model in _models(span, _plain(span))))
            for span in _eligible_spans(spans):
                span_id = str(_field(span, 'span_id') or _field(span, 'id') or '')
                if not span_id:
                    continue
                response, raw_response = await self._get_span(trace_id, span_id, semaphore, deadline=deadline)
                detail = _field(response, 'span') or response
                raw_detail = _field(raw_response, 'span') if raw_response else None
                detail_fallback = raw_detail is None
                if detail_fallback:
                    raw_detail = _plain(detail)
                fallback_count += int(detail_fallback)
                detail_messages = _conversation_messages(raw_detail)
                if _usable(detail_messages):
                    record = _with_span_models(
                        _record(summary, raw_summary, detail, raw_detail, detail_messages, project_names), span_models
                    )
                    if not missing_reply or _has_reply(detail_messages):
                        return record, fallback_count
                    if fallback is None:
                        fallback = record
        except Exception as exc:  # the summary messages may still be usable; logged once per batch
            if fallback is None and not _usable(messages):
                raise  # nothing to show: keep the load failure visible instead of an empty trace
            span_lookup_failed = True
            issues.span_errors.append(f'{trace_id}: {type(exc).__name__}: {exc}')
        if missing_reply and not span_lookup_failed:
            issues.missing_reply.append(trace_id)
        if fallback is not None:
            return fallback, fallback_count
        if _usable(messages):
            messages = await self._with_system_prompt(
                summary, raw_summary, messages, semaphore, issues, deadline=deadline
            )
            return _with_span_models(
                _record(summary, raw_summary, summary, raw_summary, messages, project_names), span_models
            ), fallback_count
        return None, fallback_count

    async def _get_span(
        self, trace_id: str, span_id: str, semaphore: asyncio.Semaphore, *, deadline: float | None = None
    ) -> tuple[Any, Any]:
        async with semaphore:
            marker = _CAPTURE_REQUEST.set(object()) if self._registration is not None else None
            try:
                response = await _await_target_deadline(
                    self._client.traces.get_span_async(
                        trace_id=trace_id,
                        span_id=span_id,
                        timeout_ms=_request_timeout_ms(deadline),
                    ),
                    deadline,
                )
                raw_response = self._capture.pop(f'/traces/{trace_id}/spans/{span_id}')
            finally:
                if marker is not None:
                    _CAPTURE_REQUEST.reset(marker)
        return response, raw_response

    async def _with_system_prompt(
        self,
        summary: Any,
        raw_summary: Any,
        messages: list[dict[str, Any]],
        semaphore: asyncio.Semaphore,
        issues: _HydrationIssues,
        *,
        deadline: float | None = None,
    ) -> list[dict[str, Any]]:
        """Prepend the system prompt, which Orq keeps only on the span detail."""
        if any(message.get('role') == 'system' for message in messages):
            return messages
        trace_id = str(_field(summary, 'trace_id') or _field(summary, 'id') or '')
        span_id = str(_field(summary, 'span_id') or _field(raw_summary, 'span_id') or '')
        if not trace_id or not span_id:
            return messages
        try:
            async with semaphore:
                response = await _await_target_deadline(
                    self._client.traces.get_span_async(
                        trace_id=trace_id,
                        span_id=span_id,
                        timeout_ms=_request_timeout_ms(deadline),
                    ),
                    deadline,
                )
        except _TargetDeadlineExceeded:
            raise
        except Exception as exc:  # noqa: BLE001 - enrichment is best-effort and logged
            issues.system_prompt_errors.append(f'{type(exc).__name__}: {exc}')
            return messages
        detail = _plain(_field(response, 'span') or response)
        system = [m for m in _conversation_messages(detail) if m.get('role') == 'system' and _message_has_content(m)]
        return [*system, *messages]

    async def _list_spans(
        self, trace_id: str, semaphore: asyncio.Semaphore, *, deadline: float | None = None
    ) -> list[Any]:
        spans: list[Any] = []
        page_token: str | None = None
        used_tokens: set[str] = set()
        pages = 0
        while True:
            if pages >= MAX_SPAN_PAGES:
                raise OrqSourceError(f'span pagination exceeded {MAX_SPAN_PAGES} pages for trace {trace_id!r}')
            pages += 1
            async with semaphore:
                response = await _await_target_deadline(
                    self._client.traces.list_spans_async(
                        trace_id=trace_id,
                        limit=PAGE_SIZE,
                        page_token=page_token,
                        timeout_ms=_request_timeout_ms(deadline),
                    ),
                    deadline,
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

    async def list_spans(self, trace_id: str) -> list[Any]:
        """Load every bounded span summary for one trace using the canonical pager."""
        if not trace_id:
            return []
        return await self._list_spans(trace_id, asyncio.Semaphore(self._hydration_concurrency))

    async def enrich_selected_signal_spans(self, records: Sequence[TraceRecord]) -> tuple[TraceRecord, ...]:
        """Attach status from spans matching selected transcript tool-call IDs only.

        Span listing happens after population selection. A sibling span contributes
        nothing unless its recorded call ID exactly matches a call in that trace's
        projected conversation.
        """
        semaphore = asyncio.Semaphore(self._hydration_concurrency)

        async def enrich(record: TraceRecord) -> TraceRecord:
            call_ids = {
                str(call_id)
                for message in record.messages
                if message.get('role') == 'assistant'
                for call in message.get('tool_calls') or []
                if isinstance(call, Mapping) and (call_id := (call.get('id') or call.get('call_id')))
            }
            try:
                spans = await self._list_spans(record.trace_id, semaphore)
                by_id = {
                    str(span_id): span for span in spans if (span_id := (_field(span, 'span_id') or _field(span, 'id')))
                }
                selected_span = by_id.get(record.span_id)
                if selected_span is None:
                    coverage = {
                        'selected_span_id': record.span_id,
                        'selected_span_found': False,
                        'scoped_span_count': 0,
                        'matched_tool_span_count': 0,
                        'missing_call_ids': sorted(call_ids),
                    }
                    logger.warning(
                        'selected trace {} span {} was missing during signal enrichment',
                        record.trace_id,
                        record.span_id,
                    )
                    return record.model_copy(
                        update={'capture_metadata': {**record.capture_metadata, 'signal_span_coverage': coverage}}
                    )
                children: dict[str, list[str]] = defaultdict(list)
                for span in spans:
                    child_id = str(_field(span, 'span_id') or _field(span, 'id') or '')
                    parent_id = _field(span, 'parent_span_id') or _field(span, 'parent_id')
                    if child_id and parent_id:
                        children[str(parent_id)].append(child_id)
                scoped_ids = {record.span_id}
                queue = deque([record.span_id])
                while queue:
                    for child_id in children.get(queue.popleft(), []):
                        if child_id not in scoped_ids:
                            scoped_ids.add(child_id)
                            queue.append(child_id)
                matching = [
                    span
                    for span_id, span in by_id.items()
                    if span_id in scoped_ids and _span_tool_call_id(span) in call_ids
                ]
                statuses: dict[str, dict[str, Any]] = {}
                enrichment_errors: list[str] = []
                for span in matching:
                    span_id = str(_field(span, 'span_id') or _field(span, 'id') or '')
                    call_id = _span_tool_call_id(span)
                    if not span_id or call_id is None:
                        continue
                    try:
                        detail_response, raw_response = await self._get_span(record.trace_id, span_id, semaphore)
                        detail = _field(detail_response, 'span') or detail_response
                        raw_detail = _field(raw_response, 'span') if raw_response else None
                        detail = raw_detail or _plain(detail)
                    except Exception as error:  # noqa: BLE001 - summary status may still carry coverage
                        enrichment_errors.append(f'{span_id}: {type(error).__name__}: {error}')
                        detail = span
                    status = _signal_status(detail)
                    if status:
                        statuses[call_id] = status
                try:
                    selected_response, selected_raw = await self._get_span(record.trace_id, record.span_id, semaphore)
                    selected_detail = _field(selected_response, 'span') or selected_response
                    selected_raw_span = _field(selected_raw, 'span') if selected_raw else None
                    selected_detail = selected_raw_span or _plain(selected_detail)
                    selected_detail_captured = True
                except Exception as error:  # noqa: BLE001 - retain explicit missing-coverage state
                    enrichment_errors.append(f'{record.span_id}: {type(error).__name__}: {error}')
                    selected_detail = selected_span
                    selected_detail_captured = False
                selected_data = _selected_span_signal_data(selected_detail, record.messages)
                coverage = {
                    'selected_span_id': record.span_id,
                    'selected_span_found': True,
                    'scoped_span_count': len(scoped_ids),
                    'matched_tool_span_count': len(matching),
                    'matched_call_ids': sorted(statuses),
                    'missing_call_ids': sorted(call_ids - set(statuses)),
                    'selected_data_correlated': bool(selected_data.get('step_order') is not None),
                    'responses_items_captured': sum(
                        len(items)
                        for items in _mapping(selected_data.get('output_items_by_order')).values()
                        if isinstance(items, list)
                    ),
                    'selected_detail_captured': selected_detail_captured,
                    'enrichment_errors': enrichment_errors,
                }
                if selected_data.get('output_item_count') and selected_data.get('step_order') is None:
                    logger.warning(
                        'selected trace {} response items could not be correlated to one transcript step',
                        record.trace_id,
                    )
                capture_metadata = {
                    **record.capture_metadata,
                    'signal_span_coverage': coverage,
                    'signal_tool_spans': statuses,
                    'signal_selected_span': selected_data,
                }
                return record.model_copy(update={'capture_metadata': capture_metadata})
            except Exception as error:  # noqa: BLE001 - enrichment must not discard a selected trace
                logger.warning('selected trace {} signal-span enrichment failed: {}', record.trace_id, error)
                return record.model_copy(
                    update={
                        'capture_metadata': {
                            **record.capture_metadata,
                            'signal_span_coverage': {
                                'selected_span_id': record.span_id,
                                'selected_span_found': False,
                                'enrichment_error': f'{type(error).__name__}: {error}',
                            },
                        }
                    }
                )

        return tuple(await asyncio.gather(*(enrich(record) for record in records)))

    async def first_error_message(self, trace_id: str, spans: Sequence[Any]) -> str | None:
        """Fetch the first errored span's status text from its raw SDK response, when present."""
        if not trace_id or self._registration is None:
            return None
        failed_span_id: str | None = None
        for span in spans:
            if is_error_span(span):
                failed_span_id = str(_field(span, 'span_id') or _field(span, 'id') or '')
                if failed_span_id:
                    break
        if not failed_span_id:
            return None

        async with asyncio.Semaphore(self._hydration_concurrency):
            marker = _CAPTURE_REQUEST.set(object())
            try:
                await self._client.traces.get_span_async(
                    trace_id=trace_id,
                    span_id=failed_span_id,
                    timeout_ms=SDK_TIMEOUT_MS,
                )
                raw_response = self._capture.pop(f'/traces/{trace_id}/spans/{failed_span_id}')
            finally:
                _CAPTURE_REQUEST.reset(marker)
        raw_span = _field(raw_response, 'span') if raw_response else None
        attributes = _field(raw_span, 'attributes')
        for path in (('otlp', 'status', 'message'), ('otel', 'status_description')):
            value = attributes
            for key in path:
                value = _field(value, key)
                if value is None:
                    break
            if isinstance(value, str) and value.strip():
                return value
        return None


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


def _load_target_reached(records: list[TraceRecord], limit: int, targets: set[str] | None) -> bool:
    """Stop at the normal usable-record limit or once every requested ID was hydrated."""
    if targets is None:
        return len(records) >= limit
    return targets.issubset({record.trace_id for record in records})


def _matching_targets(summaries: list[tuple[Any, Any, bool]], targets: set[str] | None) -> list[tuple[Any, Any, bool]]:
    """Keep every summary normally and only requested IDs on targeted reloads."""
    if targets is None:
        return summaries
    return [item for item in summaries if str(_field(item[0], 'trace_id') or _field(item[0], 'id')) in targets]


def build_oql(facets: FacetSelection, numeric: NumericFilters, project_names: Mapping[str, str]) -> str:
    """Compile selected categorical and numeric filters into deterministic OQL."""

    # A bare model call (jev, the compiler, any router call) is a generate_content trace, so the base filter
    # would hide every trace of the model or provider the user just picked; drop it for those.
    clauses = [] if facets.model or facets.provider else [BASE_FILTER]
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


def _with_span_models(record: TraceRecord | None, span_models: tuple[str, ...]) -> TraceRecord | None:
    if record is None or not span_models or record.model not in {'', 'unknown'}:
        return record
    return record.model_copy(update={'model': ', '.join(span_models)})


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
    response_models = _response_models(selected_summary, raw_selected) or _response_models(trace_summary, raw_trace)
    model = (
        response_models[0]
        if response_models
        else _metadata_label(
            _metadata_value(selected_summary, raw_selected, 'model') or _first_list_value(trace_summary, 'models')
        )
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
    tool_definition_count, tool_definition_tokens = _tool_definition_size(
        raw_selected, selected, raw_trace, trace_summary
    )
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
        tool_definition_count=tool_definition_count,
        tool_definition_tokens=tool_definition_tokens,
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
    direct = _normalised_messages(payload.get('messages'))
    inputs = [gen_ai.get('input'), attributes.get('gen_ai.input'), payload.get('input')]
    outputs = [gen_ai.get('output'), attributes.get('gen_ai.output'), payload.get('output')]
    messages = (
        direct
        if _usable(direct)
        else next((parsed for value in inputs if (parsed := _normalised_messages(value, default_role='user'))), [])
    )
    output_messages = next(
        (parsed for value in outputs if (parsed := _normalised_messages(value, default_role='assistant'))), []
    )
    for message in output_messages:
        if not _message_has_content(message):
            continue
        if not messages or message != messages[-1]:
            messages.append(message)

    # Nested span-detail envelopes retain source parts for consumers that inspect
    # multimodal payloads; the normalized text remains available as ``content``.
    for value in (*inputs, *outputs):
        decoded = _plain(value)
        if isinstance(decoded, str):
            try:
                decoded = json.loads(decoded)
            except json.JSONDecodeError:
                continue
        source_messages = decoded.get('messages') if isinstance(decoded, Mapping) else None
        if not isinstance(source_messages, list):
            continue
        for source in source_messages:
            if not isinstance(source, Mapping):
                continue
            source_parts = source.get('parts')
            if not isinstance(source_parts, list):
                continue
            role = source.get('role')
            text = ''.join(
                str(part.get('content') or part.get('text') or '')
                for part in source_parts
                if isinstance(part, Mapping) and part.get('type') != 'reasoning'
            )
            for message in messages:
                if message.get('role') == role and message.get('content') == text:
                    message['parts'] = list(source_parts)
                    if source_parts and all(isinstance(part, Mapping) and 'content' in part for part in source_parts):
                        message.pop('content', None)
                    break
    return messages


def _summary_output_tokens(summary: Any) -> Any:
    """Read completion tokens from typed SDK summaries or their compact row projection."""
    output_tokens = _field(_field(summary, 'usage'), 'completion_tokens')
    return output_tokens if output_tokens is not None else _field(summary, 'tokens_out')


def _has_reply(messages: list[dict[str, Any]]) -> bool:
    return next(
        (
            message.get('role') in {'assistant', 'tool'}
            for message in reversed(messages)
            if _message_has_content(message)
        ),
        False,
    )


def _normalised_messages(
    value: Any, *, default_role: Literal['user', 'assistant'] | None = None
) -> list[dict[str, Any]]:
    message_format = detect_message_format(value)
    mixed_responses = _mixed_chat_parts_and_responses(value, default_role=default_role)
    if mixed_responses is not None:
        parsed, _ = parse_messages(mixed_responses, hinted='responses', default_role=default_role or 'user')
        return [message.to_chat_completion() for message in parsed]
    if message_format == 'chat_completions' or _has_chat_text_parts(value):
        messages = _message_list(value, default_role=default_role)
        # Chat shaped payloads can still carry OTel-style ``parts`` instead of
        # ``content``. Keep their text usable by downstream chat consumers.
        for message in messages:
            parts = message.get('parts')
            if message.get('content') is None and isinstance(parts, list):
                text = [
                    part_text
                    for part in parts
                    if isinstance(part, Mapping) and isinstance(part_text := part.get('text'), str)
                ]
                if text:
                    message['content'] = ''.join(text)
        return messages
    parsed, _ = parse_messages(value, hinted=message_format, default_role=default_role or 'user')
    messages = [message.to_chat_completion() for message in parsed]
    if message_format == 'otel_genai':
        _attach_otel_tool_result_metadata(messages, value)
    return messages


def _mixed_chat_parts_and_responses(
    value: Any, *, default_role: Literal['user', 'assistant'] | None
) -> list[dict[str, Any]] | None:
    """Bridge text-only chat parts when a payload also contains Responses items."""
    decoded = _plain(value)
    if isinstance(decoded, str):
        try:
            decoded = json.loads(decoded)
        except json.JSONDecodeError:
            return None
    if isinstance(decoded, Mapping):
        sides = ('output', 'input') if default_role == 'assistant' else ('input', 'output')
        decoded = next((decoded[key] for key in ('messages', *sides) if isinstance(decoded.get(key), list)), None)
    if not isinstance(decoded, list) or not _has_chat_text_parts(decoded):
        return None
    if not any(isinstance(item, Mapping) and is_responses_item(dict(item)) for item in decoded):
        return None

    prepared: list[dict[str, Any]] = []
    for item in decoded:
        if not isinstance(item, Mapping):
            continue
        message = dict(item)
        parts = message.get('parts')
        if message.get('content') is None and isinstance(parts, list):
            text: list[str] = [
                part_text
                for part in parts
                if isinstance(part, Mapping) and isinstance(part_text := part.get('text'), str)
            ]
            if text:
                message['content'] = ''.join(text)
        prepared.append(message)
    return prepared


def _attach_otel_tool_result_metadata(messages: list[dict[str, Any]], value: Any) -> None:
    """Keep explicit OTel tool outcomes in trace-only metadata for projection."""
    decoded = _plain(value)
    if isinstance(decoded, str):
        try:
            decoded = json.loads(decoded)
        except json.JSONDecodeError:
            return
    if not isinstance(decoded, (dict, list)):
        return

    outcomes: dict[str, dict[str, Any]] = {}
    pending = [decoded]
    while pending:
        item = pending.pop()
        if isinstance(item, list):
            pending.extend(item)
            continue
        if not isinstance(item, dict):
            continue
        parts = item.get('parts')
        if isinstance(parts, list):
            for part in parts:
                if not isinstance(part, dict) or part.get('type') != 'tool_call_response':
                    continue
                call_id = part.get('id') or part.get('call_id')
                if not isinstance(call_id, str) or not call_id:
                    continue
                explicit = {key: part[key] for key in ('status', 'is_error', 'error') if key in part}
                if explicit:
                    outcomes[call_id] = explicit
        pending.extend(child for key, child in item.items() if key != 'parts')

    for message in messages:
        call_id = message.get('tool_call_id')
        outcome = outcomes.get(call_id) if isinstance(call_id, str) else None
        if outcome:
            message['trace_finder_metadata'] = {'tool_result': outcome}


def _has_chat_text_parts(value: Any) -> bool:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return False
    if isinstance(value, dict):
        value = value.get('messages', [value])
    if not isinstance(value, list):
        return False
    messages_with_parts = [message for message in value if isinstance(message, dict) and 'parts' in message]
    return bool(messages_with_parts) and all(
        isinstance(message.get('parts'), list)
        and all(
            isinstance(part, dict) and isinstance(part.get('text'), str) and 'content' not in part
            for part in message['parts']
        )
        for message in messages_with_parts
    )


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
    if value.get('type') == 'reasoning':
        return False
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


def _span_tool_call_id(span: Any) -> str | None:
    raw = _plain(span)
    if not isinstance(raw, Mapping):
        return None
    attributes = raw.get('attributes')
    if not isinstance(attributes, Mapping):
        return None
    gen_ai = attributes.get('gen_ai')
    tool = gen_ai.get('tool') if isinstance(gen_ai, Mapping) else None
    call = tool.get('call') if isinstance(tool, Mapping) else None
    candidates = [
        attributes.get('gen_ai.tool.call.id'),
        attributes.get('tool_call_id'),
        attributes.get('call_id'),
        gen_ai.get('tool.call.id') if isinstance(gen_ai, Mapping) else None,
        call.get('id') if isinstance(call, Mapping) else None,
    ]
    return next((value for value in candidates if isinstance(value, str) and value), None)


def _selected_span_signal_data(span: Any, messages: Sequence[dict[str, Any]]) -> dict[str, Any]:  # noqa: C901 -- correlate optional raw fields and output IDs in one bounded span payload.
    """Extract source payload, usage, tools and timing correlated to this selected LLM span."""
    raw = _plain(span)
    attributes = _mapping(_field(raw, 'attributes'))
    gen_ai = _mapping(attributes.get('gen_ai'))
    tool = _mapping(gen_ai.get('tool'))
    definitions = tool.get('definitions') or attributes.get('gen_ai.tool.definitions')
    request_candidates = [
        attributes.get('openresponses.request'),
        attributes.get('orq.openresponses.request'),
        _field(_mapping(attributes.get('openresponses')), 'request'),
        _field(_mapping(_field(_mapping(attributes.get('orq')), 'openresponses')), 'request'),
    ]
    for candidate in request_candidates:
        decoded = _plain(candidate)
        if isinstance(decoded, str):
            try:
                decoded = json.loads(decoded)
            except json.JSONDecodeError:
                continue
        if isinstance(decoded, Mapping):
            if isinstance(decoded.get('tools'), list):
                definitions = decoded['tools']
                break
    usage = _mapping(_field(raw, 'usage') or gen_ai.get('usage'))
    input_details = _mapping(usage.get('input_tokens_details') or usage.get('prompt_tokens_details'))
    prompt_tokens = _first_number(usage, 'input_tokens', 'prompt_tokens')
    completion_tokens = _first_number(usage, 'output_tokens', 'completion_tokens')
    cached_tokens = _first_number(input_details, 'cached_tokens', 'cache_read_input_tokens')
    cost = _first_number(usage, 'cost_usd', 'cost')
    metrics = {
        key: value
        for key, value in {
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
            'cached_tokens': cached_tokens,
            'cost_usd': cost,
        }.items()
        if value is not None
    }
    start = _parse_time(_field(raw, 'started_at') or _field(raw, 'start_time'))
    end = _parse_time(_field(raw, 'ended_at') or _field(raw, 'end_time'))
    candidates = [
        attributes.get('openresponses.response'),
        attributes.get('orq.openresponses.response'),
        _field(_mapping(attributes.get('openresponses')), 'response'),
        _field(_mapping(_field(_mapping(attributes.get('orq')), 'openresponses')), 'response'),
    ]
    items: list[dict[str, Any]] = []
    for candidate in candidates:
        decoded = _plain(candidate)
        if isinstance(decoded, str):
            try:
                decoded = json.loads(decoded)
            except json.JSONDecodeError:
                continue
        if isinstance(decoded, Mapping):
            response_usage = _mapping(decoded.get('usage'))
            if response_usage and not usage:
                usage = response_usage
            decoded = decoded.get('output')
        if isinstance(decoded, list):
            items = [dict(item) for item in decoded if isinstance(item, Mapping)]
            if items:
                break

    # Correlate output items only by a raw Responses call ID or exact output text.
    # Ambiguous items remain captured as span evidence with coverage marked false.
    matched_orders: set[int] = set()
    item_sidecars: dict[str, list[dict[str, Any]]] = {}
    unmapped_items: list[dict[str, Any]] = []
    for item in items:
        item_type = item.get('type')
        call_id = item.get('call_id')
        candidates_order: set[int] = set()
        if isinstance(call_id, str) and call_id:
            for index, message in enumerate(messages):
                if message.get('role') == 'assistant' and any(
                    isinstance(call, Mapping) and (call.get('id') or call.get('call_id')) == call_id
                    for call in message.get('tool_calls') or []
                ):
                    candidates_order.add(index)
        elif item_type == 'message':
            text = _response_item_text(item)
            for index, message in enumerate(messages):
                if (
                    message.get('role') == 'assistant'
                    and isinstance(message.get('content'), str)
                    and text == message['content']
                ):
                    candidates_order.add(index)
        if len(candidates_order) != 1:
            if item_type != 'message':
                unmapped_items.append(item)
            continue
        order = next(iter(candidates_order))
        matched_orders.add(order)
        # Custom/MCP items carry fields ATIF cannot model; keep their source payload
        # beside the correlated step. Standard calls are already represented natively.
        if item_type in {'custom_tool_call', 'custom_tool_call_output', 'mcp_call'} or (
            isinstance(item_type, str) and item_type.startswith('orq:')
        ):
            item_sidecars.setdefault(str(order), []).append(item)
        elif item_type in {'function_call', 'function_call_output', 'message'}:
            item_sidecars.setdefault(str(order), []).append({
                key: item[key]
                for key in ('id', 'call_id', 'type', 'name', 'status', 'error_type', 'finish_reason')
                if key in item
            })
    selected_order = next(iter(matched_orders)) if len(matched_orders) == 1 else None
    input_details = _mapping(usage.get('input_tokens_details') or usage.get('prompt_tokens_details'))
    metrics = {
        key: value
        for key, value in {
            'prompt_tokens': _first_number(usage, 'input_tokens', 'prompt_tokens'),
            'completion_tokens': _first_number(usage, 'output_tokens', 'completion_tokens'),
            'cached_tokens': _first_number(input_details, 'cached_tokens', 'cache_read_input_tokens'),
            'cost_usd': _first_number(usage, 'cost_usd', 'cost'),
        }.items()
        if value is not None
    }
    result: dict[str, Any] = {
        'step_order': selected_order,
        'output_items_by_order': item_sidecars,
        'unmapped_output_items': unmapped_items,
        'output_item_count': len(items),
        'tool_definitions': definitions if isinstance(definitions, list) else [],
        'tool_definitions_present': isinstance(definitions, list),
    }
    if metrics:
        result['metrics'] = metrics
    if start is not None:
        result['started_at'] = start.isoformat()
        result['start_timestamp'] = start.timestamp()
    if end is not None:
        result['ended_at'] = end.isoformat()
        result['end_timestamp'] = end.timestamp()
    return result


def _response_item_text(item: Mapping[str, Any]) -> str | None:
    content = item.get('content')
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            part.get('text') or part.get('refusal')
            for part in content
            if isinstance(part, Mapping) and isinstance(part.get('text') or part.get('refusal'), str)
        ]
        return ''.join(parts) if parts else None
    return None


def _first_number(mapping: Mapping[str, Any], *names: str) -> int | float | None:
    for name in names:
        value = mapping.get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return value
    return None


def _signal_status(span: Any) -> dict[str, Any]:
    raw = _plain(span)
    attributes = _mapping(_field(raw, 'attributes'))
    status = next(
        (
            value
            for value in (
                attributes.get('gen_ai.tool.call.status'),
                attributes.get('tool.status'),
                attributes.get('status'),
            )
            if isinstance(value, str) and value.strip()
        ),
        None,
    )
    result: dict[str, Any] = {}
    start = _parse_time(_field(raw, 'started_at') or _field(raw, 'start_time'))
    end = _parse_time(_field(raw, 'ended_at') or _field(raw, 'end_time'))
    if start is not None:
        result['start_timestamp'] = start.timestamp()
    if end is not None:
        result['end_timestamp'] = end.timestamp()
    if is_error_span(raw):
        result['status'] = 'error'
        error_type = attributes.get('error.type') or attributes.get('error_type')
        if isinstance(error_type, str) and error_type:
            result['error_type'] = error_type
        return result
    if status:
        lowered = status.casefold()
        result['status'] = 'error' if 'error' in lowered or 'fail' in lowered else status
    return result


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


def _tool_definition_size(*sources: Any) -> tuple[int, int]:
    for source in sources:
        attributes = _mapping(_field(source, 'attributes'))
        gen_ai = _mapping(attributes.get('gen_ai'))
        definitions = _mapping(gen_ai.get('tool')).get('definitions') or attributes.get('gen_ai.tool.definitions')
        if isinstance(definitions, list) and definitions:
            serialized = json.dumps(definitions, ensure_ascii=False, separators=(',', ':'), default=str)
            return len(definitions), math.ceil(len(serialized) / 4)
    return 0, 0


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


def _aware_bound(value: datetime, name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise OrqSourceError(f'{name} must include a timezone offset')
    return value.astimezone(timezone.utc)


def _check_window(start: datetime | None, end: datetime | None, limit: int) -> tuple[datetime, datetime]:
    """Validate a trace search window and return its resolved UTC bounds."""
    if limit < 1:
        raise OrqSourceError('limit must be at least 1')
    resolved_end = _aware_bound(end, 'end') if end is not None else datetime.now(timezone.utc)
    resolved_start = _aware_bound(start, 'start') if start is not None else resolved_end - DEFAULT_LOOKBACK
    if resolved_start > resolved_end:
        raise OrqSourceError('start must not be after end')
    return resolved_start, resolved_end


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
