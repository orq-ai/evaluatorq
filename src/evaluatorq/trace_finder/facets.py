"""Load a defensive, live catalogue of Orq trace facet values."""

from __future__ import annotations

import asyncio
import hashlib
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Mapping
from datetime import datetime, timedelta, timezone
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from loguru import logger

from .models import FACET_NAMES, FacetCatalogue, FacetName

if TYPE_CHECKING:
    from orq_ai_sdk import Orq


_FACET_FIELDS: Mapping[FacetName, str] = MappingProxyType({
    'project': 'project_id',
    'model': 'model',
    'provider': 'provider',
    'status': 'status',
    'product': 'product',
    'trace_type': 'attributes.orq.leading_span.span_type',
    'agent_name': 'agent_name',
    'tool_name': 'tool_name',
})
if set(_FACET_FIELDS) != set(FACET_NAMES):
    raise RuntimeError('trace facet field mapping is out of sync with FACET_NAMES')

_CACHE_TTL = 300.0
_CACHE_MAX_ENTRIES = 32
_catalogue_cache: OrderedDict[tuple[Any, ...], tuple[float, FacetCatalogue]] = OrderedDict()
_catalogue_inflight: dict[tuple[Any, ...], asyncio.Task[FacetCatalogue]] = {}
_cache_lock: asyncio.Lock | None = None
_cache_lock_loop: asyncio.AbstractEventLoop | None = None


def _get_cache_lock() -> asyncio.Lock:
    """Use a lock scoped to the current event loop, never reusing loop-bound state."""
    global _cache_lock, _cache_lock_loop
    loop = asyncio.get_running_loop()
    if _cache_lock is None or _cache_lock_loop is not loop:
        _cache_lock = asyncio.Lock()
        _cache_lock_loop = loop
        _catalogue_inflight.clear()
    return _cache_lock


def _credential_fingerprint(client: Orq) -> str:
    """Fingerprint configured authentication without retaining or exposing its secret."""
    config = client.sdk_configuration
    auth = config.security
    if callable(auth):
        auth = auth()
    security = getattr(auth, 'api_key', None)
    if security is None and isinstance(auth, Mapping):
        security = auth.get('api_key')
    secret = (
        security
        if isinstance(security, str)
        else (
            security.get('api_key') or security.get('token')
            if isinstance(security, Mapping)
            else getattr(security, 'api_key', None) or getattr(security, 'token', None)
        )
    )
    return hashlib.sha256(str(secret or '').encode()).hexdigest()


def _cache_key(client: Orq, start: datetime, end: datetime, limit: int) -> tuple[Any, ...]:
    config = client.sdk_configuration
    http_client = config.async_client
    headers = getattr(http_client, 'headers', {})
    scope = tuple(
        sorted(
            (str(key).casefold(), str(value))
            for key, value in headers.items()
            if str(key).casefold() in {'x-orq-workspace-id', 'x-orq-project-id'}
        )
    )
    return (
        config.get_server_details()[0],
        _credential_fingerprint(client),
        scope,
        start.isoformat(),
        end.isoformat(),
        limit,
    )


def facet_catalogue_cache_hit(
    client: Orq, start: datetime, end: datetime, limit: int = 50, *, default_window: bool = False
) -> bool:
    """Report whether an unexpired cache entry exists for this request."""
    if default_window:
        key = (
            *_cache_key(client, start, end, limit)[:3],
            'rolling-default',
            round((end - start).total_seconds()),
            limit,
        )
    else:
        key = _cache_key(client, start, end, limit)
    cached = _catalogue_cache.get(key)
    return cached is not None and time.monotonic() - cached[0] < _CACHE_TTL


async def load_facet_catalogue_cached(
    client: Orq,
    start: datetime,
    end: datetime,
    limit: int = 50,
    *,
    default_window: bool = False,
    now: datetime | None = None,
    loader: Callable[..., Awaitable[FacetCatalogue]] | None = None,
) -> FacetCatalogue:
    """Load all returned facet values from a bounded five-minute, single-flight cache."""
    if default_window:
        anchor = now or datetime.now(timezone.utc)
        stable_start = anchor - timedelta(days=1)
        key = (
            *_cache_key(client, stable_start, anchor, limit)[:3],
            'rolling-default',
            round((end - start).total_seconds()),
            limit,
        )
    else:
        key = _cache_key(client, start, end, limit)
    lock = _get_cache_lock()
    async with lock:
        cached = _catalogue_cache.get(key)
        if cached is not None and time.monotonic() - cached[0] < _CACHE_TTL:
            _catalogue_cache.move_to_end(key)
            return cached[1].model_copy(deep=True)
        task = _catalogue_inflight.get(key)
        if task is None:
            load = loader or load_facet_catalogue
            task = asyncio.ensure_future(load(client, start=start, end=end, limit=limit))
            _catalogue_inflight[key] = task
    try:
        catalogue = await asyncio.shield(task)
    except BaseException:
        if task.done():
            async with lock:
                if _catalogue_inflight.get(key) is task:
                    del _catalogue_inflight[key]
        raise
    async with lock:
        if _catalogue_inflight.get(key) is task:
            _catalogue_inflight.pop(key)
            _catalogue_cache[key] = (time.monotonic(), catalogue.model_copy(deep=True))
            _catalogue_cache.move_to_end(key)
            while len(_catalogue_cache) > _CACHE_MAX_ENTRIES:
                _catalogue_cache.popitem(last=False)
    return catalogue.model_copy(deep=True)


async def invalidate_facet_catalogue_cache() -> None:
    """Invalidate successful entries and cancel outstanding cache loads."""
    lock = _get_cache_lock()
    async with lock:
        _catalogue_cache.clear()
        tasks = tuple(_catalogue_inflight.values())
        _catalogue_inflight.clear()
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


async def close_facet_catalogue_cache() -> None:
    """Close the process-local cache; intended for runtime shutdown."""
    await invalidate_facet_catalogue_cache()


async def load_facet_catalogue(
    client: Orq,
    *,
    start: datetime,
    end: datetime,
    limit: int = 50,
) -> FacetCatalogue:
    """Load available facet values, retaining a bounded frequent subset on overflow."""

    tasks = [
        asyncio.create_task(_safe_facet_values(client, field, start=start, end=end, limit=limit))
        for field in _FACET_FIELDS.values()
    ]
    try:
        results = await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    values_by_name = {name: values for name, (values, _, _) in zip(_FACET_FIELDS, results, strict=True)}
    truncated_facets = frozenset(
        name for name, (_, truncated, _) in zip(_FACET_FIELDS, results, strict=True) if truncated
    )
    value_counts = {
        name: counts for name, (_, _, counts) in zip(_FACET_FIELDS, results, strict=True) if counts is not None
    }
    if truncated_facets:
        logger.warning(
            'Orq facet values exceed the {}-value limit for {}; using returned values ranked by frequency',
            limit,
            ', '.join(sorted(truncated_facets)),
        )

    project_names = await _project_names(client)
    missing_projects = set(values_by_name['project']) - set(project_names)
    if missing_projects:
        raise ValueError(f'cannot resolve facet project ids: {sorted(missing_projects)}')
    labels = project_labels(project_names)
    values_by_name['project'] = tuple(labels[project_id] for project_id in values_by_name['project'])
    if 'project' in value_counts:
        value_counts['project'] = {labels[project_id]: count for project_id, count in value_counts['project'].items()}
    return FacetCatalogue.model_validate({
        **values_by_name,
        'truncated_facets': truncated_facets,
        'value_counts': value_counts,
    })


def project_labels(project_names: Mapping[str, str]) -> dict[str, str]:
    """Disambiguate equal project names using their stable IDs."""
    counts: dict[str, int] = {}
    for name in project_names.values():
        counts[name] = counts.get(name, 0) + 1
    return {
        project_id: f'{name} ({project_id})' if counts[name] > 1 else name for project_id, name in project_names.items()
    }


async def _safe_facet_values(
    client: Orq,
    field: str,
    *,
    start: datetime,
    end: datetime,
    limit: int,
) -> tuple[tuple[str, ...], bool, dict[str, int] | None]:
    """Return values ranked by trace count, the overflow flag, and each value's count.

    The counts are ``None`` unless Orq reported a count for every value it returned.
    """
    response = await client.traces.list_facet_values_async(field=field, from_=start, to=end, limit=limit)
    raw_values = _read(response, 'values')
    if not isinstance(raw_values, (list, tuple)):
        raise TypeError(f'trace facet field {field} returned no readable values')
    ranked: list[tuple[str, int, int]] = []
    counted = True
    for index, item in enumerate(raw_values):
        value = _read(item, 'value')
        if not isinstance(value, str) or not value or value == 'unknown':
            continue
        count = _read(item, 'count')
        valid = type(count) is int and count >= 0
        counted = counted and valid
        ranked.append((value, count if valid else 0, index))
    ranked.sort(key=lambda item: (-item[1], item[2]))
    counts: dict[str, int] = {}
    for value, count, _ in ranked:
        counts.setdefault(value, count)
    return tuple(counts), bool(_read(response, 'has_more')), counts if counted and counts else None


async def _project_names(client: Orq) -> dict[str, str]:
    """Resolve every visible project id to its human-readable name."""

    projects: dict[str, str] = {}
    starting_after: str | None = None
    seen_cursors: set[str] = set()
    while True:
        response = await client.projects.list_async(limit=100, starting_after=starting_after)
        page = _read(response, 'data')
        page_items = list(page) if isinstance(page, (list, tuple)) else []
        for project in page_items:
            project_id = _read(project, 'project_id')
            name = _read(project, 'name')
            if project_id and name:
                projects[str(project_id)] = str(name)
        if not _read(response, 'has_more'):
            return projects
        if not page_items:
            raise RuntimeError('project response reported more pages without projects')
        cursor = str(_read(page_items[-1], 'project_id') or '')
        if not cursor or cursor in seen_cursors:
            raise RuntimeError(f'repeated project page cursor {cursor!r}')
        seen_cursors.add(cursor)
        starting_after = cursor


def _read(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)
